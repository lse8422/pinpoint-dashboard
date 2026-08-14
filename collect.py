# 구글 뉴스 RSS에서 화성시 기사를 매일 수집해 news_live.json에 누적하는 스크립트 (검증본 news_data.json은 읽기만 한다)
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))

BASE = os.path.dirname(os.path.abspath(__file__))
VERIFIED_FILE = os.path.join(BASE, "news_data.json")   # 빅카인즈 검증본 — 절대 쓰지 않는다
LIVE_FILE = os.path.join(BASE, "news_live.json")       # 자동 수집분 — 여기에만 누적한다

QUERIES = ["화성시", "동탄"]     # 두 검색어로 나눠 받아 누락을 줄인다
# 평소에는 이틀치만 받는다. 빈 기간을 메울 때만 워크플로에서 값을 넘겨 늘린다.
DAYS = int(os.environ.get("COLLECT_DAYS") or 2)
MAX_NEW = int(os.environ.get("COLLECT_MAX") or 100)   # 한 번에 처리할 최대 건수 (비용·오염 안전장치)
BATCH = 15                       # AI 분류 시 한 번에 보낼 기사 수
MODEL = "claude-haiku-4-5-20251001"

CATS = ["교통", "환경", "행정", "문화관광", "산업경제", "안전", "복지"]

# 화성시 기사임을 확인해주는 단서. 하나라도 있어야 통과시킨다.
# "화성"만 있는 제목도 통과시키되, 아래 EXCLUDES를 먼저 걸러 동음이의어를 막는다.
HINTS = [
    "화성", "정명근",
    "동탄", "병점", "향남", "봉담", "남양", "정남", "매송", "비봉", "마도",
    "송산", "서신", "팔탄", "장안", "우정", "양감", "새솔", "기배", "진안",
    "융건릉", "제부도", "궁평", "전곡항", "우음도", "화옹지구",
]

# 동음이의어를 걸러낸다. HINTS보다 먼저 검사한다.
# 특히 수원 화성(화성행궁·수원화성)은 수원시 관광지라 반드시 막아야 한다.
EXCLUDES = [
    # 수원 화성 (문화유산)
    "수원화성", "수원 화성", "화성행궁", "화성행렬", "화성열차", "화성어차",
    "화성성역", "정조대왕", "화성유수부",
    # 행성 화성
    "화성 탐사", "화성탐사", "화성 이주", "화성인", "화성 착륙", "적색행성",
    "NASA", "나사", "스페이스X", "스페이스엑스", "머스크", "로버", "큐리오시티",
    "퍼서비어런스", "화성 진출", "테라포밍", "화성 생명체",
    # 그 밖의 동음이의어
    "화성암", "화성학",
]

# 분야 분류용 키워드. 가장 많이 맞은 분야로 정한다.
CAT_KEYWORDS = {
    "교통": ["도로", "철도", "GTX", "전철", "지하철", "버스", "교통", "주차", "교차로",
             "신호등", "트램", "노선", "개통", "정체", "고속도로", "인도", "자전거",
             "역세권", "환승", "통행", "차량", "운전", "보행"],
    "환경": ["환경", "미세먼지", "폐기물", "쓰레기", "하천", "오염", "악취", "재활용",
             "탄소", "태양광", "생태", "녹지", "매립", "소각", "수질", "대기질",
             "습지", "갯벌", "미세플라스틱", "정화", "배출"],
    "행정": ["시장", "시청", "시의회", "조례", "예산", "행정", "정책", "공무원", "민원",
             "인사", "시정", "특례시", "청사", "국정감사", "감사", "협약", "위원회",
             "공청회", "브리핑", "간담회", "국회", "도의회", "선거"],
    "문화관광": ["축제", "공연", "문화", "관광", "박물관", "도서관", "전시", "체육",
                 "스포츠", "축구", "예술", "음악회", "행사", "페스티벌", "관광객",
                 "유적", "문화재", "캠핑", "여행", "야구", "마라톤"],
    "산업경제": ["기업", "산업", "투자", "공장", "반도체", "일자리", "창업", "소상공인",
                 "상권", "경제", "유치", "산업단지", "부동산", "분양", "아파트", "개발",
                 "물류", "매출", "수출", "고용", "임대", "상가", "재개발"],
    "안전": ["사고", "화재", "안전", "범죄", "경찰", "소방", "구조", "재난", "침수",
             "폭우", "태풍", "붕괴", "사망", "부상", "실종", "단속", "검찰", "구속",
             "폭염", "지진", "누출", "대피", "응급", "119"],
    "복지": ["복지", "어르신", "노인", "아동", "청소년", "보육", "어린이집", "장애인",
             "건강", "의료", "보건", "지원금", "돌봄", "여성", "다문화", "급식",
             "교육", "학교", "학생", "병원", "接종", "상담", "취약계층"],
}

NEG_WORDS = ["사고", "화재", "사망", "부상", "논란", "갈등", "반발", "항의", "시위",
             "비판", "우려", "피해", "적발", "위반", "불법", "구속", "기소", "수사",
             "고발", "파업", "중단", "지연", "무산", "부실", "실패", "하락", "감소",
             "붕괴", "침수", "오염", "악취", "불편", "폐쇄", "취소", "소송", "의혹",
             "비리", "횡령", "체납", "누락", "결함", "폭행", "실종", "경고", "위기"]

POS_WORDS = ["개통", "준공", "완공", "유치", "선정", "수상", "우수", "최우수", "1위",
             "확대", "증가", "성과", "협약", "체결", "지원", "개최", "성공", "돌파",
             "최초", "기록", "호평", "만족", "개선", "해소", "승격", "출범", "착공",
             "표창", "인증", "달성", "혁신", "활성화", "무료", "장학", "기부", "감사패"]


def fetch_rss(query, days):
    """구글 뉴스 RSS에서 최근 기사 목록을 가져온다. API 키가 필요 없다."""
    q = urllib.parse.quote("%s when:%dd" % (query, days))
    url = "https://news.google.com/rss/search?q=%s&hl=ko&gl=KR&ceid=KR:ko" % q
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as res:
        root = ET.fromstring(res.read())

    items = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        source = (it.findtext("source") or "").strip()
        # 구글 뉴스 제목은 "기사제목 - 언론사" 형태라 언론사를 떼어낸다
        if source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)].strip()

        pub = it.findtext("pubDate") or ""
        try:
            dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
            date = dt.astimezone(KST).strftime("%Y-%m-%d")
        except ValueError:
            date = datetime.now(KST).strftime("%Y-%m-%d")

        if title:
            items.append({
                "date": date,
                "title": title,
                "url": (it.findtext("link") or "").strip(),
                "source": source,
            })
    return items


def norm(s):
    """제목 비교용 정규화 — 공백·특수문자를 지워 같은 기사를 걸러낸다."""
    return re.sub(r"[^\w가-힣]", "", s)


def is_hwaseong(title):
    """화성시 기사인지 판단한다. 단서가 있어야만 통과시킨다(정밀도 우선)."""
    for x in EXCLUDES:
        if x in title:
            return False
    for h in HINTS:
        if h in title:
            return True
    return False


def classify_by_rule(title):
    """키워드 사전으로 분야와 감정을 정한다. AI 키가 없을 때 쓰는 대체 경로다."""
    best, best_hit = "행정", 0
    for cat, words in CAT_KEYWORDS.items():
        hit = sum(1 for w in words if w in title)
        if hit > best_hit:
            best, best_hit = cat, hit

    neg = sum(1 for w in NEG_WORDS if w in title)
    pos = sum(1 for w in POS_WORDS if w in title)
    if neg > pos:
        senti = "부정"
    elif pos > neg:
        senti = "긍정"
    else:
        senti = "중립"

    # 제목에서 조사·형식어를 뺀 명사 후보를 키워드로 삼는다
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", title)
    stop = {"영상", "포토", "종합", "속보", "단독", "인터뷰", "기고", "사설", "화보"}
    kw = []
    for t in tokens:
        if t in stop or t in kw:
            continue
        kw.append(t)
        if len(kw) == 3:
            break

    return {"cat": best, "senti": senti, "summary": "", "kw": kw}


def call_claude(api_key, articles):
    """기사 묶음을 Claude에 보내 분류·요약·키워드를 한 번에 받는다."""
    listing = "\n".join("%d. %s" % (i + 1, a["title"]) for i, a in enumerate(articles))
    prompt = """아래는 '화성시' 검색으로 수집한 뉴스 제목 목록이다. 각 기사를 분석해 JSON 배열로만 답하라.

%s

각 항목에 대해 다음 형식으로 출력한다.
{"n": 번호, "keep": true/false, "cat": "분야", "senti": "긍정|중립|부정", "summary": "한 문장 요약", "kw": ["키워드1","키워드2","키워드3"]}

규칙:
- keep: 경기도 화성시(동탄·병점·향남·봉담·남양 등 포함)의 행정·생활과 직접 관련된 기사만 true. 경기도 전체 정치, 타 지역, 중앙 정치, 행성 화성 관련은 false.
- cat: %s 중 하나.
- senti: 화성시 입장에서의 긍정·중립·부정.
- summary: 제목을 바탕으로 한 문장(50자 내외) 요약. 사실만 담고 추측하지 않는다.
- kw: 핵심 주제어 3개. 조사가 붙은 단어("동탄은")나 형식어("영상","포토")는 금지.
- keep이 false면 cat/senti/summary/kw는 빈 값으로 둔다.
설명 없이 JSON 배열만 출력하라.""" % (listing, " / ".join(CATS))

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        payload = json.load(res)

    text = payload["content"][0]["text"]
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("응답에서 JSON 배열을 찾지 못했습니다: " + text[:200])
    return json.loads(m.group())


def brief_facts(rows, label):
    """브리핑에 쓸 수치를 코드가 직접 계산한다. AI에게 숫자를 맡기지 않기 위해서다."""
    total = len(rows)
    if not total:
        return None
    neg = [r for r in rows if r.get("senti") == "부정"]
    cats = Counter(r["cat"] for r in rows)
    kws = Counter(k for r in rows for k in (r.get("kw") or []))
    negcats = Counter(r["cat"] for r in neg)
    negdates = Counter(r["date"] for r in neg)
    top_cat = cats.most_common(1)[0] if cats else None
    top_kw = kws.most_common(1)[0] if kws else None
    top_negcat = negcats.most_common(1)[0] if negcats else None
    peak = negdates.most_common(1)[0] if negdates else None
    return {
        "기간": label,
        "전체기사": total,
        "부정기사": len(neg),
        "부정비율": "%d%%" % round(len(neg) / total * 100),
        "최다분야": "%s %d건" % top_cat if top_cat else None,
        "핵심키워드": top_kw[0] if top_kw else None,
        "부정최다분야": "%s %d건" % top_negcat if top_negcat else None,
        "부정집중일": "%s %d건" % peak if peak and peak[1] >= 2 else None,
    }


def write_briefs(api_key, facts):
    """계산된 수치를 넘겨 기간별 브리핑 문장을 받는다. 실패하면 None을 돌려 화면이 기존 방식으로 돌아가게 한다."""
    prompt = """아래는 화성시 언론보도 대시보드의 기간별 집계 결과다.
각 기간마다 화성시 홍보 담당 공무원이 아침에 읽을 브리핑을 2~3문장으로 작성하라.

%s

규칙:
- 주어진 수치만 사용한다. 새로운 숫자를 만들거나 추정하지 않는다.
- 담당자가 무엇을 먼저 확인해야 하는지가 드러나게 쓴다.
- 기간마다 문장 구조를 다르게 쓴다. 같은 틀에 숫자만 바꿔 넣지 마라.
- 강조할 표현은 [[b]]와 [[/b]]로, 부정 보도 관련 수치는 [[w]]와 [[/w]]로 감싼다.
- HTML 태그를 쓰지 마라. 위 표시자만 쓴다.
- 과장하거나 추측하지 않는다. 사실만 담담하게 쓴다.

{"all": "전체 기간 브리핑", "7": "최근 7일 브리핑", "30": "최근 30일 브리핑"}
위 형식의 JSON 객체로만 답하라.""" % json.dumps(facts, ensure_ascii=False, indent=2)

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={"content-type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        payload = json.load(res)
    text = payload["content"][0]["text"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("브리핑 응답에서 JSON을 찾지 못했습니다: " + text[:200])
    out = json.loads(m.group())
    return {k: v for k, v in out.items() if k in ("all", "7", "30") and isinstance(v, str) and v.strip()}


def load_live():
    """기존 수집분과 브리핑을 읽는다. 파일이 없거나 깨졌으면 빈 상태로 시작한다."""
    if not os.path.exists(LIVE_FILE):
        return [], None
    try:
        with open(LIVE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("items", []), data.get("briefs")
        return [], None
    except (ValueError, OSError) as e:
        print("기존 news_live.json을 읽지 못해 새로 시작합니다: %s" % e)
        return [], None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    mode = "AI 분류" if api_key else "키워드 규칙 분류"
    print("== 화성시 뉴스 자동 수집 (%s) ==" % mode)

    # 검증본은 중복 판단에만 쓰고 절대 수정하지 않는다
    with open(VERIFIED_FILE, encoding="utf-8") as f:
        verified = json.load(f)
    live, prev_briefs = load_live()
    seen = {norm(a["title"]) for a in verified} | {norm(a["title"]) for a in live}
    print("검증본 %d건 / 기존 수집분 %d건" % (len(verified), len(live)))

    # 수집
    fetched = []
    for q in QUERIES:
        try:
            got = fetch_rss(q, DAYS)
            print("  '%s' 검색 %d건" % (q, len(got)))
            fetched.extend(got)
        except Exception as e:
            print("  '%s' 검색 실패: %s" % (q, e))

    # 중복 제거 + 화성시 무관 기사 제외
    fresh, dropped = [], []
    for a in fetched:
        key = norm(a["title"])
        if key in seen:
            continue
        seen.add(key)
        if not is_hwaseong(a["title"]):
            dropped.append(a["title"])
            continue
        fresh.append(a)

    print("신규 %d건 / 화성시 무관으로 제외 %d건" % (len(fresh), len(dropped)))
    for t in dropped[:20]:
        print("    제외: %s" % t)

    if not fresh:
        # 새 기사가 없어도 브리핑은 다시 만든다. "최근 7일" 구간이 날마다 옮겨가기 때문이다.
        print("추가할 기사가 없습니다. 브리핑만 갱신합니다.")

    if len(fresh) > MAX_NEW:
        print("안전장치: %d건까지만 처리합니다." % MAX_NEW)
        fresh = fresh[:MAX_NEW]

    # 분류
    added = []
    if api_key and fresh:
        for i in range(0, len(fresh), BATCH):
            chunk = fresh[i:i + BATCH]
            print("  AI 분류 중… %d~%d" % (i + 1, i + len(chunk)))
            try:
                results = call_claude(api_key, chunk)
            except Exception as e:
                print("  AI 분류 실패(%s) — 이 묶음은 규칙으로 처리합니다." % e)
                results = None

            if results is None:
                for src in chunk:
                    r = classify_by_rule(src["title"])
                    added.append(dict(r, date=src["date"], title=src["title"],
                                      url=src["url"], source=src["source"]))
                continue

            for r in results:
                idx = int(r.get("n", 0)) - 1
                if not (0 <= idx < len(chunk)):
                    continue
                src = chunk[idx]
                if not r.get("keep"):
                    print("    AI 제외: %s" % src["title"])
                    continue
                cat, senti = r.get("cat"), r.get("senti")
                if cat not in CATS or senti not in ("긍정", "중립", "부정"):
                    r2 = classify_by_rule(src["title"])
                    cat, senti = r2["cat"], r2["senti"]
                added.append({
                    "date": src["date"], "cat": cat, "senti": senti,
                    "title": src["title"],
                    "summary": (r.get("summary") or "").strip(),
                    "kw": [k for k in (r.get("kw") or []) if k][:3],
                    "url": src["url"], "source": src["source"],
                })
            time.sleep(1)
    else:
        for src in fresh:
            r = classify_by_rule(src["title"])
            added.append(dict(r, date=src["date"], title=src["title"],
                              url=src["url"], source=src["source"]))

    live.extend(added)
    live.sort(key=lambda a: a["date"])

    # 브리핑 문장 — 검증본까지 합친 전체를 기준으로 계산한다(화면이 보는 모수와 같아야 한다)
    briefs = prev_briefs   # 새로 못 만들면 지난번 문장을 그대로 남긴다
    if api_key:
        allrows = verified + live
        last = max(r["date"] for r in allrows)
        facts = []
        for key, days, label in (("all", None, "수집 기간 전체"), ("7", 7, "최근 7일"), ("30", 30, "최근 30일")):
            if days is None:
                rows = allrows
            else:
                cut = (datetime.strptime(last, "%Y-%m-%d") - timedelta(days=days - 1)).strftime("%Y-%m-%d")
                rows = [r for r in allrows if r["date"] >= cut]
            f = brief_facts(rows, label)
            if f:
                facts.append(dict({"키": key}, **f))
        try:
            briefs = write_briefs(api_key, facts)
            print("브리핑 생성 완료: %s" % ", ".join(sorted(briefs)))
        except Exception as e:
            print("브리핑 생성 실패(%s) — 지난번 문장을 유지합니다." % e)

    out = {
        "updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "source": "구글 뉴스 RSS",
        "method": mode,
        "count": len(live),
        "items": live,
    }
    if briefs:
        out["briefs"] = briefs
    with open(LIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("신규 %d건 반영 — 자동 수집분 누적 %d건" % (len(added), len(live)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
