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

# 구글 뉴스 RSS는 검색어 하나당 100건이 상한이다. 기간을 90일로 늘려도 100건만 온다.
# 실측 — when:2d 로 화성시 100건 + 동탄 100건을 받아 신규가 155건이었고,
# MAX_NEW=100 에서 55건이 잘려 나갔다. 천장은 우리 코드가 아니라 구글이다.
# 그래서 검색어를 늘려 천장 자체를 올린다. 읍면동 이름은 HINTS 에 이미 들어 있다.
QUERIES = ["화성시", "동탄", "화성시청", "병점", "향남", "봉담", "남양"]
# 평소에는 이틀치만 받는다. 빈 기간을 메울 때만 워크플로에서 값을 넘겨 늘린다.
DAYS = int(os.environ.get("COLLECT_DAYS") or 2)
# 검색어를 늘려도 이 값이 낮으면 받아 놓고 버리게 된다. 함께 올린다.
# 없애지는 않는다. 요금 폭주와 데이터 오염을 막는 안전장치다.
MAX_NEW = int(os.environ.get("COLLECT_MAX") or 250)   # 한 번에 처리할 최대 건수 (비용·오염 안전장치)
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
    # 검색어에 '남양'을 넣으면서 함께 걸리는 것들. 화성시 남양읍과 무관하다.
    "남양유업", "남양주", "남양알로에", "남양홀딩스",
    # 수원 화성 (문화유산)
    "수원화성", "수원 화성", "화성행궁", "화성행렬", "화성열차", "화성어차",
    "화성성역", "정조대왕", "화성유수부",
    # 행성 화성 — 짧은 낱말("나사", "로버")은 "나사못", "로버트"까지 걸러내므로
    # 반드시 앞뒤를 붙여 한정한다.
    "화성 탐사", "화성탐사", "화성 이주", "화성인", "화성 착륙", "적색행성",
    "NASA", "미 항공우주국", "스페이스X", "스페이스엑스", "일론 머스크",
    "탐사 로버", "탐사로버", "큐리오시티", "퍼서비어런스",
    "화성 진출", "테라포밍", "화성 생명체",
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
             "교육", "학교", "학생", "병원", "접종", "상담", "취약계층"],
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


# ── 본문 붙이기 ─────────────────────────────────────────────
# 구글 뉴스 RSS는 제목만 준다. 실제로 재봤다.
#   description 이 제목과 다른 내용을 담은 기사: 30건 중 0건
#   구글 링크를 디코딩해 원문 주소를 얻은 기사: 6건 중 0건
# 네이버 검색 API는 본문 일부와 실제 언론사 주소를 함께 준다.
# 키가 없으면 이 단계를 통째로 건너뛴다. 없다고 수집을 멈추지 않는다.
NAVER_ID = os.environ.get("NAVER_CLIENT_ID", "").strip()
NAVER_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "").strip()


def strip_tags(t):
    t = re.sub(r"<[^>]+>", "", t or "")
    for a, b in [("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&apos;", "'"), ("&nbsp;", " ")]:
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def naver_search(query, display=100, start=1):
    """네이버 뉴스 검색. 본문 일부와 언론사 원문 주소를 얻는다."""
    url = ("https://openapi.naver.com/v1/search/news.json?query=%s&display=%d&start=%d&sort=date"
           % (urllib.parse.quote(query), display, start))
    req = urllib.request.Request(url, headers={
        "X-Naver-Client-Id": NAVER_ID,
        "X-Naver-Client-Secret": NAVER_SECRET,
    })
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.load(res).get("items", [])


def attach_bodies(items):
    """수집한 기사에 본문 일부와 언론사 원문 주소를 붙인다.
    제목이 정확히 같지 않아도 앞부분이 겹치면 같은 기사로 본다."""
    if not (NAVER_ID and NAVER_SECRET):
        print("  네이버 키가 없어 본문은 붙이지 않습니다. 제목만으로 판정합니다.")
        return 0

    pool = {}
    for q in QUERIES:
        for page in (1, 101, 201):
            try:
                got = naver_search(q, 100, page)
            except Exception as e:
                print("  네이버 검색 실패(%s %d): %s" % (q, page, str(e)[:60]))
                break
            if not got:
                break
            for it in got:
                t = strip_tags(it.get("title", ""))
                if t:
                    pool[norm(t)] = {
                        "body": strip_tags(it.get("description", "")),
                        "url": it.get("originallink") or it.get("link") or "",
                    }
            time.sleep(0.12)
    print("  네이버에서 %d건을 받아 두었습니다." % len(pool))

    # 제목 앞부분으로도 맞춰 본다. 언론사마다 제목 뒤를 조금씩 다르게 붙인다.
    prefix = {}
    for k, v in pool.items():
        p = k[:14]
        if len(p) >= 10:
            prefix.setdefault(p, v)

    hit = 0
    for a in items:
        k = norm(a["title"])
        m = pool.get(k) or prefix.get(k[:14])
        if m and m["body"]:
            a["body"] = m["body"][:300]
            if m["url"]:
                a["url"] = m["url"]      # 구글 중계 주소를 언론사 주소로 바꾼다
            hit += 1
    print("  본문을 붙인 기사 %d건 / %d건 (%.0f%%)"
          % (hit, len(items), hit / max(1, len(items)) * 100))
    return hit


def parse_pubdate(pub):
    """RSS 날짜를 읽는다. 어느 형식으로도 못 읽으면 None을 돌려 그 기사를 버리게 한다."""
    pub = (pub or "").strip()
    if not pub:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z",
                "%d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(pub, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST).strftime("%Y-%m-%d")
    return None


def fetch_rss(query, days):
    """구글 뉴스 RSS에서 최근 기사 목록을 가져온다. API 키가 필요 없다."""
    q = urllib.parse.quote("%s when:%dd" % (query, days))
    url = "https://news.google.com/rss/search?q=%s&hl=ko&gl=KR&ceid=KR:ko" % q
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as res:
        root = ET.fromstring(res.read())

    items = []
    skipped = 0
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        source = (it.findtext("source") or "").strip()
        # 구글 뉴스 제목은 "기사제목 - 언론사" 형태라 언론사를 떼어낸다
        if source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)].strip()

        # 날짜를 못 읽으면 오늘 날짜로 채우지 않는다.
        # 틀린 날짜가 들어가면 추이 차트가 조용히 왜곡된다. 그럴 바엔 그 기사를 버린다.
        date = parse_pubdate(it.findtext("pubDate") or "")
        if not date:
            skipped += 1
            continue

        if title:
            items.append({
                "date": date,
                "title": title,
                "url": (it.findtext("link") or "").strip(),
                "source": source,
            })
    if skipped:
        print("    날짜를 읽지 못해 %d건 제외" % skipped)
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

    out = {"cat": best, "senti": senti, "summary": "", "kw": kw}
    # 키워드가 하나도 안 맞았는데 "행정"이라고 단정하면 근거 없는 라벨이 된다.
    # 분야 7종은 고정이라 값은 그대로 두되, 근거가 없다는 사실을 함께 남긴다.
    if best_hit == 0:
        out["conf"] = "low"
    return out


def call_claude(api_key, articles):
    """기사 묶음을 Claude에 보내 분류·요약·키워드를 한 번에 받는다.
    본문이 붙어 있으면 제목과 함께 보낸다. 제목 32자로 내리던 판정이
    본문을 보고 내리는 판정으로 바뀐다."""
    def one(i, a):
        line = "%d. %s" % (i + 1, a["title"])
        if a.get("body"):
            line += "\n   [본문] " + a["body"][:220]
        return line
    listing = "\n".join(one(i, a) for i, a in enumerate(articles))
    prompt = """아래는 '화성시' 검색으로 수집한 뉴스 제목 목록이다. 각 기사를 분석해 JSON 배열로만 답하라.

%s

각 항목에 대해 다음 형식으로 출력한다.
{"n": 번호, "keep": true/false, "cat": "분야", "senti": "긍정|중립|부정", "summary": "한 문장 요약", "kw": ["키워드1","키워드2","키워드3"]}

규칙:
- keep: 경기도 화성시(동탄·병점·향남·봉담·남양 등 포함)의 행정·생활과 직접 관련된 기사만 true. 경기도 전체 정치, 타 지역, 중앙 정치, 행성 화성 관련은 false.
- cat: %s 중 하나.
- senti: 화성시 입장에서의 긍정·중립·부정.
- summary: 한 문장(50자 내외) 요약. 본문이 있으면 본문을 바탕으로, 없으면 제목만으로 쓴다.
  둘 다 사실만 담고 추측하지 않는다.
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


def brief_facts(rows, label, base=None):
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
        # 2건은 같은 사건을 두 매체가 쓴 것일 수 있어 "집중"이라 부르기 어렵다.
        # 3건 이상일 때만 넘기고, 표현도 사실 그대로 "가장 많았던 날"로 둔다.
        "부정최다일": "%s %d건" % peak if peak and peak[1] >= 3 else None,
        # 검증본(빅카인즈)과 자동수집분(구글 RSS)은 수집 방식이 달라 모집단이 다르다.
        # 예전에는 '평상시부정비율'과 '평상시대비'를 넘기고 AI에게 견주어 쓰라고 시켰다.
        # 그 결과 화면이 "평상시 8.6%보다 낮은 수준"이라고 말했는데,
        # 이는 우리가 심사에서 하지 않겠다고 답한 바로 그 견줌이다.
        # 그래서 판정값을 아예 넘기지 않는다. 참고선 숫자만 참고용으로 넘긴다.
        "검증본참고선": ("%.1f%% (수집 방식이 달라 직접 비교하지 않는다)" % base) if base else None,
    }


def write_briefs(api_key, facts):
    """계산된 수치를 넘겨 기간별 브리핑 문장을 받는다. 실패하면 None을 돌려 화면이 기존 방식으로 돌아가게 한다."""
    prompt = """아래는 화성시 언론보도 대시보드의 기간별 집계 결과다.
각 기간마다 화성시 홍보 담당 공무원이 아침에 읽을 브리핑을 2~3문장으로 작성하라.

%s

규칙:
- 주어진 수치만 사용한다. 새로운 숫자를 만들거나 추정하지 않는다.
- '검증본참고선'은 수집 방식이 다른 별개 자료다. 절대 견주지 마라.
  "참고선보다 낮다", "평상시보다 낮은 수준", "참고선 대비 양호" 같은 표현을 쓰면 안 된다.
  "~보다", "~대비", "~에 비해", "밑돈다", "웃돈다", "개선되었다", "양호하다"도 마찬가지다.
  비율은 그냥 "부정 142건(8%%)"처럼 사실만 적는다.
  이유 — 검증본은 사람이 걸러낸 빅카인즈 자료이고 이 수치는 자동 수집분이다.
  모집단이 달라 비율을 나란히 놓으면 담당자가 잘못 읽는다.
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


def top_keywords(rows, n=10):
    """기간 안에서 가장 많이 나온 키워드를 센다. 집계는 코드가 하고 AI는 문장만 쓴다."""
    c = Counter(k for r in rows for k in (r.get("kw") or []) if k)
    return [k for k, _ in c.most_common(n)]


def keyword_facts(rows, kw):
    """키워드 하나에 대한 수치를 계산한다. 요약문에 쓸 사실은 전부 여기서 나온다."""
    hit = [r for r in rows if kw in (r.get("kw") or [])]
    if len(hit) < 3:
        return None
    senti = Counter(r["senti"] for r in hit)
    cats = Counter(r["cat"] for r in hit)
    dates = sorted(r["date"] for r in hit)
    neg_titles = [r["title"] for r in hit if r["senti"] == "부정"][:3]
    return {
        "키워드": kw,
        "기사수": len(hit),
        "부정": senti.get("부정", 0),
        "긍정": senti.get("긍정", 0),
        "최다분야": "%s %d건" % cats.most_common(1)[0] if cats else None,
        "기간": "%s~%s" % (dates[0], dates[-1]),
        "부정기사예": neg_titles or None,
    }


def write_keyword_notes(api_key, facts):
    """키워드별 한두 문장 해설을 받는다. 숫자는 이미 계산해 넘기므로 AI는 문장만 쓴다."""
    prompt = """아래는 화성시 언론보도 대시보드에서 키워드별로 집계한 결과다.
각 키워드마다 화성시 홍보 담당자가 읽을 해설을 **한두 문장**으로 작성하라.

%s

규칙:
- 주어진 수치만 사용한다. 새로운 숫자를 만들거나 추정하지 않는다.
- 이 키워드가 어떤 맥락에서 나오는지, 담당자가 눈여겨볼 점이 무엇인지 쓴다.
- 부정 기사 예시가 있으면 그 내용을 반영한다. 없으면 언급하지 않는다.
- 키워드마다 문장 구조를 다르게 쓴다.
- 강조는 [[b]]와 [[/b]]로, 부정 관련 수치는 [[w]]와 [[/w]]로 감싼다.
- HTML 태그를 쓰지 마라. 위 표시자만 쓴다.
- 과장하거나 추측하지 않는다.

{"키워드1": "해설", "키워드2": "해설"}
위 형식의 JSON 객체로만 답하라.""" % json.dumps(facts, ensure_ascii=False, indent=2)

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 2500,
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
        raise ValueError("키워드 해설 응답에서 JSON을 찾지 못했습니다: " + text[:200])
    out = json.loads(m.group())
    return {k: v.strip() for k, v in out.items() if isinstance(v, str) and v.strip()}


# 제목을 토큰으로 쪼갤 때 버릴 낱말. 어느 기사에나 나와서 묶는 데 방해가 된다.
CLUSTER_STOP = set("화성 화성시 화성특례시 시 및 등 통해 위해 대한 관련 오늘 내일 "
                   "속보 단독 종합 포토 영상 사진".split())


def title_tokens(title):
    return set(w for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", title) if w not in CLUSTER_STOP)


def cluster_negative(rows, threshold=0.35, min_size=2, top=5):
    """부정 기사를 사건 단위로 묶는다.

    같은 사건을 여러 매체가 제목만 바꿔 쓰면 지금은 별개 기사로 세어진다.
    제목 토큰이 얼마나 겹치는지(자카드 유사도)로 같은 사건을 찾아 묶는다.
    """
    neg = [r for r in rows if r.get("senti") == "부정"]
    if len(neg) < 2:
        return []

    toks = [title_tokens(r["title"]) for r in neg]
    parent = list(range(len(neg)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(neg)):
        if not toks[i]:
            continue
        for j in range(i + 1, len(neg)):
            if not toks[j]:
                continue
            inter = len(toks[i] & toks[j])
            if not inter:
                continue
            if inter / len(toks[i] | toks[j]) >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    groups = {}
    for i in range(len(neg)):
        groups.setdefault(find(i), []).append(neg[i])

    out = []
    for g in groups.values():
        if len(g) < min_size:
            continue
        g.sort(key=lambda r: r["date"])
        srcs = sorted(set(r.get("source", "") for r in g if r.get("source")))
        out.append({
            "label": g[-1]["title"],          # 가장 최근 제목을 대표로 삼는다
            "count": len(g),
            "sources": len(srcs),
            "first": g[0]["date"],
            "last": g[-1]["date"],
            "cat": Counter(r["cat"] for r in g).most_common(1)[0][0],
            "titles": [r["title"] for r in g],
        })
    # 최근에 끝난 이슈를 앞에, 같으면 규모가 큰 것을 앞에 둔다
    out.sort(key=lambda x: (x["last"], x["count"]), reverse=True)
    return out[:top]


def issue_score(it, today):
    """담당자가 오늘 먼저 볼 순서를 정한다. 최신성을 가장 크게 본다."""
    last = datetime.strptime(it["last"], "%Y-%m-%d").date()
    first = datetime.strptime(it["first"], "%Y-%m-%d").date()
    age = (today - last).days
    span = (last - first).days + 1
    recency = max(0, 45 - age * 7)           # 오늘일수록 크다 (최대 45)
    spread = min(30, it["sources"] * 10)     # 몇 개 매체가 다뤘나 (최대 30)
    volume = min(15, it["count"] * 5)        # 기사 수 (최대 15)
    lasting = 10 if span >= 7 else 0         # 오래 이어지면 가산
    return min(100, recency + spread + volume + lasting)


def issue_state(it, today):
    """이슈가 지금 어떤 국면인지 한 낱말로 말한다."""
    last = datetime.strptime(it["last"], "%Y-%m-%d").date()
    first = datetime.strptime(it["first"], "%Y-%m-%d").date()
    age = (today - last).days
    span = (last - first).days + 1
    if age <= 1 and it["count"] >= 2:
        return "확산 중"
    if age <= 1:
        return "신규"
    if age <= 3:
        return "진행 중"
    if span >= 7:
        return "장기"
    return "종결"


def build_today(allrows, today, top=3, window=14):
    """오늘 확인할 것을 뽑는다.

    묶인 이슈만 보면 오늘 새로 뜬 단일 보도를 놓친다.
    그래서 1건짜리도 그룹으로 유지하고 점수로 정렬한다.
    """
    cut = (today - timedelta(days=window - 1)).strftime("%Y-%m-%d")
    recent = [r for r in allrows if r["date"] >= cut]
    groups = cluster_negative(recent, min_size=1, top=300)
    if not groups:
        return []
    for g in groups:
        g["score"] = issue_score(g, today)
        g["state"] = issue_state(g, today)
    groups.sort(key=lambda g: g["score"], reverse=True)
    return groups[:top]


def detect_surge(live_items, today, recent_days=3, base_days=12, min_count=3, min_ratio=2.0):
    """분야별 부정 보도가 갑자기 늘었는지 본다.

    검증본(빅카인즈)과 자동수집분은 수집 범위가 달라 섞으면 비교가 왜곡된다.
    그래서 자동수집분끼리만 비교한다.
    적은 수에서 배수가 튀는 것을 막으려고 최소 건수 조건을 둔다.
    """
    def win(back, days):
        end = today - timedelta(days=back)
        start = end - timedelta(days=days - 1)
        return [r for r in live_items
                if start.strftime("%Y-%m-%d") <= r["date"] <= end.strftime("%Y-%m-%d")]

    cur = Counter(r["cat"] for r in win(0, recent_days) if r.get("senti") == "부정")
    base = Counter(r["cat"] for r in win(recent_days, base_days) if r.get("senti") == "부정")

    out = []
    for cat in CATS:
        a = cur.get(cat, 0)
        if a < min_count:
            continue
        b_day = base.get(cat, 0) / base_days
        if b_day <= 0:
            continue                      # 비교 기준이 없으면 배수를 말하지 않는다
        ratio = (a / recent_days) / b_day
        if ratio >= min_ratio:
            out.append({"cat": cat, "recent": a, "days": recent_days,
                        "base": base.get(cat, 0), "base_days": base_days,
                        "base_daily": round(b_day, 2), "ratio": round(ratio, 1)})
    out.sort(key=lambda x: -x["ratio"])
    return out


def ops_record(repo, token):
    """자동 수집이 며칠째 무중단으로 돌았는지 센다. 실패하면 이 항목만 빠진다."""
    if not repo:
        return None
    req = urllib.request.Request(
        "https://api.github.com/repos/%s/actions/runs?per_page=100" % repo,
        headers={"User-Agent": "pinpoint", "Accept": "application/vnd.github+json",
                 **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    runs = [x for x in data.get("workflow_runs", []) if x.get("event") == "schedule"]
    if not runs:
        return None
    days = sorted(set(x["created_at"][:10] for x in runs))
    fails = sum(1 for x in runs if x.get("conclusion") not in ("success", None))
    return {"runs": len(runs), "days": len(days), "since": days[0],
            "last": days[-1], "failures": fails}


class CorruptedLive(Exception):
    """기존 수집분을 읽지 못했다는 뜻. 이때는 절대 새 파일로 덮어쓰지 않는다."""


def load_live():
    """기존 수집분과 브리핑을 읽는다.

    파일이 아예 없으면 처음 실행이므로 빈 상태로 시작해도 된다.
    그러나 파일이 있는데 읽지 못하면 손상된 것이므로 예외를 던진다.
    빈 배열로 시작해 저장해버리면 그동안 쌓인 수집분이 통째로 사라진다.
    """
    if not os.path.exists(LIVE_FILE):
        print("news_live.json이 없습니다. 첫 실행으로 보고 시작합니다.")
        return [], None, None, None, None
    try:
        with open(LIVE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError) as e:
        raise CorruptedLive("news_live.json을 읽지 못했습니다: %s" % e)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise CorruptedLive("news_live.json 구조가 예상과 다릅니다.")
    notes = data.get("keyword_notes") or {}
    return (data["items"], data.get("briefs"), data.get("briefs_updated"),
            notes.get("items"), notes.get("updated"))


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    mode = "AI 분류" if api_key else "키워드 규칙 분류"
    print("== 화성시 뉴스 자동 수집 (%s) ==" % mode)

    # 검증본은 중복 판단에만 쓰고 절대 수정하지 않는다
    with open(VERIFIED_FILE, encoding="utf-8") as f:
        verified = json.load(f)
    live, prev_briefs, prev_briefs_at, prev_kw_notes, prev_kw_notes_at = load_live()
    seen = {norm(a["title"]) for a in verified} | {norm(a["title"]) for a in live}
    print("검증본 %d건 / 기존 수집분 %d건" % (len(verified), len(live)))

    # 수집 — 검색어별로 따로 담는다. 뒤에서 자를 때 한 검색어만 살아남지 않게 하기 위해서다.
    by_query = []
    for q in QUERIES:
        try:
            got = fetch_rss(q, DAYS)
            print("  '%s' 검색 %d건" % (q, len(got)))
            by_query.append(got)
        except Exception as e:
            print("  '%s' 검색 실패: %s" % (q, e))
            by_query.append([])

    # 검색어를 번갈아 꺼내 섞는다. MAX_NEW로 잘라도 두 검색어가 고르게 남는다.
    fetched = []
    for i in range(max((len(g) for g in by_query), default=0)):
        for g in by_query:
            if i < len(g):
                fetched.append(g[i])

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

    # 분류 전에 본문을 붙인다. 붙은 기사는 제목이 아니라 본문으로 판정된다.
    if fresh:
        try:
            attach_bodies(fresh)
        except Exception as e:
            print("본문 붙이기 실패 — 제목만으로 진행합니다: %s" % str(e)[:80])

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

            # AI가 번호를 빠뜨리거나 두 번 답할 수 있다.
            # 처리한 번호를 기록해 누락은 규칙으로 채우고, 중복은 첫 번째만 쓴다.
            handled = set()
            for r in results:
                if not isinstance(r, dict):
                    continue
                try:
                    idx = int(r.get("n", 0)) - 1
                except (TypeError, ValueError):
                    continue
                if not (0 <= idx < len(chunk)) or idx in handled:
                    continue
                handled.add(idx)
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
                    **({"body": src["body"]} if src.get("body") else {}),
                    "title": src["title"],
                    "summary": (r.get("summary") or "").strip(),
                    "kw": [k for k in (r.get("kw") or []) if k][:3],
                    "url": src["url"], "source": src["source"],
                })

            missing = [i for i in range(len(chunk)) if i not in handled]
            if missing:
                print("    AI가 %d건을 빠뜨려 규칙으로 채웁니다." % len(missing))
                for i in missing:
                    src = chunk[i]
                    r2 = classify_by_rule(src["title"])
                    added.append(dict(r2, date=src["date"], title=src["title"],
                                      url=src["url"], source=src["source"]))
            time.sleep(1)
    else:
        for src in fresh:
            r = classify_by_rule(src["title"])
            added.append(dict(r, date=src["date"], title=src["title"],
                              url=src["url"], source=src["source"]))

    before = len(live)
    live.extend(added)
    live.sort(key=lambda a: a["date"])

    # 저장 직전 안전장치 — 누적분이 줄어드는 일은 정상 동작에서 일어나지 않는다.
    # 줄었다면 어딘가 잘못된 것이므로 덮어쓰지 않고 멈춘다.
    if len(live) < before:
        print("누적분이 %d건에서 %d건으로 줄었습니다. 저장하지 않고 중단합니다." % (before, len(live)))
        return 1

    # 브리핑 문장 — 검증본까지 합친 전체를 기준으로 계산한다(화면이 보는 모수와 같아야 한다)
    briefs = prev_briefs        # 새로 못 만들면 지난번 문장을 그대로 남긴다
    briefs_at = prev_briefs_at  # 그때는 작성 시각도 지난번 것을 유지한다
    if api_key:
        allrows = verified + live
        last = max(r["date"] for r in allrows)
        # 사람이 고른 검증본 2개월치가 '평상시'다. 지금이 높은지 낮은지 견줄 유일한 기준이다.
        vneg = sum(1 for r in verified if r.get("senti") == "부정")
        base_ratio = round(vneg / len(verified) * 100, 1) if verified else None
        facts = []
        for key, days, label in (("all", None, "수집 기간 전체"), ("7", 7, "최근 7일"), ("30", 30, "최근 30일")):
            if days is None:
                rows = allrows
            else:
                cut = (datetime.strptime(last, "%Y-%m-%d") - timedelta(days=days - 1)).strftime("%Y-%m-%d")
                rows = [r for r in allrows if r["date"] >= cut]
            f = brief_facts(rows, label, base_ratio)
            if f:
                facts.append(dict({"키": key}, **f))
        try:
            briefs = write_briefs(api_key, facts)
            briefs_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
            print("브리핑 생성 완료: %s" % ", ".join(sorted(briefs)))
        except Exception as e:
            print("브리핑 생성 실패(%s) — 지난번 문장과 작성 시각을 유지합니다." % e)

    # 부정 기사를 사건 단위로 묶는다. 실패해도 화면이 죽지 않도록 통째로 감싼다.
    issues = None
    try:
        allrows = verified + live
        found = cluster_negative(allrows)
        if found:
            issues = {"updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                      "items": found}
            print("부정 이슈 묶음 %d개 (기사 %d건)"
                  % (len(found), sum(i["count"] for i in found)))
        else:
            print("2건 이상 묶인 부정 이슈가 없습니다.")
    except Exception as e:
        print("이슈 묶기 실패(%s) — 이 기능만 건너뜁니다." % e)

    # 키워드별 해설 — 최근 30일 기준. 전체 기간 해설은 이미 브리핑이 담당한다.
    kw_notes = prev_kw_notes
    kw_notes_at = prev_kw_notes_at
    if api_key:
        try:
            allrows = verified + live
            last = max(r["date"] for r in allrows)
            cut = (datetime.strptime(last, "%Y-%m-%d") - timedelta(days=29)).strftime("%Y-%m-%d")
            recent = [r for r in allrows if r["date"] >= cut]
            wanted = top_keywords(recent, 10)
            facts = [f for f in (keyword_facts(recent, k) for k in wanted) if f]
            if facts:
                kw_notes = write_keyword_notes(api_key, facts)
                kw_notes_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
                print("키워드 해설 생성 완료: %d개" % len(kw_notes))
        except Exception as e:
            print("키워드 해설 실패(%s) — 지난번 해설을 유지합니다." % e)

    # 데이터 갱신 시각과 브리핑 작성 시각을 분리한다.
    # 브리핑을 새로 못 만든 날에 오늘 쓴 것처럼 보이면 화면이 사실과 다른 말을 하게 된다.
    out = {
        "updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "source": "구글 뉴스 RSS",
        "method": mode,
        "count": len(live),
        "items": live,
    }
    # 오늘 확인할 것 — 담당자가 화면을 열자마자 볼 세 가지
    today_block = None
    try:
        allrows = verified + live
        T = datetime.strptime(max(r["date"] for r in allrows), "%Y-%m-%d").date()
        picks = build_today(allrows, T)
        surge = detect_surge(live, T)
        if picks:
            today_block = {"updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                           "items": picks, "surge": surge}
            print("오늘 확인할 것 %d건 (최고 %d점) · 급증 분야 %d개"
                  % (len(picks), picks[0]["score"], len(surge)))
            for s in surge:
                print("    급증: %s 최근 %d일 %d건 (평소 대비 %.1f배)"
                      % (s["cat"], s["days"], s["recent"], s["ratio"]))
    except Exception as e:
        print("오늘 확인할 것 생성 실패(%s) — 이 기능만 건너뜁니다." % e)

    # 운영 기록 — 며칠째 무중단으로 돌고 있는지. 우리 화면의 신뢰 근거다.
    ops = None
    try:
        ops = ops_record(os.environ.get("GITHUB_REPOSITORY"),
                         os.environ.get("GITHUB_TOKEN"))
        if ops:
            print("운영 기록: %d일 연속 · 정기실행 %d회 · 실패 %d회"
                  % (ops["days"], ops["runs"], ops["failures"]))
    except Exception as e:
        print("운영 기록 조회 실패(%s) — 이 항목만 건너뜁니다." % e)

    # 사람이 검수한 정확도 — quality.json이 있으면 그대로 싣는다
    quality = None
    qpath = os.path.join(BASE, "quality.json")
    if os.path.exists(qpath):
        try:
            with open(qpath, encoding="utf-8") as f:
                quality = json.load(f)
            print("검수 결과 반영: %s" % quality.get("checked"))
        except Exception as e:
            print("quality.json을 읽지 못했습니다(%s)." % e)

    # 무엇을 걸러냈는지 남긴다. 로그만 있으면 나중에 무엇을 놓쳤는지 확인할 수 없다.
    if dropped:
        out["dropped"] = {
            "count": len(dropped),
            "reason": "제목에 화성시 단서가 없거나 동음이의어로 판단",
            "samples": dropped[:20],
        }
    low = sum(1 for a in added if a.get("conf") == "low")
    if low:
        out["low_confidence"] = low
        print("근거 키워드 없이 기본 분야로 처리된 기사 %d건" % low)

    if briefs:
        out["briefs"] = briefs
        if briefs_at:
            out["briefs_updated"] = briefs_at
    if issues:
        out["issues"] = issues
    if today_block:
        out["today"] = today_block
    if ops:
        out["ops"] = ops
    if quality:
        out["quality"] = quality
    if kw_notes:
        out["keyword_notes"] = {
            "updated": kw_notes_at,
            "window": "최근 30일",
            "items": kw_notes,
        }
    with open(LIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("신규 %d건 반영 — 자동 수집분 누적 %d건" % (len(added), len(live)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CorruptedLive as e:
        # 기존 수집분을 읽지 못한 날은 아무것도 저장하지 않고 끝낸다.
        # 빈 파일로 덮어쓰면 그동안 쌓인 기사가 전부 사라진다.
        print("중단: %s" % e)
        print("기존 파일을 보존했습니다. 저장소의 news_live.json을 확인해 주세요.")
        sys.exit(1)
