# 경기 5대 시의 부정 보도 비율을 같은 방법으로 재수집해 대조 통계를 만드는 스크립트
# (화성시도 같은 조건으로 다시 모은다. 방법이 달라지면 비교가 성립하지 않기 때문이다.)
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import collect  # 날짜 해석·정규화·RSS 파서를 그대로 쓴다

# 분야는 화성시 화면과 같은 7종을 쓴다. 다르면 견줄 수가 없다.
CATS = collect.CATS

KST = timezone(timedelta(hours=9))
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "benchmark.json")
# 덮어쓰기만 하면 과거가 사라진다. 회차별 결과를 따로 쌓아 추세를 볼 수 있게 한다.
HIST = os.path.join(BASE, "benchmark_history.json")
MODEL = "claude-haiku-4-5-20251001"
BATCH = 15
DAYS = int(os.environ.get("BENCH_DAYS", "14"))
CAP = int(os.environ.get("BENCH_CAP", "400"))   # 도시당 최대 처리 건수

# 인구는 비교 맥락용으로만 표시한다. 기사 수를 인구로 나누지는 않는다.
# RSS는 최근 것만 주기 때문에 절대 건수가 수집 시점에 좌우되어 인구당 값은 방어할 수 없다.
# 대조 대상은 우리가 임의로 고른 것이 아니라 화성시연구원이 실제로 쓰는 기준을 따랐다.
# "경기도 안에서 5대 시를 비교를 하거든요. 화성시·용인시·수원시·성남시·고양시.
#  규모를 어느 정도 비슷하게 봐서 비교를 하기 때문에."
#  — 조진숙 화성시연구원 데이터센터장, 2026-08-27 2차 예선 심사평
CITIES = [
    {"name": "화성시", "query": "화성시", "pop": 1_000_000},
    {"name": "수원시", "query": "수원시", "pop": 1_190_000},
    {"name": "용인시", "query": "용인시", "pop": 1_070_000},
    {"name": "고양시", "query": "고양시", "pop": 1_080_000},
    {"name": "성남시", "query": "성남시", "pop": 920_000},
]

# 도시 이름이 다른 뜻으로 쓰이는 제목을 막는다. 화성은 collect.py의 목록을 그대로 쓴다.
EXTRA_EXCLUDES = {
    "수원시": ["수원화성", "수원 화성", "화성행궁"],
    "용인시": [],
    "성남시": [],
    # keep_title은 시 이름에서 "시"를 떼고 찾으므로 "고양"이 "고양이"에 걸린다.
    # 고양시와 무관한 반려동물 기사가 통째로 섞이므로 반드시 막는다.
    "고양시": ["고양이", "길고양", "들고양", "아기고양", "새끼고양"],
    "화성시": [],
}


def keep_title(city, title):
    """그 도시 기사인지 1차로 거른다. 시 이름이 제목에 있어야 한다."""
    t = title.replace(" ", "")
    if city["name"].replace("시", "") not in t:
        return False
    if city["name"] == "화성시" and not collect.is_hwaseong(title):
        return False
    for bad in EXTRA_EXCLUDES.get(city["name"], []):
        if bad.replace(" ", "") in t:
            return False
    return True


def classify(api_key, city, articles):
    """제목 묶음의 긍부정과 분야를 판정받는다.
    분야를 함께 받는 이유 — 전체 부정 비율 하나로는 행정 대응이 나오지 않는다.
    "어느 분야에서 우리가 다른 시보다 나쁜가"에 답할 수 있어야 소관 부서로 이어진다."""
    listing = "\n".join("%d. %s" % (i + 1, a["title"]) for i, a in enumerate(articles))
    prompt = """아래는 '%s' 검색으로 수집한 뉴스 제목 목록이다. 각 기사를 판정해 JSON 배열로만 답하라.

%s

각 항목에 대해 다음 형식으로 출력한다.
{"n": 번호, "keep": true/false, "cat": "분야", "senti": "긍정|중립|부정"}

규칙:
- keep: 경기도 %s의 행정·생활과 직접 관련된 기사만 true. 타 지역, 중앙 정치, 스포츠 구단 경기 결과는 false.
- cat: %s 중 하나.
- senti: %s 입장에서의 긍정·중립·부정. 사건·사고, 비판, 갈등, 처벌, 민원, 부실은 부정으로 본다.
- keep이 false면 cat과 senti는 빈 값으로 둔다.
설명 없이 JSON 배열만 출력하라.""" % (city["query"], listing, city["name"],
                                    " / ".join(CATS), city["name"])

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={"content-type": "application/json",
                 "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        payload = json.load(res)
    text = payload["content"][0]["text"]
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("응답에서 JSON 배열을 찾지 못했습니다: " + text[:200])
    return json.loads(m.group())


def run_city(api_key, city):
    print("\n== %s ==" % city["name"])
    raw = collect.fetch_rss(city["query"], DAYS)
    print("  RSS %d건" % len(raw))

    seen, rows = set(), []
    for it in raw:
        key = collect.norm(it["title"])
        if key in seen or not keep_title(city, it["title"]):
            continue
        seen.add(key)
        rows.append(it)
    print("  1차 통과 %d건" % len(rows))
    rows = rows[:CAP]

    kept, neg = 0, 0
    cats = {}          # 분야별 {건수, 부정}
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        try:
            res = classify(api_key, city, chunk)
        except Exception as e:
            print("    분류 실패(%d~) — 건너뜁니다: %s" % (i, e))
            continue
        # 응답을 그대로 세면 안 된다. 모델이 항목을 빠뜨리거나 더 만들면
        # 집계가 조용히 어긋나 비율이 틀린다. 번호로 기사와 맞춰 확인한다.
        seen_n = set()
        for r in res:
            n = r.get("n")
            if not isinstance(n, int) or not (1 <= n <= len(chunk)):
                print("    번호 이상 — 건너뜁니다: %r" % (n,))
                continue
            if n in seen_n:
                print("    번호 중복 %d — 건너뜁니다" % n)
                continue
            seen_n.add(n)
            if not r.get("keep"):
                continue
            senti = r.get("senti")
            if senti not in ("긍정", "중립", "부정"):
                print("    판정값 이상(%r) — 건너뜁니다" % (senti,))
                continue
            kept += 1
            if senti == "부정":
                neg += 1
            # 분야가 우리 7종에 없으면 분야별 집계에서만 뺀다. 전체 집계는 그대로 둔다.
            cat = r.get("cat")
            if cat in CATS:
                c = cats.setdefault(cat, {"count": 0, "neg": 0})
                c["count"] += 1
                if senti == "부정":
                    c["neg"] += 1
        missing = len(chunk) - len(seen_n)
        if missing:
            print("    응답 누락 %d건 (요청 %d건 / 응답 %d건)" % (missing, len(chunk), len(seen_n)))
        time.sleep(0.4)
        print("    %d/%d 처리" % (min(i + BATCH, len(rows)), len(rows)))

    ratio = round(neg / kept * 100, 1) if kept else None
    print("  → %d건 중 부정 %d건 = %s%%" % (kept, neg, ratio))
    # 분야별 표본이 너무 작으면 비율이 튄다. 5건 미만인 분야는 내보내지 않는다.
    cats = {k: v for k, v in cats.items() if v["count"] >= 5}
    if cats:
        print("     분야별: %s" % " · ".join(
            "%s %d/%d" % (k, v["neg"], v["count"]) for k, v in sorted(cats.items())))
    return {"name": city["name"], "pop": city["pop"],
            "count": kept, "neg": neg, "ratio": ratio, "cats": cats}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ANTHROPIC_API_KEY가 없습니다. 대조 통계는 AI 판정이 있어야 성립하므로 중단합니다.")
        sys.exit(1)

    today = datetime.now(KST).date()
    cities = [run_city(api_key, c) for c in CITIES]
    ok = [c for c in cities if c["ratio"] is not None]

    out = {
        "collected": today.isoformat(),
        "window": "최근 %d일" % DAYS,
        "from": (today - timedelta(days=DAYS)).isoformat(),
        "to": today.isoformat(),
        "method": "구글 뉴스 RSS를 도시별 단일 검색어로 같은 날 같은 조건으로 수집하고, "
                  "같은 AI 모델·같은 지시문으로 분야와 긍부정을 판정했습니다. "
                  "기사 수는 수집 시점에 좌우되므로 비교에 쓰지 않고 부정 비율만 견줍니다. "
                  "분야별 값은 표본이 5건 이상인 분야만 담았습니다.",
        "cities": sorted(ok, key=lambda c: -c["ratio"]),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n저장: benchmark.json")

    # 이력 — 같은 날 다시 돌리면 그날 것을 갈아끼운다. 날짜가 중복되면 추세가 어긋난다.
    try:
        hist = json.load(open(HIST, encoding="utf-8")) if os.path.exists(HIST) else []
    except Exception:
        hist = []
    entry = {"collected": out["collected"], "from": out["from"], "to": out["to"],
             "cities": [{"name": c["name"], "count": c["count"],
                         "neg": c["neg"], "ratio": c["ratio"]} for c in out["cities"]]}
    hist = [h for h in hist if h.get("collected") != entry["collected"]]
    hist.append(entry)
    hist.sort(key=lambda h: h["collected"])
    hist = hist[-52:]        # 1년치까지만 둔다
    with open(HIST, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)
    print("저장: benchmark_history.json (%d회차)" % len(hist))
    for c in out["cities"]:
        print("  %s  %4d건  부정 %3d건  %5.1f%%" % (c["name"], c["count"], c["neg"], c["ratio"]))


if __name__ == "__main__":
    main()
