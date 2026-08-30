# 본문을 붙이면 판정이 얼마나 달라지는지 재는 실험. 배포본은 건드리지 않는다.
#
# 심사에서 "내용적인 분석까지 한다고 했었잖아요"라는 말씀을 들었다.
# 우리는 지금도 기사 제목만 보고 판정한다.
# 그래서 본문을 붙여 다시 판정해 보고, 무엇이 달라지는지를 숫자로 남긴다.
#
# 상시로 켜지 않는 이유는 두 가지다.
#   1. 사람이 검수한 93%/81%은 제목 기준으로 잰 값이다. 판정이 바뀌면 그 숫자가 화면을 설명하지 못한다.
#   2. 매일 도는 작업에 브라우저를 넣으면 실패 지점이 하나 는다.
#      시간은 문제가 아니다. 하루 신규가 44~94건이라 3분이면 끝난다.
#      문제는 언론사가 깃허브 주소를 막는지를 아직 안 재봤다는 것이다.
#      집에서는 95%였다. 이 실행 로그가 그 답을 준다.
#
# 그래서 이 파일은 손으로만 돌린다. 결과는 body_experiment.json 에만 쓴다.
# 본문은 판정에만 쓰고 버린다. 남의 기사라 저장소에 남기지 않는다.
import json
import os
import sys

import collect          # 판정 함수를 그대로 쓴다. 다른 함수를 쓰면 비교가 성립하지 않는다.

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(BASE, "news_live.json")
BODIES = os.path.join(BASE, "_bodies_tmp.json")   # fetch_bodies.js 가 그 자리에서 만든다. 커밋하지 않는다.
OUT = os.path.join(BASE, "body_experiment.json")

BATCH = 15
LIMIT = int(os.environ.get("EXP_LIMIT", "300"))


def main():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("ANTHROPIC_API_KEY 가 없습니다. 이 실험은 판정을 다시 받아야 하므로 키가 필요합니다.")
        sys.exit(1)

    live = json.load(open(LIVE, encoding="utf-8"))
    bodies = json.load(open(BODIES, encoding="utf-8"))

    # 본문이 충분히 붙은 것만 고른다. 200자 미만은 사진 기사이거나 안내문이다.
    pool = []
    for a in live["items"]:
        b = bodies.get(a["title"])
        if b and len(b.get("body") or "") >= 200:
            # 300자까지만 담아 두었다. 화면에 적어 둔 저작권 기준과 같게 유지한다.
            pool.append((a, b["body"][:300]))
    print("본문이 붙은 기사 %d건 / 자동수집분 %d건" % (len(pool), len(live["items"])))

    pool = pool[-LIMIT:]           # 최근 것부터. 오래된 기사는 판정 기준이 달랐을 수 있다.
    print("이번 실험 대상 %d건\n" % len(pool))

    changed_cat, changed_senti, neg_flip, rows = 0, 0, 0, []
    for i in range(0, len(pool), BATCH):
        chunk = pool[i:i + BATCH]
        arts = [{"title": a["title"], "body": body} for a, body in chunk]
        try:
            got = collect.call_claude(key, arts)
        except Exception as e:
            print("  %d~%d번 실패, 건너뜁니다: %s" % (i + 1, i + len(chunk), str(e)[:80]))
            continue
        by_n = {g.get("n"): g for g in got if isinstance(g, dict)}
        for j, (a, _) in enumerate(chunk):
            g = by_n.get(j + 1)
            if not g or not g.get("keep"):
                continue
            row = {
                "title": a["title"],
                "제목판정": {"cat": a["cat"], "senti": a["senti"]},
                "본문판정": {"cat": g.get("cat"), "senti": g.get("senti")},
                "본문요약": g.get("summary"),
            }
            c = a["cat"] != g.get("cat")
            s = a["senti"] != g.get("senti")
            if c:
                changed_cat += 1
            if s:
                changed_senti += 1
            # 우리 화면은 부정만 쓴다. 부정이냐 아니냐가 바뀐 것이 진짜 중요한 변화다.
            if (a["senti"] == "부정") != (g.get("senti") == "부정"):
                neg_flip += 1
                row["부정여부바뀜"] = True
            if c or s:
                rows.append(row)
        print("  %d / %d건 판정" % (min(i + BATCH, len(pool)), len(pool)))

    n = len(pool)
    res = {
        "잰날": live.get("updated"),
        "대상": n,
        "분야바뀜": changed_cat,
        "긍부정바뀜": changed_senti,
        "부정여부바뀜": neg_flip,
        "바뀐것": rows[:120],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("\n════ 결과 ════")
    print("  대상            %d건" % n)
    print("  분야가 바뀜      %d건 (%.1f%%)" % (changed_cat, changed_cat / max(1, n) * 100))
    print("  긍부정이 바뀜    %d건 (%.1f%%)" % (changed_senti, changed_senti / max(1, n) * 100))
    print("  부정 여부가 바뀜 %d건 (%.1f%%)  ← 화면이 실제로 쓰는 판정"
          % (neg_flip, neg_flip / max(1, n) * 100))
    print("\n저장: body_experiment.json  (news_live.json 은 건드리지 않았습니다)")


if __name__ == "__main__":
    main()
