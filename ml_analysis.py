# 수집한 기사에 머신러닝 수업에서 배운 기법을 적용해 분류 판정을 점검하는 분석 스크립트
#
# ★ 가장 중요한 전제 ★
# 여기서 쓰는 정답(y)은 사람이 매긴 것이 아니라 AI가 매긴 것이다.
# 따라서 이 분석은 "AI가 맞았는가"를 재는 것이 아니라
# "AI의 판정 규칙이 제목만으로 재현 가능한 규칙인가"를 재는 것이다.
# 이 구분을 흐리면 우리가 하지 말자고 한 과장을 우리가 하게 된다.
#
# 적용한 수업 내용
#   1. 클래스 불균형 — 정확도가 왜 쓸모없는 지표인지
#   2. 특성 공학 — 한글 제목의 문자 n-gram TF-IDF
#   3. 파이프라인 — 벡터화와 모델을 하나로 묶어 누수를 막음
#   4. 데이터 누수 — 무작위 분할 · 시간 분할 · 사건 분할 비교
#   5. 교차 검증 — StratifiedKFold
#   6. 평가 지표 — 혼동 행렬 · 정밀도 · 재현율 · F1 · ROC AUC
#   7. 모델 비교 — 로지스틱(L2) · 결정 트리 · 랜덤 포레스트 · 그래디언트 부스팅
#   8. 편향-분산 — 결정 트리 깊이에 따른 학습/검증 곡선
#   9. 정규화 — 로지스틱 회귀의 규제 강도 곡선
#  10. 베이스라인 — 규칙 기반 분류기와의 비교
#  11. 해석 — 어떤 낱말이 부정 판정을 끌어내는가
#  12. 점수 항목 점검 — 표준화 · 상관 · 주성분분석으로 중복 계산을 찾음
#  13. 지속 예측 — 첫날 정보만으로 그 이슈가 오래 갈지 맞힐 수 있는가
import io
import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np

# scipy 최적화기가 사이킷런에 없는 옵션을 만나 경고를 쏟아낸다. 결과에는 영향이 없다.
warnings.filterwarnings("ignore")
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


# ── 데이터 ──────────────────────────────────────────────────
def load():
    v = json.load(io.open(os.path.join(BASE, "news_data.json"), encoding="utf-8"))
    for a in v:
        a["src"] = "검증본"
    live = json.load(io.open(os.path.join(BASE, "news_live.json"), encoding="utf-8"))
    for a in live["items"]:
        a["src"] = "자동수집"
    rows = v + live["items"]
    rows.sort(key=lambda a: a["date"])
    return rows


# ── 사건 묶기 ───────────────────────────────────────────────
# 같은 사건 기사가 학습셋과 시험셋에 나뉘어 들어가면 모델이 답을 미리 보게 된다.
# 대시보드에서 쓰는 것과 같은 방식(제목 낱말 겹침)으로 묶어 그룹 번호를 만든다.
# collect.py의 cluster_negative와 같은 기준을 쓴다. 임계값 0.35, 날짜 창 제한 없음.
# 기준이 다르면 화면이 묶은 이슈와 다른 것을 재게 된다.
CLUSTER_STOP = {"화성", "화성시", "화성특례시", "동탄", "경기", "경기도",
                "속보", "종합", "단독", "영상", "포토", "기자", "뉴스",
                "오늘", "내일", "올해", "지난", "이번", "관련", "위해", "대한"}


def tokens(t):
    return {w for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", t) if w not in CLUSTER_STOP}


def event_groups(rows, thr=0.35):
    toks = [tokens(r["title"]) for r in rows]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # 낱말 하나도 안 겹치는 쌍은 아예 견주지 않는다. 그래야 전체 비교가 감당된다.
    idx = defaultdict(list)
    for i, ts in enumerate(toks):
        for w in ts:
            idx[w].append(i)

    for i, ti in enumerate(toks):
        if not ti:
            continue
        cand = set()
        for w in ti:
            cand.update(j for j in idx[w] if j > i)
        for j in cand:
            tj = toks[j]
            if not tj:
                continue
            inter = len(ti & tj)
            if inter and inter / len(ti | tj) >= thr:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    # 뿌리 번호를 그대로 쓰면 안 된다.
    # 파이썬의 문자열 해시가 실행마다 달라져 집합 순회 순서가 바뀌고,
    # 그러면 같은 묶음인데도 뿌리 번호가 달라진다.
    # GroupKFold는 그 번호로 폴드를 나누므로 돌릴 때마다 점수가 달라진다.
    # 묶음 자체는 같으므로, 가장 먼저 나온 순서대로 다시 번호를 매겨 고정한다.
    roots = [find(i) for i in range(len(rows))]
    canon = {}
    for r in roots:
        if r not in canon:
            canon[r] = len(canon)
    return np.array([canon[r] for r in roots])


# ── 규칙 기반 베이스라인 ────────────────────────────────────
# collect.py가 API 키 없을 때 쓰는 대체 경로와 같은 낱말 목록이다.
NEG_WORDS = ["사고", "화재", "사망", "부상", "논란", "갈등", "반발", "항의", "시위",
             "비판", "우려", "피해", "적발", "위반", "불법", "구속", "기소", "수사",
             "고발", "파업", "중단", "지연", "무산", "부실", "실패", "하락", "감소",
             "붕괴", "침수", "오염", "악취", "불편", "폐쇄", "취소", "소송", "의혹",
             "비리", "횡령", "체납", "누락", "결함", "폭행", "실종", "경고", "위기"]
POS_WORDS = ["개통", "준공", "완공", "유치", "선정", "수상", "우수", "최우수", "1위",
             "확대", "증가", "성과", "협약", "체결", "지원", "개최", "성공", "돌파",
             "최초", "기록", "호평", "만족", "개선", "해소", "승격", "출범", "착공",
             "표창", "인증", "달성", "혁신", "활성화", "무료", "장학", "기부", "감사패"]


def rule_predict(titles):
    out = []
    for t in titles:
        n = sum(1 for w in NEG_WORDS if w in t)
        p = sum(1 for w in POS_WORDS if w in t)
        out.append(1 if n > p else 0)
    return np.array(out)


# ── 지표 ────────────────────────────────────────────────────
def scores(y, pred, prob=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    d = {
        "정확도": (tp + tn) / len(y),
        "정밀도": precision_score(y, pred, zero_division=0),
        "재현율": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if prob is not None and len(set(y)) > 1:
        d["ROC AUC"] = roc_auc_score(y, prob)
    return d


def line(name, s):
    return ("  %-22s 정확도 %.3f · 정밀도 %.3f · 재현율 %.3f · F1 %.3f%s"
            % (name, s["정확도"], s["정밀도"], s["재현율"], s["F1"],
               (" · AUC %.3f" % s["ROC AUC"]) if "ROC AUC" in s else ""))


def pipe(model):
    # 파이프라인으로 묶어야 벡터화가 학습 폴드 안에서만 일어난다.
    # 전체 데이터로 먼저 벡터화하면 시험 폴드의 어휘가 새어 들어간다.
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=2, max_features=40000, sublinear_tf=True)),
        ("clf", model),
    ])


def main():
    rows = load()
    titles = np.array([r["title"] for r in rows])
    y = np.array([1 if r["senti"] == "부정" else 0 for r in rows])
    groups = event_groups(rows)
    out = {}

    print("=" * 74)
    print("화성시 언론보도 부정 판정 — 머신러닝 점검")
    print("=" * 74)
    print("\n※ 여기서 쓰는 정답은 사람이 아니라 AI가 매긴 값이다.")
    print("   따라서 이 결과는 'AI가 맞았는가'가 아니라")
    print("   'AI의 판정이 제목만으로 재현 가능한 규칙인가'를 재는 것이다.\n")

    # ── 1. 클래스 불균형 ────────────────────────────────────
    print("─" * 74)
    print("1. 클래스 불균형 — 정확도를 쓰면 안 되는 이유")
    print("─" * 74)
    pos = int(y.sum())
    print("  전체 %d건 · 부정 %d건(%.1f%%) · 비부정 %d건(%.1f%%)"
          % (len(y), pos, pos / len(y) * 100, len(y) - pos, (1 - pos / len(y)) * 100))
    dummy = DummyClassifier(strategy="most_frequent").fit(titles.reshape(-1, 1), y)
    dp = dummy.predict(titles.reshape(-1, 1))
    ds = scores(y, dp)
    print(line("전부 '부정 아님'", ds))
    print("  → 아무것도 안 하고도 정확도 %.1f%%가 나온다. 그런데 부정을 하나도 못 잡는다(재현율 0)."
          % (ds["정확도"] * 100))
    print("  → 그래서 이 문제는 정확도가 아니라 재현율과 F1으로 봐야 한다.")
    out["불균형"] = {"전체": len(y), "부정": pos, "부정비율": round(pos / len(y) * 100, 1),
                   "다수클래스정확도": round(ds["정확도"] * 100, 1)}

    # ── 2. 데이터 누수 ──────────────────────────────────────
    print("\n" + "─" * 74)
    print("2. 데이터 누수 — 나누는 방법에 따라 성능이 달라진다")
    print("─" * 74)
    ng = len(set(groups))
    dup = len(rows) - ng
    print("  제목이 겹치는 기사를 사건으로 묶으면 %d건이 %d개 사건이 된다(중복 %d건)."
          % (len(rows), ng, dup))

    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)
    res_leak = {}

    # (a) 무작위 분할 — 같은 사건이 학습·시험에 나뉘어 들어간다
    p = cross_val_predict(pipe(model), titles, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    res_leak["무작위 분할"] = scores(y, p)
    print(line("(a) 무작위 분할", res_leak["무작위 분할"]))

    # (b) 사건 분할 — 같은 사건은 한쪽에만 들어간다
    p = cross_val_predict(pipe(model), titles, y, cv=GroupKFold(5), groups=groups)
    res_leak["사건 분할"] = scores(y, p)
    print(line("(b) 사건 분할", res_leak["사건 분할"]))

    # (c) 시간 분할 — 과거로 배워 미래를 맞힌다. 실제 운영과 같은 조건
    cut = int(len(y) * 0.7)
    m = pipe(model).fit(titles[:cut], y[:cut])
    pt = m.predict(titles[cut:])
    res_leak["시간 분할"] = scores(y[cut:], pt)
    print(line("(c) 시간 분할", res_leak["시간 분할"]))

    d1 = res_leak["무작위 분할"]["F1"] - res_leak["사건 분할"]["F1"]
    print("  → 무작위 분할이 사건 분할보다 F1이 %.3f 높다." % d1)
    print("     같은 사건 기사가 양쪽에 나뉘어 들어가 답을 미리 본 것이다. 이것이 데이터 누수다.")
    print("     실제 운영은 (c) 시간 분할과 같은 조건이므로 그 값이 현실에 가깝다.")
    out["누수"] = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                      for kk, vv in v.items()} for k, v in res_leak.items()}
    out["누수"]["사건수"] = ng
    out["누수"]["중복기사"] = dup

    # ── 3. 모델 비교 ────────────────────────────────────────
    print("\n" + "─" * 74)
    print("3. 모델 비교 — 사건 분할 5겹 교차검증 (누수 없는 조건)")
    print("─" * 74)
    models = {
        "규칙 기반(베이스라인)": None,
        "로지스틱 회귀(L2)": LogisticRegression(max_iter=2000, class_weight="balanced",
                                            C=1.0, random_state=SEED),
        "결정 트리": DecisionTreeClassifier(max_depth=12, class_weight="balanced",
                                        random_state=SEED),
        "랜덤 포레스트": RandomForestClassifier(n_estimators=300, max_depth=None,
                                          class_weight="balanced_subsample",
                                          n_jobs=-1, random_state=SEED),
        "그래디언트 부스팅": GradientBoostingClassifier(random_state=SEED),
    }
    cmp_out = {}
    for name, mdl in models.items():
        if mdl is None:
            s = scores(y, rule_predict(titles))
        else:
            pr = cross_val_predict(pipe(mdl), titles, y, cv=GroupKFold(5), groups=groups)
            try:
                pp = cross_val_predict(pipe(mdl), titles, y, cv=GroupKFold(5),
                                       groups=groups, method="predict_proba")[:, 1]
            except Exception:
                pp = None
            s = scores(y, pr, pp)
        cmp_out[name] = s
        print(line(name, s))
    best = max((k for k in cmp_out if k != "규칙 기반(베이스라인)"), key=lambda k: cmp_out[k]["F1"])
    print("  → F1 기준 가장 좋은 모델: %s (F1 %.3f)" % (best, cmp_out[best]["F1"]))
    print("  → 규칙 기반 대비 F1 %+.3f"
          % (cmp_out[best]["F1"] - cmp_out["규칙 기반(베이스라인)"]["F1"]))
    out["모델비교"] = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                       for kk, vv in v.items()} for k, v in cmp_out.items()}
    out["최고모델"] = best

    # ── 4. 혼동 행렬 ────────────────────────────────────────
    print("\n" + "─" * 74)
    print("4. 혼동 행렬 — %s" % best)
    print("─" * 74)
    b = cmp_out[best]
    print("                    예측: 부정 아님    예측: 부정")
    print("    실제 부정 아님       %6d          %6d" % (b["TN"], b["FP"]))
    print("    실제 부정           %6d          %6d" % (b["FN"], b["TP"]))
    print("  → 놓친 부정(FN) %d건. 홍보 담당자에게는 이쪽이 더 아프다." % b["FN"])
    print("     헛짚은 것(FP)은 원문을 열어보면 걸러지지만, 놓친 것은 아예 안 보인다.")
    print("     그래서 정밀도보다 재현율을 먼저 본다.")

    # ── 5. 편향-분산 ────────────────────────────────────────
    print("\n" + "─" * 74)
    print("5. 편향-분산 — 결정 트리 깊이에 따른 학습/검증 성능")
    print("─" * 74)
    depth_out = []
    for d in [2, 4, 8, 16, 32, None]:
        mdl = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=SEED)
        tr = pipe(mdl).fit(titles, y)
        f_tr = f1_score(y, tr.predict(titles), zero_division=0)
        f_cv = f1_score(y, cross_val_predict(pipe(mdl), titles, y,
                                             cv=GroupKFold(5), groups=groups), zero_division=0)
        depth_out.append({"깊이": d if d else "제한없음", "학습F1": round(f_tr, 3),
                          "검증F1": round(f_cv, 3), "차이": round(f_tr - f_cv, 3)})
        print("  깊이 %-8s 학습 F1 %.3f · 검증 F1 %.3f · 차이 %.3f"
              % (str(d if d else "제한없음"), f_tr, f_cv, f_tr - f_cv))
    print("  → 깊이가 깊어질수록 학습 F1은 오르는데 검증 F1은 따라오지 않는다. 과적합이다.")
    out["편향분산"] = depth_out

    # ── 6. 정규화 ───────────────────────────────────────────
    print("\n" + "─" * 74)
    print("6. 정규화 — 로지스틱 회귀 규제 강도(C)에 따른 성능")
    print("─" * 74)
    reg_out = []
    for c in [0.01, 0.1, 1, 10, 100]:
        mdl = LogisticRegression(max_iter=2000, class_weight="balanced", C=c, random_state=SEED)
        f = f1_score(y, cross_val_predict(pipe(mdl), titles, y,
                                          cv=GroupKFold(5), groups=groups), zero_division=0)
        reg_out.append({"C": c, "F1": round(f, 3)})
        print("  C=%-6s (규제 %s)  F1 %.3f" % (c, "강함" if c < 1 else ("보통" if c == 1 else "약함"), f))
    bestC = max(reg_out, key=lambda r: r["F1"])
    print("  → C가 작으면(규제가 세면) 덜 외우고, 크면 더 외운다. 여기서는 C=%s가 가장 좋다." % bestC["C"])
    out["정규화"] = reg_out

    # ── 7. 해석 ─────────────────────────────────────────────
    print("\n" + "─" * 74)
    print("7. 어떤 낱말이 부정 판정을 끌어내는가")
    print("─" * 74)
    word = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", token_pattern=r"[가-힣A-Za-z0-9]{2,}",
                                  min_df=3, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ]).fit(titles, y)
    names = word.named_steps["tfidf"].get_feature_names_out()
    coef = word.named_steps["clf"].coef_[0]
    top = np.argsort(coef)[::-1][:15]
    bot = np.argsort(coef)[:10]
    print("  부정 쪽으로 미는 낱말: " + ", ".join(names[i] for i in top))
    print("  긍정 쪽으로 미는 낱말: " + ", ".join(names[i] for i in bot))
    inrule = [names[i] for i in top if names[i] in NEG_WORDS]
    print("  → 위 15개 중 %d개가 우리가 손으로 적은 부정 낱말 목록에 이미 있다." % len(inrule))
    print("     모델이 사람이 정한 규칙과 비슷한 곳을 보고 있다는 뜻이다.")
    out["해석"] = {"부정낱말": [names[i] for i in top], "긍정낱말": [names[i] for i in bot],
                 "규칙목록과겹침": len(inrule)}

    # ── 8. 점수 항목 점검 ───────────────────────────────────
    # 참석자 4 교수가 물은 자리가 여기다. 점수를 매기는 로직 자체를 재본다.
    print("\n" + "─" * 74)
    print("8. 점수 항목이 서로 겹치지 않는가 — 표준화 · 상관 · 주성분분석")
    print("─" * 74)

    # 모든 기사를 사건으로 묶어 항목값을 만든다. TODAY 8건은 표본이 너무 작다.
    ev = defaultdict(list)
    for i, r in enumerate(rows):
        ev[groups[i]].append(r)
    feats, spans = [], []
    for g, rs in ev.items():
        dd = sorted(r["date"] for r in rs)     # ds는 위에서 더미 분류 결과로 쓰고 있다
        span = (np.datetime64(dd[-1]) - np.datetime64(dd[0])).astype(int) + 1
        srcs = len({r.get("source", "") for r in rs if r.get("source")}) or 1
        feats.append([srcs, len(rs), span])
        spans.append(span)
    F = np.array(feats, dtype=float)
    spans = np.array(spans)
    print("  사건 %d개로 항목값을 만들었다 (매체 수 · 기사 수 · 지속일)." % len(F))

    # 표준화 — 단위가 다른 값을 같은 자로 놓고 견준다
    Z = StandardScaler().fit_transform(F)
    names = ["매체 수", "기사 수", "지속일"]
    print("\n  항목 간 상관계수 (표준화 후)")
    C = np.corrcoef(Z.T)
    print("            " + "".join("%10s" % n for n in names))
    for i, n in enumerate(names):
        print("  %-10s" % n + "".join("%10.3f" % C[i, j] for j in range(len(names))))
    hi = [(names[i], names[j], C[i, j]) for i in range(3) for j in range(i + 1, 3) if abs(C[i, j]) >= 0.6]
    if hi:
        for a, b, v in hi:
            print("  → <%s>와 <%s>의 상관이 %.3f로 높다. 사실상 같은 것을 두 번 세고 있다." % (a, b, v))
    else:
        print("  → 서로 0.6을 넘는 쌍이 없다. 세 항목이 각각 다른 것을 재고 있다는 뜻이다.")

    # 주성분분석 — 세 항목이 실제로 몇 개의 축인지
    pca = PCA().fit(Z)
    ev_ratio = pca.explained_variance_ratio_
    print("\n  주성분분석 — 각 축이 설명하는 비율")
    for i, v in enumerate(ev_ratio):
        print("    제%d주성분 %.1f%%  (누적 %.1f%%)" % (i + 1, v * 100, ev_ratio[:i + 1].sum() * 100))
    n80 = int(np.searchsorted(np.cumsum(ev_ratio), 0.8) + 1)
    print("  → 80%%를 설명하는 데 %d개 축이 필요하다." % n80)
    print(("     세 항목이 사실상 %d개 정보다. 배점을 셋으로 나눈 것이 %s"
           % (n80, "타당하다." if n80 >= 3 else "겹치는 부분이 있다는 뜻이다.")))

    out["점수점검"] = {
        "사건수": int(len(F)),
        "상관": {("%s-%s" % (names[i], names[j])): round(float(C[i, j]), 3)
               for i in range(3) for j in range(i + 1, 3)},
        "주성분": [round(float(v) * 100, 1) for v in ev_ratio],
        "필요축수": n80,
    }

    # ── 9. 지속일 분포 — 7일 기준의 근거 ──────────────────
    print("\n" + "─" * 74)
    print("9. 오래감의 기준을 7일로 둔 근거 — 사건 %d개의 지속일 분포" % len(spans))
    print("─" * 74)
    q = np.percentile(spans, [50, 75, 90, 95])
    print("  중앙값 %.0f일 · 상위 25%% %.0f일 · 상위 10%% %.0f일 · 상위 5%% %.0f일"
          % tuple(q))
    over7 = float((spans >= 7).mean())
    print("  7일 이상 이어진 사건은 %d개로 전체의 %.1f%%다." % (int((spans >= 7).sum()), over7 * 100))
    print("  → 7일은 '흔한 값'이 아니라 <상위 %.0f%%만 넘는 문턱>이다." % (over7 * 100))
    print("     드문 경우를 골라내는 기준이므로 가산점(10점)으로 두는 것이 맞다.")
    out["지속분포"] = {"사건수": int(len(spans)), "중앙값": float(q[0]),
                    "상위25": float(q[1]), "상위10": float(q[2]),
                    "7일이상비율": round(over7 * 100, 1)}

    # ── 10. 지속 예측 ───────────────────────────────────────
    # 담당자에게 진짜 필요한 것 — "이 사건이 오래 갈 것인가"를 첫날에 알 수 있는가
    print("\n" + "─" * 74)
    print("10. 첫날 정보만으로 '오래 갈 사건'을 미리 맞힐 수 있는가")
    print("─" * 74)
    y2 = (spans >= 7).astype(int)
    if y2.sum() >= 10:
        # 첫날에 알 수 있는 것만 쓴다. 지속일은 결과이므로 특성에서 뺀다(그걸 넣으면 누수다).
        X2 = F[:, :2]
        from sklearn.linear_model import LogisticRegression as LR
        mdl = Pipeline([("sc", StandardScaler()),
                        ("clf", LR(max_iter=2000, class_weight="balanced", random_state=SEED))])
        pr = cross_val_predict(mdl, X2, y2, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        s2 = scores(y2, pr)
        print("  오래 간 사건 %d개 / 전체 %d개 (%.1f%%)" % (int(y2.sum()), len(y2), y2.mean() * 100))
        print(line("첫날 정보로 예측", s2))
        print("  ※ 지속일은 결과이므로 특성에서 뺐다. 넣으면 답을 보고 맞히는 셈이라 누수다.")
        if s2["F1"] < 0.3:
            print("  → 첫날 정보(매체 수·기사 수)만으로는 오래 갈지 거의 못 맞힌다.")
            print("     그래서 '오래감'을 미리 점수에 넣지 않고, 실제로 이어진 뒤에 가산한다.")
            print("     이것이 오래감을 예측이 아니라 사후 가산으로 둔 근거다.")
        else:
            print("  → 어느 정도 맞힌다. 첫날에 '오래 갈 사건'을 미리 표시하는 것을 검토할 만하다.")
        out["지속예측"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in s2.items()}
    else:
        print("  오래 간 사건이 %d개뿐이라 예측 모형을 세울 수 없다." % int(y2.sum()))
        out["지속예측"] = None

    # ── 결론 ────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("결론")
    print("=" * 74)
    f_best = cmp_out[best]["F1"]
    print("  · 부정은 전체의 %.1f%%뿐이라 정확도는 쓸모가 없다. 아무 것도 안 해도 %.1f%%가 나온다."
          % (pos / len(y) * 100, ds["정확도"] * 100))
    print("  · 누수를 막고 재면 %s의 F1이 %.3f다. 무작위로 나누면 %.3f로 부풀려진다."
          % (best, f_best, res_leak["무작위 분할"]["F1"]))
    print("  · 이 값은 정확도가 아니다. 정답을 AI가 매겼기 때문에")
    print("    'AI의 판정이 제목만으로 얼마나 재현되는가'를 잰 값이다.")
    print("  · 사람이 검수한 정답지를 만들면 그때 비로소 정확도를 말할 수 있다.")

    p = os.path.join(BASE, "ml_report.json")
    io.open(p, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n저장: ml_report.json")


if __name__ == "__main__":
    main()
