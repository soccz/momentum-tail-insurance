"""
헤드라인 수치 게이트 — 논문의 핵심 정량 주장이 정본 CSV(output/tables/)에서 재현되는지 검증.

원장(docs/05_results_ledger.md)의 고정값과 대조한다. 이 게이트가 통과하면
"README·논문의 헤드라인 수치는 전부 이 저장소의 CSV로 소급된다"가 기계적으로 보증된다.
CI(.github/workflows/verify.yml)에서 매 푸시마다 실행.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "output/tables"
checks = []

def check(name, actual, expected, tol):
    ok = abs(actual - expected) <= tol
    checks.append((name, actual, expected, tol, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {actual:.4f} (기대 {expected}±{tol})")
    return ok

# ── 발견 0: 재현 — 한국 모멘텀은 약하고, BS15 관리는 작동한다 (E00·E01)
t3 = pd.read_csv(T / "table3.csv", index_col=0)
check("WML 샤프 (전체표본)", t3.loc["WML", "Sharpe"], 0.166, 0.02)
check("WML 왜도", t3.loc["WML", "Skewness"], -1.165, 0.05)
check("WML* 샤프 (BS15 관리)", t3.loc["WML*", "Sharpe"], 0.26, 0.02)
check("WML* IR", t3.loc["WML*", "Info ratio"], 0.229, 0.03)
assert t3.loc["WML", "Minimum"] < -94, "최악월 -94.9% 소실"

# ── 발견 1: 예측은 좋아진다 (E03·E33)
lb = pd.read_csv(T / "p2_leaderboard.csv", index_col=0)
check("ENS QLIKE", lb.loc["ENS", "QLIKE"], 0.253, 0.005)
check("RW126 QLIKE (벤치마크)", lb.loc["RW126", "QLIKE"], 0.334, 0.005)
check("ENS 표본외 R2 (%)", lb.loc["ENS", "OOS_R2"], 38.9, 0.5)
cw = pd.read_csv(T / "p26_cw_rc.csv", index_col=0)
check("중첩+다중성 동시보정 FWER p", float(cw["rc_p"].iloc[0]), 0.0024, 0.003)

# ── 발견 1b: 샤프비율은 따라오지 않는다 (E12·E34)
sl = pd.read_csv(T / "p27_tail_slope.csv").set_index("axis")
assert sl.loc["sharpe", "lo"] <= 0 <= sl.loc["sharpe", "hi"], "샤프 축 기울기 CI가 0을 포함해야 함"
assert sl.loc["es5", "lo"] <= 0 <= sl.loc["es5", "hi"], "꼬리 축 기울기 CI가 0을 포함해야 함"
print("PASS  전이곡선 기울기 CI (샤프·꼬리 축 모두 0 포함)")

# ── 발견 2: 개선은 꼬리로 간다 (E10·E17)
pf = pd.read_csv(T / "p3_portfolio.csv", index_col=0)
check("ENS ΔES5 vs RW126 (%p, +=개선)", pf.loc["ENS", "ES5"] - pf.loc["RW126", "ES5"], 0.86, 0.03)
rw = pd.read_csv(T / "p16_romano_wolf.csv", index_col=0)
assert rw.loc["ENS", "rw_p"] > 0.10, "ENS는 FWER 비생존이어야 함 (정직 보고의 대상)"
assert rw.loc["HAR", "rw_p"] < 0.05 and rw.loc["Lasso", "rw_p"] < 0.05, "HAR·Lasso는 FWER 생존"
check("HARX Romano–Wolf 보정 p (탐색적)", rw.loc["HARX", "rw_p"], 0.0224, 0.005)
print("PASS  Romano–Wolf 위계 (ENS 비생존 · HAR/Lasso 생존)")

# ── 발견 3: 경계 — 한국−미국 차이 자체가 유의 (E37)
d = pd.read_csv(T / "p30_kr_us_diff.csv", index_col=0)
check("한국−미국 꼬리 증분 차 D (ENS, %p)", d.loc["ENS", "D"], 2.01, 0.05)
assert d.loc["ENS", "p_le0"] < 0.01, "차이 검정 p<0.01"
# ML−HAR 분해: 순수 ML 결합의 꼬리 이득은 0 (E35)
mh = pd.read_csv(T / "p28_ml_vs_har.csv", index_col=0)
assert mh.loc["ENS", "dES5_vs_HAR"] < 0.1, "ENS−HAR 꼬리 이득은 ~0이어야 함 (구조가 판매자)"
print("PASS  경계·분해 (D=+2.01 p<0.01 · ENS−HAR ≈ 0)")

fails = [c for c in checks if not c[4]]
print(f"\n{'='*50}\n게이트 {len(checks)}건 중 {len(checks)-len(fails)}건 PASS" + (f", {len(fails)}건 FAIL" if fails else " — 전부 통과"))
if fails:
    raise SystemExit(1)
