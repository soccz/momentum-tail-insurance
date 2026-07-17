"""P19: 무위험수익률 대안(CD 3개월) 강건성 — E13 콜금리 선택의 민감도.

rf_monthly.csv의 rf_cd3m(CD 91일물 월환산)으로 시장 초과수익을 재계산해
표 1의 '시장(초과) 샤프 0.22'와 '모멘텀=시장 초과의 77%' 서술이 RF 선택에
둔감함을 확인한다. WML은 self-financing이라 영향 없음(정의 확인용 병기).

출력: output/tables/p19_rf_alt.csv
게이트: |샤프(CD) − 샤프(콜)| < 0.03 이면 PASS.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/mnt/20t/졸업논문")
ffm = pd.read_csv(ROOT / "data/processed/ff_own_monthly.csv", parse_dates=["month"]).set_index("month")
wm = pd.read_csv(ROOT / "data/processed/wml_monthly.csv", parse_dates=["month"]).set_index("month")
rf = pd.read_csv(ROOT / "data/processed/rf_monthly.csv", parse_dates=["month"]).set_index("month")
ffm.index = ffm.index.to_period("M"); wm.index = wm.index.to_period("M")
rf.index = rf.index.to_period("M")

df = pd.concat([ffm["rmrf"], wm["wml"], rf[["rf_call", "rf_cd3m"]]], axis=1).dropna()
df["ex_call"] = df["rmrf"] - df["rf_call"]
df["ex_cd"] = df["rmrf"] - df["rf_cd3m"]

def stats(s):
    mu, sd = s.mean() * 12, s.std() * np.sqrt(12)
    return dict(mean_ann=mu * 100, vol_ann=sd * 100, sharpe=mu / sd,
                t=s.mean() / s.std() * np.sqrt(len(s)))

rows = {nm: stats(df[c]) for nm, c in [
    ("시장 gross", "rmrf"), ("시장 초과(콜금리)", "ex_call"),
    ("시장 초과(CD 3개월)", "ex_cd"), ("WML (self-financing)", "wml")]}
out = pd.DataFrame(rows).T
print(f"===== P19 RF 대안 (공통표본 {df.index[0]}~{df.index[-1]}, n={len(df)}) =====")
print(out.round(3).to_string())

ratio_call = rows["WML (self-financing)"]["sharpe"] / rows["시장 초과(콜금리)"]["sharpe"]
ratio_cd = rows["WML (self-financing)"]["sharpe"] / rows["시장 초과(CD 3개월)"]["sharpe"]
print(f"\nWML/시장초과 샤프비: 콜 {ratio_call:.2f} vs CD {ratio_cd:.2f}")
spread = ((1 + df.rf_cd3m) ** 12 - 1).mean() - ((1 + df.rf_call) ** 12 - 1).mean()
print(f"CD−콜 평균 스프레드: 연 {spread*100:.2f}%p")

out.round(4).to_csv(ROOT / "output/tables/p19_rf_alt.csv")
diff = abs(rows["시장 초과(CD 3개월)"]["sharpe"] - rows["시장 초과(콜금리)"]["sharpe"])
print(f"\n[게이트] |샤프(CD)−샤프(콜)| = {diff:.3f} < 0.03 → {'PASS' if diff < 0.03 else 'FAIL'}")
