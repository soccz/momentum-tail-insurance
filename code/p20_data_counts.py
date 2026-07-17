"""P20: §3.1 데이터 기술 수치의 정본 산출 — 종목수·상폐수·필터적중.

2026-07-14 데이터 감사에서 논문 §3.1의 4,685(종목)·1,840(상폐)이 현재
mom_prices.parquet에서 재현되지 않음을 확인(이전 데이터 빈티지 산출 추정).
build_factor.py의 유니버스 필터(AdjClose 유효·>0, 1990-01-01+)를 그대로 적용한
재현 가능한 정본 카운트를 고정한다. 필터적중 1,410은 기존과 정확 일치.

출력: output/tables/p20_data_counts.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/20t/졸업논문")
SRC = "/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet"

df = pd.read_parquet(SRC, columns=["Date", "Code", "AdjClose"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["AdjClose"])
df = df[df.AdjClose > 0]                      # build_factor.load_panel과 동일
adj = df.pivot_table(index="Date", columns="Code", values="AdjClose", aggfunc="last").sort_index()
adj = adj.loc[adj.index >= "1990-01-01"]
adj = adj.dropna(axis=1, how="all")

n_stocks = adj.shape[1]
last_valid = adj.apply(lambda s: s.last_valid_index())
# 상폐 정의는 p13과 동일: 마지막 유효가가 표본 종료 30일 이전
n_delisted = int((last_valid < adj.index.max() - pd.Timedelta(days=30)).sum())

ret = adj.pct_change(fill_method=None)
n_filter = int((ret.abs() > 0.5).sum().sum())
by_year = ret.abs().gt(0.5).groupby(ret.index.year).sum().sum(axis=1)
top = by_year.sort_values(ascending=False).head(4)

# 거래정지 재개일 점프 소실(와이드 pct_change 구조상 NaN) 집계 — 부록 B 공개용
has = adj.notna()
resumed = has & ~has.shift(1, fill_value=False)
resumed.iloc[0] = False
ever = has.cumsum().shift(1, fill_value=0) > 0
jump = (adj / adj.ffill().shift(1) - 1)[resumed & ever].stack()
n_resume, resume_med = len(jump), jump.median()

out = pd.DataFrame([dict(
    n_stocks=n_stocks, n_delisted=n_delisted, n_filter_hits=n_filter,
    filter_top_years=str(top.to_dict()), n_resume_lost=n_resume,
    resume_jump_median=round(float(resume_med), 3),
    panel_start=str(adj.index.min().date()), panel_end=str(adj.index.max().date()))])
out.to_csv(ROOT / "output/tables/p20_data_counts.csv", index=False)
print(out.T.to_string())
print(f"\n[게이트] 필터적중 == 1410 → {'PASS' if n_filter == 1410 else 'FAIL: ' + str(n_filter)}")
