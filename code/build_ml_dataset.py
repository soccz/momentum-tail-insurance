"""
P1 — ML 데이터셋 구축 (docs/04_research_design.md §3, §6-P1).

행 = 월말 t (형성 시점). 피처 = t까지의 정보만. 타깃 = t+1월의 실현분산/수익.
출처: 검증 게이트 통과본(data/processed/wml_daily.csv 등) + 패널(횡단면 피처).

무결성 게이트 (전부 통과해야 저장):
  G1  지속성: corr(vol_6m_t, tgt_vol_next) >= 0.5   (Table 2의 예측가능성 재확인)
  G2  표본: 행 >= 400, 전-NaN 컬럼 없음
  G3  베이스라인 재현: 피처의 fvar_126로 만든 WML* 샤프가 0.26 ± 0.02
      (데이터셋이 기존 검증 결과와 정확히 이어짐을 보증 — look-ahead 구조 검증 겸용)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/mnt/20t/졸업논문")
OUT = ROOT / "data/processed/ml_dataset.csv"
MOM = "/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet"
PBR = ROOT / "data/processed/pbr_monthly.parquet"

# ---------- 1) 일별 시계열 파생 피처 (전부 trailing window → look-ahead 구조적으로 불가) ----------
wd = pd.read_csv(ROOT / "data/processed/wml_daily.csv", parse_dates=["Date"]).set_index("Date")
ffd = pd.read_csv(ROOT / "data/processed/ff_own_daily.csv", parse_dates=["Date"]).set_index("Date")
wm = pd.read_csv(ROOT / "data/processed/wml_monthly.csv", parse_dates=["month"]).set_index("month")

r = wd["wml"]; rw = wd["winner"]; rl = wd["loser"]; rm = ffd["rmrf"].reindex(r.index)

def ann_vol(sq_sum, days):        # 창 내 제곱합 → 연율 변동성(%)
    return np.sqrt(252.0 / days * sq_sum) * 100.0

sq = r ** 2
feat_d = pd.DataFrame(index=r.index)
for lab, w in [("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)]:
    feat_d[f"vol_{lab}"] = ann_vol(sq.rolling(w, min_periods=w).sum(), w)

neg = (r.where(r < 0) ** 2).fillna(0.0)     # 하방 준분산 (Patton-Sheppard)
pos = (r.where(r > 0) ** 2).fillna(0.0)
feat_d["semineg_6m"] = ann_vol(neg.rolling(126, min_periods=126).sum(), 126)
feat_d["semi_ratio_6m"] = (neg.rolling(126).sum() /
                           (neg.rolling(126).sum() + pos.rolling(126).sum()))
feat_d["vol_win_6m"] = ann_vol((rw ** 2).rolling(126, min_periods=126).sum(), 126)
feat_d["vol_los_6m"] = ann_vol((rl ** 2).rolling(126, min_periods=126).sum(), 126)
feat_d["vol_gap_6m"] = feat_d["vol_los_6m"] - feat_d["vol_win_6m"]   # 패자쪽 위험 초과

feat_d["mkt_vol_1m"] = ann_vol((rm ** 2).rolling(21, min_periods=21).sum(), 21)
feat_d["mkt_vol_6m"] = ann_vol((rm ** 2).rolling(126, min_periods=126).sum(), 126)
mkt_idx = (1 + rm.fillna(0)).cumprod()
feat_d["mkt_ret_1m"] = mkt_idx.pct_change(21)
feat_d["mkt_ret_12m"] = mkt_idx.pct_change(252)
feat_d["mkt_drawdown"] = mkt_idx / mkt_idx.cummax() - 1.0
feat_d["bear"] = (mkt_idx.pct_change(504) < 0).astype(float)          # DM(2016): 과거 24개월 음수
feat_d["rebound"] = ((feat_d["bear"] > 0) & (feat_d["mkt_ret_1m"] > 0)).astype(float)

# Eq(5) 그대로의 월분산 예측치 — 베이스라인 재현 게이트(G3)용이자 Layer-0 피처
feat_d["fvar_126"] = 21.0 * sq.rolling(126, min_periods=126).mean()

P = lambda idx: idx.to_period("M")
feat_me = feat_d.groupby(P(feat_d.index)).last()                      # 월말 t 시점 값

# ---------- 2) 타깃: t+1월의 실현분산·하방준분산·WML 수익 ----------
by_m = sq.groupby(P(sq.index))
tgt = pd.DataFrame({
    "tgt_var_next": by_m.sum().shift(-1),                              # 월 실현분산
    "tgt_vol_next": (np.sqrt(12 * by_m.sum()) * 100).shift(-1),        # 연율 변동성(%)
    "tgt_semineg_next": (np.sqrt(12 * neg.groupby(P(neg.index)).sum()) * 100).shift(-1),
})
wml_p = wm["wml"].copy(); wml_p.index = P(wm.index)
tgt["tgt_wml_next"] = wml_p.shift(-1).reindex(tgt.index)               # t+1월 WML 수익

# ---------- 3) 횡단면 피처 (형성 시점 t의 패널 정보) ----------
print("[panel] loading for cross-sectional features ...", flush=True)
raw = pd.read_parquet(MOM, columns=["Date", "Code", "AdjClose", "MarketCap"])
raw = raw[raw["AdjClose"] > 0]
price = raw.pivot_table(index="Date", columns="Code", values="AdjClose", aggfunc="last").sort_index().loc["1990":]
cap = raw.pivot_table(index="Date", columns="Code", values="MarketCap", aggfunc="last").reindex_like(price)
me = price.groupby(P(price.index)).apply(lambda x: x.index[-1]); me = pd.DatetimeIndex(me.values)
pbr = pd.read_parquet(PBR).pivot_table(index="Date", columns="Code", values="PBR", aggfunc="last").sort_index()

xs = {}
for k in range(12, len(me)):
    T = me[k]
    sig = price.loc[me[k-1]] / price.loc[me[k-12]] - 1
    w = cap.loc[T]
    ok = sig.notna() & w.notna() & (w > 0)
    if ok.sum() < 30: continue
    s = sig[ok]
    dec = pd.qcut(s, 10, labels=False, duplicates="drop")
    row = {"disp_signal": s.std(), "n_stocks": float(len(s))}
    # 승자·패자 decile의 B/M 스프레드 (가장 최근 PBR ≤ T)
    p_idx = pbr.index[pbr.index <= T]
    if len(p_idx):
        bm = 1.0 / pbr.loc[p_idx[-1]]
        bl = bm.reindex(dec.index[dec == dec.min()]).median()
        bw = bm.reindex(dec.index[dec == dec.max()]).median()
        if np.isfinite(bl) and np.isfinite(bw):
            row["bm_spread"] = bl - bw          # 패자(가치) − 승자(성장) 기울기
    xs[T.to_period("M")] = row
xs = pd.DataFrame(xs).T

# ---------- 4) 조립 + 게이트 ----------
ds = feat_me.join(xs, how="inner").join(tgt, how="inner").dropna(subset=["tgt_vol_next", "tgt_wml_next"])
ds.index.name = "month"

g1 = ds["vol_6m"].corr(ds["tgt_vol_next"])
g2_rows, g2_allnan = len(ds), [c for c in ds.columns if ds[c].isna().all()]
sigma_hat = np.sqrt(ds["fvar_126"])                                    # t시점 예측 → t+1월에 적용
wml_star = (0.12 / np.sqrt(12)) / sigma_hat * ds["tgt_wml_next"]
g3 = np.sqrt(12) * wml_star.mean() / wml_star.std()

print(f"[G1] persistence corr(vol_6m, tgt_vol_next) = {g1:.3f}  (>=0.5)")
print(f"[G2] rows = {g2_rows} (>=400) | all-NaN cols = {g2_allnan or '없음'}")
print(f"[G3] baseline WML* sharpe from features = {g3:.3f}  (0.26 ± 0.02)")
assert g1 >= 0.5 and g2_rows >= 400 and not g2_allnan and abs(g3 - 0.26) <= 0.02, "게이트 실패"

ds.round(6).to_csv(OUT)
print(f"\n[out] {OUT}  {ds.shape[0]}행 × {ds.shape[1]}열  "
      f"({ds.index[0]} ~ {ds.index[-1]})")
print("피처:", [c for c in ds.columns if not c.startswith('tgt_')])
