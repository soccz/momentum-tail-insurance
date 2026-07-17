"""
P6 — 확장 실험 (설계 §2 Layer 3·4).

Layer 3 "무엇을 예측할 것인가": 분산 대신 하방준분산(Patton-Sheppard)을 타깃으로 스케일.
  크래시가 하방이므로, 하방위험 타깃팅이 왜도·최악월을 더 개선하는지.
Layer 4 "분자까지": Daniel-Moskowitz(2016) 동적가중 w ∝ μ̂/σ̂² — μ̂를 상수(학습평균) vs Ridge로.
  한국 소표본에서 μ 예측이 되는지(아마 안 됨)를 정직하게 보고.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path("/mnt/20t/졸업논문")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0); ds.index = pd.PeriodIndex(ds.index, freq="M")
fc = pd.read_csv(ROOT / "data/processed/ml_forecasts.csv", index_col=0); fc.index = pd.PeriodIndex(fc.index, freq="M")
ds = ds.dropna()
FEAT = [c for c in ds.columns if not c.startswith("tgt_")]
X_all = ds[FEAT].values
wml = ds["tgt_wml_next"].values
n = len(ds); FIRST_TRAIN, RETRAIN = 120, 12
oos = np.arange(FIRST_TRAIN, n)
TGT_M = 0.12 / np.sqrt(12)

def perf(r):
    r = r[np.isfinite(r)]
    return dict(sharpe=np.sqrt(12) * r.mean() / r.std(), skew=stats.skew(r),
                worst=r.min() * 100, mdd=((1 + pd.Series(r)).cumprod() /
                (1 + pd.Series(r)).cumprod().cummax() - 1).min() * 100)

def expanding_ridge(y, log_target=True):
    """확장윈도우 Ridge (P2와 동일 프로토콜). log_target: 분산계열은 log, 수익률은 level."""
    pred = np.full(n, np.nan)
    yy = np.log(y) if log_target else y
    for t0 in range(FIRST_TRAIN, n, RETRAIN):
        tr, te = np.arange(t0), np.arange(t0, min(t0 + RETRAIN, n))
        sc = StandardScaler().fit(X_all[tr]); Z = sc.transform(X_all)
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[tr], yy[tr])
        if log_target:
            resid = yy[tr] - m.predict(Z[tr])
            p = np.exp(m.predict(Z[te]) + resid.var() / 2)
            lo, hi = y[tr].min() * 0.5, y[tr].max() * 2
            pred[te] = np.clip(p, lo, hi)
        else:
            pred[te] = m.predict(Z[te])
    return pred

# ---------- Layer 3: 하방준분산 타깃 ----------
semivar = (ds["tgt_semineg_next"].values / 100) ** 2 / 12          # 연율 vol% → 월 하방준분산
var_total = ds["tgt_var_next"].values
pred_semi = expanding_ridge(semivar)
pred_var = expanding_ridge(var_total)
rw126_var = ds["fvar_126"].values
rw126_semi = np.full(n, np.nan)                                     # RW형 하방: 직전 126d 대응은 피처에 없어 6m semineg 사용
semi_feat = (ds["semineg_6m"].values / 100) ** 2 / 12

rows = {}
rows["RW126 (분산타깃, BS15)"] = perf(TGT_M / np.sqrt(rw126_var[oos]) * wml[oos])
rows["RW-semi (하방타깃, 무학습)"] = perf(TGT_M / np.sqrt(2 * semi_feat[oos]) * wml[oos])
rows["Ridge (분산타깃)"] = perf(TGT_M / np.sqrt(pred_var[oos]) * wml[oos])
rows["Ridge-semi (하방타깃)"] = perf(TGT_M / np.sqrt(2 * pred_semi[oos]) * wml[oos])
l3 = pd.DataFrame(rows).T
print("===== Layer 3 — 무엇을 예측할 것인가 (분산 vs 하방준분산) =====")
print(l3.round(3).to_string())

# ---------- Layer 4: 동적 μ/σ² (DM16) ----------
mu_ridge = expanding_ridge(wml, log_target=False)
r2_mu = 1 - np.nansum((wml[oos] - mu_ridge[oos]) ** 2) / np.nansum(
    (wml[oos] - pd.Series(wml).expanding().mean().shift(1).values[oos]) ** 2)
print(f"\n===== Layer 4 — 동적 μ/σ² (DM16) =====")
print(f"μ 예측 OOS R² = {r2_mu*100:.1f}%  (0 이하 = 수익률 예측 불가; 정직 보고)")

sig2 = ds["fvar_126"].values                                    # σ̂² 고정(RW126) — μ만의 효과 분리
rows4 = {}
for nm, mu in [("DM(μ=학습평균)", None), ("DM(μ=Ridge)", mu_ridge)]:
    w = np.full(n, np.nan)
    for t0 in range(FIRST_TRAIN, n, RETRAIN):
        te = np.arange(t0, min(t0 + RETRAIN, n))
        mu_t = np.full(len(te), wml[:t0].mean()) if mu is None else mu[te]
        lam = TGT_M * np.nanmean(np.sqrt(sig2[:t0])) / max(abs(np.nanmean(wml[:t0])), 1e-9) / \
              np.nanmean(sig2[:t0])                                  # 학습창에서 평균 레버리지 정규화
        w[te] = np.clip(lam * mu_t / sig2[te], 0, 2)
    rows4[nm] = perf(w[oos] * wml[oos]) | {"평균L": np.nanmean(w[oos])}
rows4["BS15형(σ만, RW126)"] = perf(TGT_M / np.sqrt(sig2[oos]) * wml[oos]) | \
                              {"평균L": np.nanmean(TGT_M / np.sqrt(sig2[oos]))}
l4 = pd.DataFrame(rows4).T
print(l4.round(3).to_string())

# ---------- 저장 + 게이트 ----------
l3.round(4).to_csv(ROOT / "output/tables/p6_layer3_semivar.csv")
l4.round(4).to_csv(ROOT / "output/tables/p6_layer4_dynamic.csv")
assert np.isfinite(l3["sharpe"]).all() and np.isfinite(l4["sharpe"]).all()
print("\n[GATES] 전 전략 유한 성과 ✓ · 표 2종 저장 ✓")
