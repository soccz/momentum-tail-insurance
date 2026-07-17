"""
P2 — 예측 경기 (docs/04_research_design.md §2 Layer 0–2, §6-P2).

타깃: 다음 달 WML 실현분산. 프로토콜(전 모델 동일):
  확장 윈도우(초기 120개월) · 연 1회 재학습 · 하이퍼파라미터는 학습창 내부 시계열 CV만.
평가: QLIKE(주지표) · RMSE(연율 vol%) · OOS R²(확장평균 벤치마크, 논문 Eq4 방식)
      · Diebold-Mariano vs RW126(=BS15 베이스라인), NW lag 3.

사전등록 모델군 (이 파일이 등록부 — 사후 추가·삭제 금지):
  [기준] ExpandingMean, RW126(BS15), EWMA(λ∈{0.7,0.8,0.9,0.94} in-window 튜닝)
  [계량] HAR(log, 1/3/12개월), HAR-X(+곰장·반등·시장분산·준분산비·leg갭)
  [ML]   Ridge, Lasso, RandomForest, HistGB, MLP, LSTM(12개월 시퀀스), ENS(HAR+Ridge+HGB 평균)

출력: data/processed/ml_forecasts.csv · output/tables/p2_leaderboard.{csv,md}
게이트: OOS n>=250 · RW126==fvar_126 정합 · 전 모델 결측 없는 리더보드.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

ROOT = Path("/mnt/20t/졸업논문")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0)
ds.index = pd.PeriodIndex(ds.index, freq="M")
ds = ds.dropna()                     # 워밍업 결측 행 제거 (초기 ~10행)

FEATURES = [c for c in ds.columns if not c.startswith("tgt_")]
y_var = ds["tgt_var_next"].values                 # 월 분산 (평가 공간)
y_log = np.log(y_var)                             # 학습 공간 (log-분산)
n = len(ds)
FIRST_TRAIN, RETRAIN = 120, 12

# HAR 회귀변수: 월분산 스케일 (연율 vol% → 월분산)
to_var = lambda v: (ds[v].values / 100.0) ** 2 / 12.0
har_X = np.log(np.column_stack([to_var("vol_1m"), to_var("vol_3m"), to_var("vol_12m")]))
harx_X = np.column_stack([har_X, ds["bear"], ds["rebound"],
                          np.log(to_var("mkt_vol_1m")), ds["semi_ratio_6m"], ds["vol_gap_6m"]])
ml_X = ds[FEATURES].values

# ---------------- 모델 구현 ----------------
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import torch
import torch.nn as nn

def smear(pred_log, resid):          # log→분산 역변환 (로그정규 보정)
    return np.exp(pred_log + resid.var() / 2.0)

def fit_linear(X, tr, model):
    m = model.fit(X[tr], y_log[tr])
    resid = y_log[tr] - m.predict(X[tr])
    return lambda te: smear(m.predict(X[te]), resid)

def fit_tree(X, tr, base, grid):
    cv = TimeSeriesSplit(3)
    g = GridSearchCV(base, grid, cv=cv, scoring="neg_mean_squared_error").fit(X[tr], y_log[tr])
    resid = y_log[tr] - g.predict(X[tr])
    return lambda te: smear(g.predict(X[te]), resid)

class LSTMNet(nn.Module):
    def __init__(self, d, h=24):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.head = nn.Linear(h, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

SEQ = 12
def fit_lstm(tr_end):
    torch.manual_seed(SEED)
    sc = StandardScaler().fit(ml_X[:tr_end])
    Z = sc.transform(ml_X)
    seqs = np.stack([Z[i - SEQ:i] for i in range(SEQ, n)])           # 행 i: t-12..t-1 → 예측대상 t
    idx = np.arange(SEQ, n)
    tr_mask = idx < tr_end
    Xtr = torch.tensor(seqs[tr_mask], dtype=torch.float32)
    ytr = torch.tensor(y_log[idx[tr_mask]], dtype=torch.float32)
    net = LSTMNet(Z.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(150):
        opt.zero_grad(); loss = nn.functional.mse_loss(net(Xtr), ytr); loss.backward(); opt.step()
    with torch.no_grad():
        resid = (ytr - net(Xtr)).numpy()
    def predict(te):
        te = np.asarray(te)
        ok = te >= SEQ
        out = np.full(len(te), np.nan)
        if ok.any():
            Xte = torch.tensor(seqs[te[ok] - SEQ], dtype=torch.float32)
            with torch.no_grad():
                out[ok] = smear(net(Xte).numpy(), resid)
        return out
    return predict

def ewma_forecast(lam):              # 재귀: s2_{t+1|t} = λ s2_t + (1-λ) RV_t
    s2 = np.full(n, np.nan); s2[0] = y_var[:12].mean()
    for t in range(1, n):
        prev = s2[t-1] if np.isfinite(s2[t-1]) else y_var[:t].mean()
        s2[t] = lam * prev + (1 - lam) * y_var[t-1]
    return s2                        # s2[t] = t시점에 선 t+1월 분산 예측(과거 RV만 사용)

EWMA_ALL = {lam: ewma_forecast(lam) for lam in (0.7, 0.8, 0.9, 0.94)}
def qlike(f, a):                     # 분산 예측 손실 (Patton 2011)
    r = a / f
    return r - np.log(r) - 1.0

# ---------------- 확장 윈도우 OOS 루프 ----------------
models = ["ExpandingMean", "RW126", "EWMA", "HAR", "HARX",
          "Ridge", "Lasso", "RF", "HGB", "MLP", "LSTM", "ENS"]
F = {m: np.full(n, np.nan) for m in models}

fit_cache = {}
for t0 in range(FIRST_TRAIN, n, RETRAIN):
    tr = np.arange(t0)                                   # 학습: 0..t0-1 (타깃까지 t0-1+1월 → t0월 이전)
    te = np.arange(t0, min(t0 + RETRAIN, n))             # 예측: 다음 12개월
    print(f"  retrain @ {ds.index[t0]}  train={len(tr)}  test={len(te)}", flush=True)

    F["ExpandingMean"][te] = y_var[:t0].mean()
    F["RW126"][te] = ds["fvar_126"].values[te]           # t시점 피처 = BS15 Eq(5) 그대로
    lam_best = min(EWMA_ALL, key=lambda l: qlike(EWMA_ALL[l][12:t0], y_var[12:t0]).mean())
    F["EWMA"][te] = EWMA_ALL[lam_best][te]

    F["HAR"][te] = fit_linear(har_X, tr, LinearRegression())(te)
    F["HARX"][te] = fit_linear(harx_X, tr, LinearRegression())(te)

    sc = StandardScaler().fit(ml_X[tr]); Z = sc.transform(ml_X)
    F["Ridge"][te] = fit_linear(Z, tr, RidgeCV(alphas=np.logspace(-3, 3, 13)))(te)
    F["Lasso"][te] = fit_linear(Z, tr, LassoCV(cv=TimeSeriesSplit(3), random_state=SEED, max_iter=5000))(te)
    F["RF"][te] = fit_tree(ml_X, tr, RandomForestRegressor(n_estimators=300, random_state=SEED),
                           {"max_depth": [3, 5, None], "min_samples_leaf": [5, 10]})(te)
    F["HGB"][te] = fit_tree(ml_X, tr, HistGradientBoostingRegressor(random_state=SEED),
                            {"learning_rate": [0.03, 0.1], "max_depth": [3, None]})(te)
    F["MLP"][te] = fit_linear(Z, tr, MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2,
                                                  max_iter=3000, random_state=SEED))(te)
    F["LSTM"][te] = fit_lstm(t0)(te)

F["ENS"] = np.nanmean(np.stack([F["HAR"], F["Ridge"], F["HGB"]]), axis=0)

# ---------------- 평가 + DM 검정 ----------------
oos = np.arange(FIRST_TRAIN, n)
def dm_vs(base, m):                  # Diebold-Mariano (QLIKE, NW lag3)
    d = qlike(F[m][oos], y_var[oos]) - qlike(F[base][oos], y_var[oos])
    d = d[np.isfinite(d)]; L = 3; g0 = d.var(); s = g0
    for l in range(1, L + 1):
        g = np.cov(d[l:], d[:-l])[0, 1]; s += 2 * (1 - l / (L + 1)) * g
    from scipy.stats import norm
    stat = d.mean() / np.sqrt(s / len(d))
    return stat, 2 * (1 - norm.cdf(abs(stat)))

rows = []
bench = y_var[oos]
sse_bench = ((bench - F["ExpandingMean"][oos]) ** 2).sum()
for m in models:
    f = F[m][oos]; ok = np.isfinite(f)
    ql = qlike(f[ok], bench[ok]).mean()
    rmse = np.sqrt((((np.sqrt(12 * f[ok]) - np.sqrt(12 * bench[ok])) * 100) ** 2).mean())
    r2 = 1 - ((bench[ok] - f[ok]) ** 2).sum() / sse_bench
    dm, p = (np.nan, np.nan) if m == "RW126" else dm_vs("RW126", m)
    rows.append(dict(model=m, QLIKE=ql, RMSE_vol=rmse, OOS_R2=r2 * 100, DM_vs_RW126=dm, p=p, n=int(ok.sum())))
lb = pd.DataFrame(rows).set_index("model").sort_values("QLIKE")

print("\n===== P2 예측 리더보드 (OOS %s ~ %s, n=%d) =====" % (ds.index[oos[0]], ds.index[oos[-1]], len(oos)))
print(lb.round(3).to_string())

# ---------------- 저장 + 게이트 ----------------
out = pd.DataFrame({"actual_var": y_var}, index=ds.index)
for m in models: out[m] = F[m]
out.iloc[oos].to_csv(ROOT / "data/processed/ml_forecasts.csv")
lb.round(4).to_csv(ROOT / "output/tables/p2_leaderboard.csv")
(ROOT / "output/tables/p2_leaderboard.md").write_text(
    f"# P2 예측 리더보드 (QLIKE 오름차순)\nOOS {ds.index[oos[0]]}~{ds.index[oos[-1]]}, n={len(oos)}\n\n"
    + lb.round(3).to_markdown())

assert len(oos) >= 250, "OOS 부족"
assert np.allclose(F["RW126"][oos], ds["fvar_126"].values[oos]), "RW126 정합 실패"
assert lb["QLIKE"].notna().all(), "리더보드 결측"
print("\n[GATES] OOS n=%d ✓ | RW126==Eq(5) ✓ | 전 모델 평가 완료 ✓" % len(oos))
