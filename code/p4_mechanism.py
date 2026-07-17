"""
P4 — 메커니즘: 왜/어디서 이득이 나는가 (블랙박스 금지).

(1) Eq(7) 성분 분해 — 총분산 = 시장성분(β²·RV_mkt) + 고유성분. 성분별 예측가능성(RW vs Ridge).
    논문 Table 4의 한국+ML판: 고유성분이 크고 더 예측 가능해야 스케일링 논리가 선다.
(2) 교차팩터 플라시보 — 같은 파이프라인을 RMRF·SMB·HML에 적용. 예측가능성이 낮은 팩터에서
    이득이 작아야 메커니즘(예측가능한 위험→관리 가치) 입증.
(3) 위기 타이밍 해부 — ML(ENS)과 RW126의 레버리지가 크래시 직전에 어떻게 갈라지는가.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path("/mnt/20t/졸업논문")
P = lambda idx: idx.to_period("M")
FIRST_TRAIN, RETRAIN = 120, 12

wd = pd.read_csv(ROOT / "data/processed/wml_daily.csv", parse_dates=["Date"]).set_index("Date")
ffd = pd.read_csv(ROOT / "data/processed/ff_own_daily.csv", parse_dates=["Date"]).set_index("Date")
ffm = pd.read_csv(ROOT / "data/processed/ff_own_monthly.csv", parse_dates=["month"]).set_index("month")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0); ds.index = pd.PeriodIndex(ds.index, freq="M")

# ---------- (1) Eq(7) 성분 분해 ----------
r_w = wd["wml"]; r_m = ffd["rmrf"].reindex(r_w.index)
cov = r_w.rolling(126).cov(r_m); varm = r_m.rolling(126).var(); varw = r_w.rolling(126).var()
beta = (cov / varm)
beta_me = beta.groupby(P(beta.index)).last()                       # t시점 β (t까지 정보)
mvar_next = (r_m ** 2).groupby(P(r_m.index)).sum().shift(-1)       # t+1월 시장 실현분산
wvar_next = (r_w ** 2).groupby(P(r_w.index)).sum().shift(-1)
mkt_comp_next = (beta_me ** 2) * mvar_next                         # 시장성분 (t β 고정)
spec_next = (wvar_next - mkt_comp_next).clip(lower=wvar_next * 0.05)  # 고유성분 (β노이즈로 음수 나는 달은 총분산의 5% 하한)

share = (mkt_comp_next / wvar_next).clip(0, 1)
print(f"[분해] 시장성분 비중: 평균 {share.mean()*100:.1f}%  (미국 논문 23%) → 고유 {100-share.mean()*100:.1f}%")

def oos_r2(target, feats):
    """확장윈도우 Ridge vs RW(자기 126d 대응) vs 확장평균 — OOS R² (%)"""
    df = pd.concat([target.rename("y"), feats], axis=1).dropna()
    y = np.log(df["y"].values); X = df.drop(columns="y").values
    n = len(df); Fhat = {k: np.full(n, np.nan) for k in ("Ridge", "RW", "Mean")}
    rw_proxy = df.iloc[:, 1].values                                # 첫 피처 = 자기 자신의 126d 실현치
    for t0 in range(FIRST_TRAIN, n, RETRAIN):
        tr, te = np.arange(t0), np.arange(t0, min(t0 + RETRAIN, n))
        sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[tr], y[tr])
        resid = y[tr] - m.predict(Z[tr])
        raw_pred = np.exp(m.predict(Z[te]) + resid.var() / 2)
        lo, hi = df["y"].values[tr].min() * 0.5, df["y"].values[tr].max() * 2.0
        Fhat["Ridge"][te] = np.clip(raw_pred, lo, hi)   # 학습창 실현범위 밖 외삽 차단(β²계열 폭발 방지)
        Fhat["RW"][te] = rw_proxy[te]
        Fhat["Mean"][te] = df["y"].values[:t0].mean()
    oos = np.arange(FIRST_TRAIN, n); act = df["y"].values[oos]
    sse_b = ((act - Fhat["Mean"][oos]) ** 2).sum()
    return {k: (1 - ((act - Fhat[k][oos]) ** 2).sum() / sse_b) * 100 for k in ("RW", "Ridge")}, len(oos)

f126 = lambda s: (21 * (s ** 2).rolling(126).mean()).groupby(P(s.index)).last()   # Eq(5)형 자기 예측치
feats_common = ds[["vol_1m", "vol_6m", "mkt_vol_1m", "bear", "rebound", "semi_ratio_6m"]]
feats_full = ds[[c for c in ds.columns if not c.startswith("tgt_")]]   # 분해엔 P2와 동일한 풀 피처

comp_rows = {}
spec_proxy = (f126(r_w) - (beta_me ** 2) * f126(r_m)).clip(lower=f126(r_w) * 0.05)
for nm, tgt, proxy in [("총분산", wvar_next, f126(r_w)),
                       ("시장성분", mkt_comp_next, (beta_me ** 2) * f126(r_m)),
                       ("고유성분", spec_next, spec_proxy)]:
    proxy = proxy.clip(lower=1e-12)
    fx = pd.concat([proxy.rename("proxy126"), feats_full], axis=1)
    (res, n_oos) = oos_r2(tgt, fx)
    comp_rows[nm] = res | {"n": n_oos}
comp = pd.DataFrame(comp_rows).T
print("\n[성분별 OOS R² (%) — RW(자기126d) vs Ridge]")
print(comp.round(1).to_string())

# ---------- (2) 교차팩터 플라시보 ----------
TGT_M = 0.12 / np.sqrt(12)
plc_rows = {}
for fac in ["wml", "rmrf", "smb", "hml"]:
    r_d = wd["wml"] if fac == "wml" else ffd[fac]
    var_next = (r_d ** 2).groupby(P(r_d.index)).sum().shift(-1)
    ret_next = (ds["tgt_wml_next"] if fac == "wml"
                else ((1 + r_d).groupby(P(r_d.index)).prod() - 1).shift(-1))
    own = pd.DataFrame({"proxy126": f126(r_d),
                        "vol_1m": np.sqrt(252/21*(r_d**2).rolling(21).sum()).groupby(P(r_d.index)).last()*100})
    fx = pd.concat([own, feats_common[["mkt_vol_1m", "bear", "rebound"]]], axis=1)
    df = pd.concat([var_next.rename("y"), fx, ret_next.rename("ret")], axis=1).dropna()
    (res, n_oos) = oos_r2(df["y"], df.drop(columns=["y", "ret"]))
    # 성과: RW vs Ridge 스케일 포트폴리오 (동일 OOS)
    y = np.log(df["y"].values); X = df.drop(columns=["y", "ret"]).values; n = len(df)
    pred = {k: np.full(n, np.nan) for k in ("RW", "Ridge")}
    for t0 in range(FIRST_TRAIN, n, RETRAIN):
        tr, te = np.arange(t0), np.arange(t0, min(t0 + RETRAIN, n))
        sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[tr], y[tr])
        resid = y[tr] - m.predict(Z[tr])
        pred["Ridge"][te] = np.exp(m.predict(Z[te]) + resid.var() / 2)
        pred["RW"][te] = df["proxy126"].values[te]
    oos = np.arange(FIRST_TRAIN, n); ret = df["ret"].values
    sh = {}
    for k in ("RW", "Ridge"):
        pr = TGT_M / np.sqrt(pred[k][oos]) * ret[oos]
        sh[k] = np.sqrt(12) * np.nanmean(pr) / np.nanstd(pr)
    raw = np.sqrt(12) * ret[oos].mean() / ret[oos].std()
    plc_rows[fac.upper()] = dict(R2_RW=res["RW"], R2_Ridge=res["Ridge"],
                                 sharpe_raw=raw, sharpe_RW=sh["RW"], sharpe_Ridge=sh["Ridge"],
                                 dSharpe_mgmt=sh["RW"] - raw, dSharpe_ML=sh["Ridge"] - sh["RW"])
plc = pd.DataFrame(plc_rows).T
print("\n[교차팩터 플라시보 — 예측가능성 vs 관리·ML 이득]")
print(plc.round(3).to_string())

# ---------- (3) 위기 타이밍 ----------
w = pd.read_csv(ROOT / "data/processed/p3_weights.csv", index_col=0)
w.index = pd.PeriodIndex(w.index, freq="M")
wret = ds["tgt_wml_next"].reindex(w.index)
crash = wret[wret < -0.10].index                                   # OOS 내 크래시월(t+1이 크래시)
print(f"\n[타이밍] OOS 크래시월(WML<-10%): {len(crash)}개 — {[str(c+1) for c in crash[:8]]}...")
tm = pd.DataFrame({
    "RW126_직전L": w.loc[crash, "RW126"], "ENS_직전L": w.loc[crash, "ENS"],
    "HARX_직전L": w.loc[crash, "HARX"]})
print(tm.describe().loc[["mean", "50%"]].round(3).to_string())
for ev, s, e in [("GFC", "2008-06", "2009-06"), ("COVID", "2020-01", "2020-12")]:
    sl = w.loc[s:e]
    print(f"[{ev}] 평균 L — RW126 {sl['RW126'].mean():.2f} · HARX {sl['HARX'].mean():.2f} · ENS {sl['ENS'].mean():.2f}")

# ---------- 저장 + 게이트 ----------
comp.round(2).to_csv(ROOT / "output/tables/p4_components.csv")
plc.round(4).to_csv(ROOT / "output/tables/p4_placebo.csv")
tm.round(4).to_csv(ROOT / "output/tables/p4_crash_timing.csv")
assert share.mean() < 0.5, "고유성분 우위 가정 위배"
assert comp.loc["고유성분", "Ridge"] > comp.loc["시장성분", "Ridge"] - 5, "성분 예측가능성 역전(점검 필요)"
print("\n[GATES] 고유성분 우위 ✓ · 산출물 3종 저장 ✓")
