"""
P28 — 4라운드 게이트(E34 후속)의 확정 major 3건 봉합 계산.

(1) 미국 극단 10분위 — 한국과 동일 평가창(2001-12+) 재평가.
    p19_us_transfer.py의 피처·모형·확장창을 문자 재현(데이터는 repo 내 data/us/ + FF 월별;
    월별 파일이 없으면 일별 F-F를 월복리 집계). 게이트: 전체 OOS 리더보드가 저장본
    us_leaderboard.csv와 근사 재현(샤프 ±0.03, ES5 ±0.3%p). 그 후 2001-12+ 창의
    ΔES5(ENS·HARX vs RW126) + 대응 원형 블록 부트스트랩 p(blk=12, B=5,000).
(2) 한국 본편 — ML의 HAR 초과 꼬리 증분: ΔES5(ENS−HAR, Ridge−HAR, HARX−HAR) + 동일 부트스트랩 p.
(3) 한국 본편 — CRRA CE(γ=10) ML증분(ENS−RW126)의 블록 부트스트랩 95% CI와 P(≤0).

산출물: output/tables/p28_us_matched_window.csv · p28_ml_vs_har.csv · p28_ce_ci.csv (원장 E35).
"""
import io, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

SEED = 42; rng = np.random.default_rng(SEED)
ROOT = Path("/mnt/20t/졸업논문")
sr = lambda r: np.sqrt(12) * np.nanmean(r) / np.nanstd(r)
def es5(r): r = r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1, int(0.05 * len(r)))]) * 100

# ============ (1) 미국 — 동일 평가창 ============
wd = pd.read_csv(ROOT / "data/us/us_wml_daily.csv", index_col=0, parse_dates=True).iloc[:, 0]
wm = pd.read_csv(ROOT / "data/us/us_wml_monthly.csv", index_col=0, parse_dates=True).iloc[:, 0]
wm.index = wm.index.to_period("M")

def load_ff_monthly():
    try:  # Ken French 공개 라이브러리 (E29의 FRED 라이브 대조와 같은 공개출처 원칙)
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
        lines = z.read(z.namelist()[0]).decode("latin1").splitlines()
        rows = []
        for l in lines:
            t = l.split(","); k = t[0].strip()
            if k.isdigit() and len(k) == 6:
                try: rows.append([k] + [float(x) for x in t[1:5]])
                except Exception: break
            elif rows: break
        df = pd.DataFrame(rows, columns=["m", "mktrf", "smb", "hml", "rf"]).set_index("m")
        df.index = pd.PeriodIndex(df.index, freq="M")
        print("[FF] 월별 파일 라이브 취득")
        return df / 100.0
    except Exception as e:
        print(f"[FF] 라이브 실패({e}) → 일별 파일 월복리 집계로 대체")
        d = pd.read_csv(ROOT / "추가/data/raw/F-F_Research_Data_Factors_daily.csv", skiprows=4)
        d = d.rename(columns={d.columns[0]: "date"})
        d = d[d["date"].astype(str).str.strip().str.len() == 8]
        d["date"] = pd.to_datetime(d["date"].astype(str).str.strip())
        d = d.set_index("date").astype(float) / 100.0
        m = pd.DataFrame({
            "mktrf": (1 + d["Mkt-RF"] + d["RF"]).groupby(d.index.to_period("M")).prod()
                     - (1 + d["RF"]).groupby(d.index.to_period("M")).prod(),
            "rf": (1 + d["RF"]).groupby(d.index.to_period("M")).prod() - 1})
        return m

ff = load_ff_monthly()
mkt = ff["mktrf"] + ff["rf"]

g = wd.groupby(wd.index.to_period("M"))
RV = g.apply(lambda x: (x ** 2).sum()); SEMI = g.apply(lambda x: (x[x < 0] ** 2).sum())
RW126 = (21 * (wd ** 2).rolling(126, min_periods=126).mean())
RW126_me = RW126.groupby(RW126.index.to_period("M")).last()

df = pd.DataFrame(index=RV.index)
df["rv"] = RV; df["l1"] = np.log(RV)
df["l3"] = np.log(RV.rolling(3).mean()); df["l6"] = np.log(RV.rolling(6).mean()); df["l12"] = np.log(RV.rolling(12).mean())
df["semir"] = (SEMI / RV).clip(0, 1)
df["mvol6"] = mkt.rolling(6).std(); df["mret1"] = mkt; df["mret12"] = mkt.rolling(12).sum()
cummkt = (1 + mkt).cumprod(); df["mdd"] = (cummkt / cummkt.cummax() - 1).reindex(df.index)
df["bear"] = (mkt.rolling(24).sum() < 0).astype(float).reindex(df.index)
df["rw126"] = RW126_me
df["target"] = np.log(RV.shift(-1)); df["rv_next"] = RV.shift(-1)
df["wml_next"] = wm.reindex(df.index).shift(-1)
df = df.dropna(subset=["l12", "mret12", "bear", "target", "wml_next", "rw126"]).copy()
n = len(df); print(f"[US 피처] {df.index[0]}~{df.index[-1]} n={n}")

FEAT = ["l1", "l3", "l6", "l12", "semir", "mvol6", "mret1", "mret12", "mdd", "bear"]
HARF = ["l1", "l3", "l12"]; HARXF = HARF + ["bear", "mvol6", "mdd", "semir"]
INIT, STEP = 120, 12
X = df[FEAT].values; ylog = df["target"].values
def fit_predict(cols, model):
    idx = [FEAT.index(c) for c in cols]
    pred = np.full(n, np.nan)
    for s in range(INIT, n, STEP):
        tr = slice(0, s)
        lo, hi = np.log(df["rv"].values[tr].min() * 0.5), np.log(df["rv"].values[tr].max() * 2)
        model.fit(X[tr][:, idx], ylog[tr])
        te = slice(s, min(s + STEP, n))
        pred[te] = np.exp(np.clip(model.predict(X[te][:, idx]), lo, hi))
    return pred

fc = {"RW126": df["rw126"].values}
fc["HAR"] = fit_predict(HARF, LinearRegression())
fc["HARX"] = fit_predict(HARXF, LinearRegression())
fc["Ridge"] = fit_predict(FEAT, RidgeCV(alphas=np.logspace(-3, 3, 20)))
fc["HGB"] = fit_predict(FEAT, HistGradientBoostingRegressor(max_depth=3, max_iter=200, random_state=SEED))
fc["ENS"] = np.nanmean([fc["HAR"], fc["Ridge"], fc["HGB"]], axis=0)

TGT = 0.12 / np.sqrt(12)
oos = np.arange(INIT, n); wn = df["wml_next"].values
RETUS = {m: (TGT / np.sqrt(f) * wn)[oos] for m, f in fc.items()}

# 게이트: 전체 OOS가 저장본과 근사 재현
ref = pd.read_csv(ROOT / "output/tables/us_leaderboard.csv", index_col=0)
for m in ["RW126", "ENS", "HARX"]:
    ds_, dr_ = sr(RETUS[m]) - ref.loc[m, "Sharpe"], es5(RETUS[m]) - ref.loc[m, "ES5"]
    print(f"[게이트] {m}: Δ샤프 {ds_:+.3f} · ΔES5 {dr_:+.3f}")
    assert abs(ds_) < 0.03 and abs(dr_) < 0.3, f"US 재현 실패 {m}"

# 2001-12+ 동창 평가
mask = df.index[oos] >= pd.Period("2001-12")
sub = {m: r[mask] for m, r in RETUS.items()}
nS = int(mask.sum()); B = 5000
def bidx(nn):
    return np.concatenate([np.arange(s, s + 12) % nn for s in rng.integers(0, nn, size=int(np.ceil(nn / 12)))])[:nn]
rows = {}
for m in ["ENS", "HARX", "Ridge", "HAR"]:
    do = es5(sub[m]) - es5(sub["RW126"])
    bs = np.array([es5(sub[m][ix]) - es5(sub["RW126"][ix]) for _ in range(B) if (ix := bidx(nS)) is not None])
    rows[m] = dict(dES5=do, p=float((bs <= 0).mean()))
raw_sub = wn[oos][mask]
do_m = es5(sub["RW126"]) - es5(raw_sub)
bs_m = np.array([es5(sub["RW126"][ix]) - es5(raw_sub[ix]) for _ in range(B) if (ix := bidx(nS)) is not None])
rows["mgmt"] = dict(dES5=do_m, p=float((bs_m <= 0).mean()))
rows["raw_sharpe"] = dict(dES5=np.nan, p=np.nan)
us = pd.DataFrame(rows).T.round(4)
us.loc["raw_sharpe", "dES5"] = round(sr(raw_sub), 4)
us["n_oos"] = nS
# 동창 QLIKE(RW126→ENS) — 예측 품질의 창 정합 문서화
rvn = df["rv_next"].values
mq = np.zeros(n, bool); mq[oos] = mask
ql = lambda f: float(np.mean(rvn[mq] / f[mq] - np.log(rvn[mq] / f[mq]) - 1))
us.loc["qlike_rw", "dES5"] = round(ql(fc["RW126"]), 4)
us.loc["qlike_ens", "dES5"] = round(ql(fc["ENS"]), 4)
us.to_csv(ROOT / "output/tables/p28_us_matched_window.csv")
print(f"    동창 QLIKE: RW {ql(fc['RW126']):.3f} → ENS {ql(fc['ENS']):.3f} ({(1 - ql(fc['ENS']) / ql(fc['RW126'])) * 100:.0f}% 개선) · 관리 ΔES5 {do_m:+.2f}(p={rows['mgmt']['p']:.3f})")
print(f"\n[1] US 동창(2001-12+, n={nS}): 무관리 샤프 {sr(raw_sub):+.2f} · "
      + " · ".join(f"{m} ΔES5 {rows[m]['dES5']:+.2f}(p={rows[m]['p']:.3f})" for m in ["ENS", "HARX"]))
print(f"    관리(RW126−무관리) ΔES5 {es5(sub['RW126']) - es5(raw_sub):+.2f}%p")

# ============ (2)(3) 한국 본편 ============
fk = pd.read_csv(ROOT / "data/processed/ml_forecasts.csv", index_col=0); fk.index = pd.PeriodIndex(fk.index, freq="M")
W = pd.read_csv(ROOT / "data/processed/p3_weights.csv", index_col=0); W.index = pd.PeriodIndex(W.index, freq="M")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0); ds.index = pd.PeriodIndex(ds.index, freq="M")
wml = ds["tgt_wml_next"].reindex(fk.index).values
RET = {m: W[m].values * wml for m in W.columns}
nK = len(fk)

rows2 = {}
for m in ["ENS", "Ridge", "HARX"]:
    do = es5(RET[m]) - es5(RET["HAR"])
    bs = np.array([es5(RET[m][ix]) - es5(RET["HAR"][ix]) for _ in range(B) if (ix := bidx(nK)) is not None])
    rows2[m] = dict(dES5_vs_HAR=do, p_improve=float((bs <= 0).mean()), p_worse=float((bs >= 0).mean()))
    print(f"[2] {m} vs HAR: ΔES5 {do:+.2f}%p · P(≤0)={rows2[m]['p_improve']:.3f} · P(≥0)={rows2[m]['p_worse']:.3f}")
pd.DataFrame(rows2).T.round(4).to_csv(ROOT / "output/tables/p28_ml_vs_har.csv")

g10 = 10
def ce(r):
    r = r[np.isfinite(r)]
    m = np.mean((1 + r) ** (1 - g10)) ** (1 / (1 - g10)) - 1
    return ((1 + m) ** 12 - 1) * 100
do_ce = ce(RET["ENS"]) - ce(RET["RW126"])
bs_ce = np.array([ce(RET["ENS"][ix]) - ce(RET["RW126"][ix]) for _ in range(B) if (ix := bidx(nK)) is not None])
lo, hi = np.percentile(bs_ce, [2.5, 97.5])
p_ce = float((bs_ce <= 0).mean())
pd.DataFrame({"dCE_gamma10": [do_ce], "ci_lo": [lo], "ci_hi": [hi], "p_le0": [p_ce]}).round(4).to_csv(
    ROOT / "output/tables/p28_ce_ci.csv", index=False)
print(f"[3] ΔCE(γ=10, ENS−RW126) = {do_ce:+.2f}%p/yr · 95% CI [{lo:+.2f}, {hi:+.2f}] · P(≤0)={p_ce:.3f}")
print("\n[GATES] US 전체창 재현 ✓ · 산출물 3종 저장 ✓")
