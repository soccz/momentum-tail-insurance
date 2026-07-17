"""
P30 — 6라운드 게이트의 확정 major 봉합: 한국−미국 꼬리 증분 차이의 공식 검정.

지적: "한국 유의·타국 비유의"는 차이의 검정이 아니다(§8.4 "확정" 서술의 근거 부족).
→ 동일 달력 창(2001-12+)에서 한국 본편과 미국 극단 10분위의 ML 꼬리 증분 차
  D = ΔES5_KR(ENS−RW126) − ΔES5_US(ENS−RW126)
  를 공통 달력월 블록 부트스트랩(blk=12, B=5,000, seed 42 — 두 시장에 같은 월 인덱스 적용,
  시장 간 동시상관 보존)으로 검정한다. HARX도 병기.

미국 예측·수익 시계열은 p28과 문자 동일 파이프라인으로 재현(게이트: us_leaderboard 재현).
산출물: output/tables/p30_kr_us_diff.csv (원장 E37).
"""
import io, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 42; rng = np.random.default_rng(SEED)
ROOT = Path("/mnt/20t/졸업논문")
sr = lambda r: np.sqrt(12) * np.nanmean(r) / np.nanstd(r)
def es5(r): r = r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1, int(0.05 * len(r)))]) * 100

# ---- 미국: p28과 동일 재현 ----
wd = pd.read_csv(ROOT / "data/us/us_wml_daily.csv", index_col=0, parse_dates=True).iloc[:, 0]
wm = pd.read_csv(ROOT / "data/us/us_wml_monthly.csv", index_col=0, parse_dates=True).iloc[:, 0]
wm.index = wm.index.to_period("M")
def load_ff_monthly():
    try:
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
        df.index = pd.PeriodIndex(df.index, freq="M"); print("[FF] 라이브 취득")
        return df / 100.0
    except Exception as e:
        raise SystemExit(f"FF 취득 실패: {e}")
ff = load_ff_monthly(); mkt = ff["mktrf"] + ff["rf"]
g = wd.groupby(wd.index.to_period("M"))
RV = g.apply(lambda x: (x ** 2).sum()); SEMI = g.apply(lambda x: (x[x < 0] ** 2).sum())
RW126me = (21 * (wd ** 2).rolling(126, min_periods=126).mean()).groupby(wd.index.to_period("M")).last()
df = pd.DataFrame(index=RV.index)
df["rv"] = RV; df["l1"] = np.log(RV)
df["l3"] = np.log(RV.rolling(3).mean()); df["l6"] = np.log(RV.rolling(6).mean()); df["l12"] = np.log(RV.rolling(12).mean())
df["semir"] = (SEMI / RV).clip(0, 1)
df["mvol6"] = mkt.rolling(6).std(); df["mret1"] = mkt; df["mret12"] = mkt.rolling(12).sum()
cummkt = (1 + mkt).cumprod(); df["mdd"] = (cummkt / cummkt.cummax() - 1).reindex(df.index)
df["bear"] = (mkt.rolling(24).sum() < 0).astype(float).reindex(df.index)
df["rw126"] = RW126me; df["target"] = np.log(RV.shift(-1)); df["rv_next"] = RV.shift(-1)
df["wml_next"] = wm.reindex(df.index).shift(-1)
df = df.dropna(subset=["l12", "mret12", "bear", "target", "wml_next", "rw126"]).copy()
n = len(df)
FEAT = ["l1", "l3", "l6", "l12", "semir", "mvol6", "mret1", "mret12", "mdd", "bear"]
HARF = ["l1", "l3", "l12"]; HARXF = HARF + ["bear", "mvol6", "mdd", "semir"]
INIT, STEP = 120, 12
X = df[FEAT].values; ylog = df["target"].values
def fit_predict(cols, model):
    idx = [FEAT.index(c) for c in cols]; pred = np.full(n, np.nan)
    for s in range(INIT, n, STEP):
        tr = slice(0, s)
        lo, hi = np.log(df["rv"].values[tr].min() * 0.5), np.log(df["rv"].values[tr].max() * 2)
        model.fit(X[tr][:, idx], ylog[tr]); te = slice(s, min(s + STEP, n))
        pred[te] = np.exp(np.clip(model.predict(X[te][:, idx]), lo, hi))
    return pred
fc = {"RW126": df["rw126"].values, "HAR": fit_predict(HARF, LinearRegression()),
      "HARX": fit_predict(HARXF, LinearRegression()),
      "Ridge": fit_predict(FEAT, RidgeCV(alphas=np.logspace(-3, 3, 20))),
      "HGB": fit_predict(FEAT, HistGradientBoostingRegressor(max_depth=3, max_iter=200, random_state=SEED))}
fc["ENS"] = np.nanmean([fc["HAR"], fc["Ridge"], fc["HGB"]], axis=0)
TGT = 0.12 / np.sqrt(12)
oos = np.arange(INIT, n); wn = df["wml_next"].values
US = pd.DataFrame({m: (TGT / np.sqrt(f) * wn)[oos] for m, f in fc.items()}, index=df.index[oos])
ref = pd.read_csv(ROOT / "output/tables/us_leaderboard.csv", index_col=0)
for m in ["RW126", "ENS", "HARX"]:
    assert abs(sr(US[m].values) - ref.loc[m, "Sharpe"]) < 0.03 and abs(es5(US[m].values) - ref.loc[m, "ES5"]) < 0.3, f"US 재현 실패 {m}"
print("[게이트] US 전체창 재현 ✓")

# ---- 한국: 본편 시계열 ----
fk = pd.read_csv(ROOT / "data/processed/ml_forecasts.csv", index_col=0); fk.index = pd.PeriodIndex(fk.index, freq="M")
W = pd.read_csv(ROOT / "data/processed/p3_weights.csv", index_col=0); W.index = pd.PeriodIndex(W.index, freq="M")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0); ds.index = pd.PeriodIndex(ds.index, freq="M")
wml = ds["tgt_wml_next"].reindex(fk.index)
KR = pd.DataFrame({m: W[m].values * wml.values for m in W.columns}, index=fk.index)

# ---- 공통 달력월 정렬 + 쌍별 차이 부트스트랩 ----
common = KR.index.intersection(US.index)
KRc, USc = KR.loc[common], US.loc[common]
nC = len(common); B = 5000
print(f"[공통 창] {common[0]}~{common[-1]} n={nC}")
def bidx():
    return np.concatenate([np.arange(s, s + 12) % nC for s in rng.integers(0, nC, size=int(np.ceil(nC / 12)))])[:nC]
rows = {}
for m in ["ENS", "HARX"]:
    d_kr = es5(KRc[m].values) - es5(KRc["RW126"].values)
    d_us = es5(USc[m].values) - es5(USc["RW126"].values)
    D = d_kr - d_us
    bs = []
    for _ in range(B):
        ix = bidx()
        bs.append((es5(KRc[m].values[ix]) - es5(KRc["RW126"].values[ix]))
                  - (es5(USc[m].values[ix]) - es5(USc["RW126"].values[ix])))
    bs = np.array(bs)
    rows[m] = dict(dKR=d_kr, dUS=d_us, D=D, p_le0=float((bs <= 0).mean()),
                   ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)))
    print(f"  {m}: ΔES5 한국 {d_kr:+.2f} − 미국 {d_us:+.2f} = D {D:+.2f}%p · P(D≤0)={rows[m]['p_le0']:.3f} · CI [{rows[m]['ci_lo']:+.2f}, {rows[m]['ci_hi']:+.2f}]")
out = pd.DataFrame(rows).T.round(4); out["n_common"] = nC
out.to_csv(ROOT / "output/tables/p30_kr_us_diff.csv")
print("\n[GATES] US 재현 ✓ · p30_kr_us_diff.csv 저장 ✓")
