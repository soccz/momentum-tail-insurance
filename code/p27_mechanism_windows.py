"""
P27 — 3라운드 게이트(E33)의 확정 major 2건 봉합 계산.

(1) 꼬리 축 전이곡선 기울기 — 샤프 축(p11 (4))과 동일한 조인트 원형 블록 부트스트랩을
    ES5–R² 기울기에 적용. 양(+)의 R² 9점 내부에서 꼬리 축도 단조인지 검정.
(2) 표 S2 조건부 위험–수익의 표본창 분해 — 전체표본(게이트: 표 S2 재현) /
    1998-10 제외 / 표본외 창(2001-12+, 분위 경계 창 내 재추정). Q5 붕괴가 어느 표본의 성질인지 확정.
(3) 표 D2 ENS 열 채움 — 약세장/비약세장 조건부 평균·샤프 (p10과 동일 정의).

산출물: output/tables/p27_tail_slope.csv · p27_quintile_windows.csv · p27_bear_ens.csv (원장 E34).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42; rng = np.random.default_rng(SEED)
ROOT = Path("/mnt/20t/졸업논문")
fc = pd.read_csv(ROOT / "data/processed/ml_forecasts.csv", index_col=0); fc.index = pd.PeriodIndex(fc.index, freq="M")
W = pd.read_csv(ROOT / "data/processed/p3_weights.csv", index_col=0); W.index = pd.PeriodIndex(W.index, freq="M")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0); ds.index = pd.PeriodIndex(ds.index, freq="M")
mw = pd.read_csv(ROOT / "data/processed/wml_monthly.csv", parse_dates=["month"]).set_index("month"); mw.index = mw.index.to_period("M")
ff = pd.read_csv(ROOT / "data/processed/ff_own_monthly.csv", parse_dates=["month"]).set_index("month"); ff.index = ff.index.to_period("M")
wml = ds["tgt_wml_next"].reindex(fc.index).values
act = fc["actual_var"].values
n = len(fc)
RET = {m: W[m].values * wml for m in W.columns}
sr = lambda r: np.sqrt(12) * np.nanmean(r) / np.nanstd(r)
es5 = lambda r: np.mean(np.sort(r[np.isfinite(r)])[:max(1, int(.05 * len(r[np.isfinite(r)])))]) * 100

# ---------- (1) 꼬리 축 기울기 조인트 부트스트랩 (p11 (4)와 동일 설계, y만 ES5) ----------
core = ["RW126", "EWMA", "HAR", "HARX", "Ridge", "Lasso", "RF", "HGB", "ENS"]
blk = 12
def bidx():
    return np.concatenate([np.arange(s, s + blk) % n for s in rng.integers(0, n, size=int(np.ceil(n / blk)))])[:n]
def slopes_of(ix):
    a = act[ix]; sse_b = ((a - fc["ExpandingMean"].values[ix]) ** 2).sum()
    xs, ys_sr, ys_es = [], [], []
    for m in core:
        f = fc[m].values[ix]
        xs.append((1 - ((a - f) ** 2).sum() / sse_b) * 100)
        ys_sr.append(sr(RET[m][ix])); ys_es.append(es5(RET[m][ix]))
    return stats.linregress(xs, ys_sr).slope, stats.linregress(xs, ys_es).slope
obs_sr, obs_es = slopes_of(np.arange(n))
boot = np.array([slopes_of(bidx()) for _ in range(2000)])
lo_sr, hi_sr = np.percentile(boot[:, 0], [2.5, 97.5]); lo_es, hi_es = np.percentile(boot[:, 1], [2.5, 97.5])
print(f"[1] 기울기(9점 조인트, B=2000) — 샤프 축 {obs_sr:+.4f} CI[{lo_sr:+.4f},{hi_sr:+.4f}] · "
      f"꼬리(ES5) 축 {obs_es:+.4f} CI[{lo_es:+.4f},{hi_es:+.4f}] (단위: R² 1%p당)")
pd.DataFrame({"axis": ["sharpe", "es5"], "slope": [obs_sr, obs_es],
              "lo": [lo_sr, lo_es], "hi": [hi_sr, hi_es]}).round(5).to_csv(ROOT / "output/tables/p27_tail_slope.csv", index=False)
assert lo_sr <= 0 <= hi_sr, "샤프 축 CI가 p11과 불일치"

# ---------- (2) 조건부 위험–수익의 표본창 분해 (p16 (5)와 동일 정의) ----------
series = {"WML": mw["wml"], "시장": ff["rmrf"], "SMB": ff["smb"], "HML": ff["hml"]}
def quints(df):
    q5 = pd.qcut(df["rv"], 5, labels=False)
    return [df["r"][q5 == i].mean() * 1200 for i in range(5)]
rows = {}
for nm, s in series.items():
    s = s.dropna(); risk = s.rolling(6).var().shift(1)
    df = pd.concat([s.rename("r"), risk.rename("rv")], axis=1).dropna()
    full = quints(df)
    ex98 = quints(df.drop(index=pd.Period("1998-10"), errors="ignore"))
    oos = quints(df.loc["2001-12":"2026-03"])
    rows[nm] = dict(full_Q1=full[0], full_Q5=full[4], ex9810_Q5=ex98[4], oos_Q1=oos[0], oos_Q5=oos[4])
    print(f"[2] {nm:>4}: 전체 Q1→Q5 {full[0]:+.1f}→{full[4]:+.1f} · 98-10 제외 Q5 {ex98[4]:+.1f} · 표본외 Q1→Q5 {oos[0]:+.1f}→{oos[4]:+.1f}")
qt = pd.DataFrame(rows).T.round(2)
qt.to_csv(ROOT / "output/tables/p27_quintile_windows.csv")
assert abs(qt.loc["WML", "full_Q1"] - 5.6) < 0.5 and abs(qt.loc["WML", "full_Q5"] - (-12.3)) < 0.5, "표 S2 재현 실패"

# ---------- (3) ENS 약세장 조건부 (p10 (a)와 동일 정의) ----------
bear = ds["bear"].reindex(fc.index).values.astype(bool)
out = {}
for m in ["ENS", "RW126", "HARX", "Ridge"]:
    out[m] = dict(bear_ann=np.nanmean(RET[m][bear]) * 1200, nonbear_ann=np.nanmean(RET[m][~bear]) * 1200,
                  bear_sr=sr(RET[m][bear]), nonbear_sr=sr(RET[m][~bear]))
out["RAW"] = dict(bear_ann=np.nanmean(wml[bear]) * 1200, nonbear_ann=np.nanmean(wml[~bear]) * 1200,
                  bear_sr=sr(wml[bear]), nonbear_sr=sr(wml[~bear]))
be = pd.DataFrame(out).T.round(3)
be.to_csv(ROOT / "output/tables/p27_bear_ens.csv")
print("[3] 약세장 조건부 (연율%, 샤프):"); print(be.round(2).to_string())
assert abs(be.loc["RW126", "bear_ann"] - 0.41) < 0.1 and abs(be.loc["HARX", "bear_ann"] - 2.41) < 0.1, "p10 재현 실패"
print("\n[GATES] 샤프 축 CI 정합 ✓ · 표 S2 재현 ✓ · p10 약세장 재현 ✓ · 산출물 3종 저장 ✓")
