"""
P29 — 5라운드 게이트의 확정 major 봉합: 국제·구성 대조 행들의 예측 품질 문서화.

지적: 표 8의 국제 행(일본·유럽·북미·아태·한국 2×3·미국 소형주)은 ΔES5만 있어
"전이 부재"와 "예측 개선 자체의 부재"가 구분되지 않는다.
→ p21/p22/p23과 문자 동일한 파이프라인(피처·모형·확장창·seed·부트스트랩 소비 순서)을 재현해
  각 행의 QLIKE(RW126→ENS) 개선율과 표본외 R²(ENS, 확장평균 기준)를 산출하고,
  기존 ΔES5·p의 재현을 게이트로 건다(±0.03 / ±0.02).

데이터: Kenneth French 일별(data/intl/로 레포 이관) · data/processed/wml_2x3_daily·monthly.csv ·
       추가/data/raw/F-F_Research_Data_Factors_daily.csv. 산출물: p29_intl_qlike.csv (원장 E36).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/mnt/20t/졸업논문")
DI = ROOT / "data/intl"
SEED = 42
sr = lambda r: np.sqrt(12) * np.nanmean(r) / np.nanstd(r)
def es5(r): r = r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1, int(0.05 * len(r)))]) * 100

def _rows(fp, ncol):
    L = open(fp, encoding="latin1").read().splitlines()
    i0 = next(i for i, l in enumerate(L) if l.strip().startswith(",") and any(k in l for k in ("WML", "Mom", "Mkt-RF")))
    r = []
    for l in L[i0 + 1:]:
        t = l.split(","); k = t[0].strip()
        if k.isdigit() and len(k) == 8:
            try: r.append([k] + [float(x) for x in t[1:1 + ncol]])
            except Exception: break
        elif r: break
    return r
def load_wml(fp):
    r = _rows(fp, 1); s = pd.Series({x[0]: x[1] for x in r})
    s.index = pd.to_datetime(s.index, format="%Y%m%d"); return s.replace([-99.99, -999.0], np.nan).dropna() / 100.0
def load_mkt(fp):
    r = _rows(fp, 4)
    df = pd.DataFrame([x[1:] for x in r], index=[x[0] for x in r], columns=["mktrf", "smb", "hml", "rf"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d"); df = df.replace([-99.99, -999.0], np.nan).dropna() / 100.0
    return df["mktrf"] + df["rf"]
def load_25(fp):
    L = open(fp, encoding="latin1").read().splitlines()
    i0 = next(i for i, l in enumerate(L) if "Average Value Weighted Returns -- Daily" in l)
    hdr = [h.strip() for h in L[i0 + 1].split(",")]
    rows = []
    for l in L[i0 + 2:]:
        t = l.split(","); k = t[0].strip()
        if k.isdigit() and len(k) == 8:
            try: rows.append([k] + [float(x) for x in t[1:len(hdr)]])
            except Exception: break
        elif rows: break
    df = pd.DataFrame(rows, columns=["d"] + hdr[1:]).set_index("d")
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.replace([-99.99, -999.0], np.nan) / 100.0
def load_us_mkt():
    L = open(ROOT / "추가/data/raw/F-F_Research_Data_Factors_daily.csv", encoding="latin1").read().splitlines()
    i0 = next(i for i, l in enumerate(L) if l.strip().startswith(",Mkt-RF"))
    rows = []
    for l in L[i0 + 1:]:
        t = l.split(","); k = t[0].strip()
        if k.isdigit() and len(k) == 8:
            try: rows.append([k, float(t[1]), float(t[4])])
            except Exception: break
        elif rows: break
    df = pd.DataFrame(rows, columns=["d", "mktrf", "rf"]).set_index("d")
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return (df["mktrf"] + df["rf"]) / 100.0

def run(wd, mm_d_or_m, label, rng, boot_models, monthly_mkt=False, wm_override=None):
    """p21/p22/p23 공통 구조 문자 재현 + QLIKE/R² 추가. boot_models 순서 = 원 스크립트의 rng 소비 순서."""
    Pm = lambda i: i.to_period("M")
    wm = wm_override if wm_override is not None else (1 + wd).groupby(Pm(wd.index)).prod() - 1
    g = wd.groupby(Pm(wd.index)); RV = g.apply(lambda x: (x ** 2).sum()); SEMI = g.apply(lambda x: (x[x < 0] ** 2).sum())
    RW = (21 * (wd ** 2).rolling(126, min_periods=126).mean()); RWme = RW.groupby(Pm(RW.index)).last()
    mm = mm_d_or_m if monthly_mkt else (1 + mm_d_or_m).groupby(Pm(mm_d_or_m.index)).prod() - 1
    df = pd.DataFrame(index=RV.index)
    df["rv"] = RV; df["l1"] = np.log(RV); df["l3"] = np.log(RV.rolling(3).mean())
    df["l6"] = np.log(RV.rolling(6).mean()); df["l12"] = np.log(RV.rolling(12).mean()); df["semir"] = (SEMI / RV).clip(0, 1)
    df["mret1"] = mm.reindex(df.index); df["mret12"] = mm.reindex(df.index).rolling(12).sum()
    df["mvol6"] = mm.reindex(df.index).rolling(6).std()
    cm = (1 + mm).cumprod(); df["mdd"] = (cm / cm.cummax() - 1).reindex(df.index)
    df["bear"] = (mm.reindex(df.index).rolling(24).sum() < 0).astype(float)
    df["rw"] = RWme; df["tgt"] = np.log(RV.shift(-1)); df["rvn"] = RV.shift(-1); df["wmln"] = wm.reindex(df.index).shift(-1)
    FEAT = ["l1", "l3", "l6", "l12", "semir", "mret1", "mret12", "mvol6", "mdd", "bear"]
    HARF = ["l1", "l3", "l12"]; HARXF = HARF + ["bear", "mvol6", "mdd"]
    df = df.dropna(subset=FEAT + ["tgt", "wmln", "rw"]).copy(); n = len(df); INIT = 120
    X = df[FEAT].values; y = df["tgt"].values; rvcol = df["rv"].values
    def fp(cols, model):
        idx = [FEAT.index(c) for c in cols]; pred = np.full(n, np.nan)
        for s in range(INIT, n, 12):
            lo, hi = np.log(rvcol[:s].min() * 0.5), np.log(rvcol[:s].max() * 2)
            model.fit(X[:s][:, idx], y[:s]); te = slice(s, min(s + 12, n))
            pred[te] = np.exp(np.clip(model.predict(X[te][:, idx]), lo, hi))
        return pred
    oos = np.arange(INIT, n); wn = df["wmln"].values; rvn = df["rvn"].values
    fc = {"RW126": df["rw"].values,
          "HAR": fp(HARF, LinearRegression()), "HARX": fp(HARXF, LinearRegression()),
          "Ridge": fp(FEAT, RidgeCV(alphas=np.logspace(-3, 3, 20))),
          "HGB": fp(FEAT, HistGradientBoostingRegressor(max_depth=3, max_iter=150, random_state=SEED))}
    fc["ENS"] = np.nanmean([fc["HAR"], fc["Ridge"], fc["HGB"]], axis=0)
    TGT = 0.12 / np.sqrt(12); RET = {m: (TGT / np.sqrt(f) * wn)[oos] for m, f in fc.items()}
    nO = len(oos); B = 3000
    def bidx():
        return np.concatenate([np.arange(s, s + 12) % nO for s in rng.integers(0, nO, size=int(np.ceil(nO / 12)))])[:nO]
    boot = {}
    for m in boot_models:
        do = es5(RET[m]) - es5(RET["RW126"])
        bs = np.array([es5(RET[m][ix]) - es5(RET["RW126"][ix]) for _ in range(B) if (ix := bidx()) is not None])
        boot[m] = (do, float((bs <= 0).mean()))
    # 예측 품질 (추가분 — rng 미소비)
    ql = lambda f: float(np.mean(rvn[oos] / f[oos] - np.log(rvn[oos] / f[oos]) - 1))
    em = np.array([rvcol[:t].mean() if t > 0 else np.nan for t in range(n)])
    r2 = lambda f: float((1 - np.nansum((rvn[oos] - f[oos]) ** 2) / np.nansum((rvn[oos] - em[oos]) ** 2)) * 100)
    q_rw, q_ens = ql(fc["RW126"]), ql(fc["ENS"])
    return dict(market=label, n_oos=nO, qlike_rw=q_rw, qlike_ens=q_ens,
                qlike_impr_pct=(1 - q_ens / q_rw) * 100, r2_rw=r2(fc["RW126"]), r2_ens=r2(fc["ENS"]),
                ens_dES5=boot.get("ENS", (np.nan, np.nan))[0], ens_p=boot.get("ENS", (np.nan, np.nan))[1],
                harx_dES5=boot.get("HARX", (np.nan, np.nan))[0], harx_p=boot.get("HARX", (np.nan, np.nan))[1])

rows = []
# ---- (1) 국제 4지역: p21 재현 (rng 하나, 시장 순서·ENS→HARX 소비 순서 동일) ----
rng_i = np.random.default_rng(SEED)
MK = [("Japan", "Japan_MOM_Factor_Daily.csv", "Japan_3_Factors_Daily.csv"),
      ("Europe", "Europe_MOM_Factor_Daily.csv", "Europe_3_Factors_Daily.csv"),
      ("NorthAmerica", "North_America_MOM_Factor_Daily.csv", "North_America_3_Factors_Daily.csv"),
      ("AsiaPacxJP", "Asia_Pacific_ex_Japan_MOM_Factor_Daily.csv", "Asia_Pacific_ex_Japan_3_Factors_Daily.csv")]
for lab, wf, mf in MK:
    rows.append(run(load_wml(DI / wf), load_mkt(DI / mf), lab, rng_i, ["ENS", "HARX"]))
# ---- (2) 한국 2×3: p22 재현 (rmrf 일별을 시장수익 대용 — 원 스크립트와 동일) ----
kr_wd = pd.read_csv(ROOT / "data/processed/wml_2x3_daily.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
kr_wm = pd.read_csv(ROOT / "data/processed/wml_2x3_monthly.csv", index_col=0).iloc[:, 0]
kr_wm.index = pd.PeriodIndex(kr_wm.index, freq="M")
kr_mkt = pd.read_csv(ROOT / "data/processed/ff_own_daily.csv", parse_dates=["Date"]).set_index("Date")["rmrf"].dropna()
rows.append(run(kr_wd, kr_mkt, "Korea_2x3", np.random.default_rng(SEED), ["ENS", "HARX", "Ridge", "HAR"], wm_override=kr_wm))
# ---- (3) 미국 소형주 ME1: p23 재현 (ME1 먼저 → rng 소비 순서 동일) ----
P25 = load_25(DI / "25_Portfolios_ME_Prior_12_2_Daily.csv")
me1 = (P25["SMALL HiPRIOR"] - P25["SMALL LoPRIOR"]).dropna()
rows.append(run(me1, load_us_mkt(), "US_ME1", np.random.default_rng(SEED), ["ENS", "HARX", "Ridge", "HAR"]))

out = pd.DataFrame(rows).set_index("market").round(4)
print(out.round(3).to_string())

# ---- 게이트: 기존 산출물 재현 ----
ref = pd.read_csv(ROOT / "output/tables/table9_intl_boundary.csv")
for lab, key in [("Japan", "Japan"), ("Europe", "Europe"), ("NorthAmerica", "NorthAmerica"),
                 ("AsiaPacxJP", "AsiaPacxJP"), ("Korea_2x3", "Korea"), ("US_ME1", "US")]:
    row = ref[(ref["series"] == key) & (ref["construction"].isin(["2x3", "smallcap_ME1_quintile"]))]
    if lab == "Korea_2x3": row = ref[(ref["series"] == "Korea") & (ref["construction"] == "2x3")]
    if lab == "US_ME1": row = ref[(ref["series"] == "US") & (ref["construction"] == "smallcap_ME1_quintile")]
    d = abs(out.loc[lab, "ens_dES5"] - row["ens_dES5"].values[0])
    dp_ = abs(out.loc[lab, "ens_p"] - row["ens_p"].values[0])
    print(f"[게이트] {lab}: |ΔdES5|={d:.3f} |Δp|={dp_:.3f}")
    assert d < 0.03 and dp_ < 0.02, f"{lab} 재현 실패"
out.to_csv(ROOT / "output/tables/p29_intl_qlike.csv")
print("\n[GATES] 6행 전부 기존 ΔES5·p 재현 ✓ · p29_intl_qlike.csv 저장 ✓")
