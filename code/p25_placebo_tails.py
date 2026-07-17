"""
P25 — 교차팩터 플라시보의 꼬리 차원(ΔES5) 보강.

계기: 블라인드 심사 게이트(E33)에서 "표 7이 Δ샤프만 보고하는데 헤드라인 차원은 꼬리"라는
major 지적이 적대적 검증을 통과 — 같은 파이프라인(p4_mechanism.py의 (2) 플라시보 블록)에서
ES5와 대응 블록 부트스트랩 p만 추가 산출한다.

- 포트폴리오 구성·예측 루프는 p4와 문자 그대로 동일(피처·창·시드·로그정규 보정).
  게이트: raw/RW/Ridge 샤프 12셀이 기존 p4_placebo.csv와 ±0.005 안에서 재현되어야 한다.
- ES5 = 표본외 월수익 하위 5% 평균(%) — p16/p21/p22와 동일 규약.
- ΔES5 관리 = ES5(RW) − ES5(raw), ΔES5 ML = ES5(Ridge) − ES5(RW). 양수 = 꼬리 얕아짐.
- p: 대응 원형 블록 부트스트랩(블록 12개월, B=5,000, seed 42), P(Δ* ≤ 0) 단측.
- 원장 E32. 논문 반영은 표 7 열 추가(사용자 승인 완료).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path("/mnt/20t/졸업논문")
P = lambda idx: idx.to_period("M")
FIRST_TRAIN, RETRAIN = 120, 12
TGT_M = 0.12 / np.sqrt(12)
B = 5000
rng = np.random.default_rng(42)

wd = pd.read_csv(ROOT / "data/processed/wml_daily.csv", parse_dates=["Date"]).set_index("Date")
ffd = pd.read_csv(ROOT / "data/processed/ff_own_daily.csv", parse_dates=["Date"]).set_index("Date")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0)
ds.index = pd.PeriodIndex(ds.index, freq="M")
feats_common = ds[["vol_1m", "vol_6m", "mkt_vol_1m", "bear", "rebound", "semi_ratio_6m"]]

f126 = lambda s: (21 * (s ** 2).rolling(126).mean()).groupby(P(s.index)).last()
es5 = lambda r: np.mean(np.sort(r[np.isfinite(r)])[:max(1, int(0.05 * len(r[np.isfinite(r)])))]) * 100

ref = pd.read_csv(ROOT / "output/tables/p4_placebo.csv", index_col=0)

rows = {}
for fac in ["wml", "rmrf", "smb", "hml"]:
    r_d = wd["wml"] if fac == "wml" else ffd[fac]
    var_next = (r_d ** 2).groupby(P(r_d.index)).sum().shift(-1)
    ret_next = (ds["tgt_wml_next"] if fac == "wml"
                else ((1 + r_d).groupby(P(r_d.index)).prod() - 1).shift(-1))
    own = pd.DataFrame({"proxy126": f126(r_d),
                        "vol_1m": np.sqrt(252/21*(r_d**2).rolling(21).sum()).groupby(P(r_d.index)).last()*100})
    fx = pd.concat([own, feats_common[["mkt_vol_1m", "bear", "rebound"]]], axis=1)
    df = pd.concat([var_next.rename("y"), fx, ret_next.rename("ret")], axis=1).dropna()
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
    R = {"raw": ret[oos]}
    for k in ("RW", "Ridge"):
        R[k] = TGT_M / np.sqrt(pred[k][oos]) * ret[oos]
    sh = {k: np.sqrt(12) * np.nanmean(v) / np.nanstd(v) for k, v in R.items()}

    # 게이트: p4_placebo.csv 샤프 재현
    for col, k in [("sharpe_raw", "raw"), ("sharpe_RW", "RW"), ("sharpe_Ridge", "Ridge")]:
        assert abs(sh[k] - ref.loc[fac.upper(), col]) < 0.005, f"{fac} {k} 샤프 불일치: {sh[k]:.4f} vs {ref.loc[fac.upper(), col]:.4f}"

    nO = len(oos)
    def bidx():
        return np.concatenate([np.arange(s, s + 12) % nO
                               for s in rng.integers(0, nO, size=int(np.ceil(nO / 12)))])[:nO]
    d_mgmt = es5(R["RW"]) - es5(R["raw"])
    d_ml = es5(R["Ridge"]) - es5(R["RW"])
    bs = np.array([[es5(R["RW"][ix]) - es5(R["raw"][ix]),
                    es5(R["Ridge"][ix]) - es5(R["RW"][ix])] for _ in range(B) if (ix := bidx()) is not None])
    rows[fac.upper()] = dict(n_oos=nO,
                             ES5_raw=es5(R["raw"]), ES5_RW=es5(R["RW"]), ES5_Ridge=es5(R["Ridge"]),
                             dES5_mgmt=d_mgmt, p_mgmt=float((bs[:, 0] <= 0).mean()),
                             dES5_ML=d_ml, p_ML=float((bs[:, 1] <= 0).mean()))
    print(f"{fac.upper():>5}: ES5 raw {es5(R['raw']):+.2f} → RW {es5(R['RW']):+.2f} → Ridge {es5(R['Ridge']):+.2f} | "
          f"관리 {d_mgmt:+.2f}%p(p={rows[fac.upper()]['p_mgmt']:.3f}) · ML {d_ml:+.2f}%p(p={rows[fac.upper()]['p_ML']:.3f})")

out = pd.DataFrame(rows).T
out.round(4).to_csv(ROOT / "output/tables/p25_placebo_tails.csv")
print("\n[GATES] p4 샤프 12셀 재현 ✓ (±0.005) · p25_placebo_tails.csv 저장 ✓")
