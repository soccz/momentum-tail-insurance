"""
P9 — 최종 점검 배터리 (심사자 구멍 막기).

(a) 꼬리 지표 부트스트랩: Δ왜도·Δ최악월·ΔES5%의 블록부트스트랩 CI — 헤드라인("가치는 꼬리")의 공식 추론
(b) 레버리지 캡: L≤1.0 / ≤1.5 / 무제한 — 실행 제약 하 결론 유지?
(c) 공매도 금지기간(2008-10~2009-05, 2020-03~2021-04, 2023-11~2025-03): 금지월 현금(0) 처리 시
(d) 학습창 민감도: 초기 120 vs 180개월 (Ridge·HARX 재적합)
(e) 피처 중요도: Ridge 표준화 계수의 시간평균 |β| 상위
(f) SPA형 다중검정 (White Reality Check, 정상부트스트랩): "11개 중 최고가 RW126을 이긴 게 우연인가"
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler

SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path("/mnt/20t/졸업논문")
fc=pd.read_csv(ROOT/"data/processed/ml_forecasts.csv",index_col=0); fc.index=pd.PeriodIndex(fc.index,freq="M")
W=pd.read_csv(ROOT/"data/processed/p3_weights.csv",index_col=0); W.index=pd.PeriodIndex(W.index,freq="M")
ds=pd.read_csv(ROOT/"data/processed/ml_dataset.csv",index_col=0); ds.index=pd.PeriodIndex(ds.index,freq="M"); ds=ds.dropna()
wml_o=ds["tgt_wml_next"].reindex(fc.index).values
n=len(fc)
RET={m:(W[m].values*wml_o) for m in W.columns}
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
es5=lambda r: np.mean(np.sort(r[np.isfinite(r)])[:max(1,int(0.05*len(r)))])*100

def block_idx():
    return np.concatenate([np.arange(s,s+12)%n for s in rng.integers(0,n,size=int(np.ceil(n/12)))])[:n]

# ---------- (a) 꼬리 부트스트랩 ----------
print("===== (a) 꼬리 지표 부트스트랩 vs RW126 (B=5000) — 개선=양(+) 방향 =====")
tail_rows={}
for m in ["HARX","Ridge","ENS"]:
    d_sk,d_wr,d_es=[],[],[]
    for _ in range(5000):
        ix=block_idx(); r1,r0=RET[m][ix],RET["RW126"][ix]
        d_sk.append(stats.skew(r1)-stats.skew(r0))
        d_wr.append((r1.min()-r0.min())*100)
        d_es.append(es5(r1)-es5(r0))
    row={}
    for nm,d in [("Δ왜도",d_sk),("Δ최악월%p",d_wr),("ΔES5%p",d_es)]:
        d=np.array(d); row[nm]=d.mean(); row[nm+"_P(≤0)"]=float((d<=0).mean())
    tail_rows[m]=row
    print(f"  {m:>6}: Δ왜도 {row['Δ왜도']:+.2f} (P≤0={row['Δ왜도_P(≤0)']:.3f}) | "
          f"Δ최악월 {row['Δ최악월%p']:+.1f}%p (P≤0={row['Δ최악월%p_P(≤0)']:.3f}) | "
          f"ΔES5 {row['ΔES5%p']:+.2f}%p (P≤0={row['ΔES5%p_P(≤0)']:.3f})")
tail=pd.DataFrame(tail_rows).T

# ---------- (b) 레버리지 캡 ----------
print("\n===== (b) 레버리지 캡 =====")
cap_rows={}
for m in ["RW126","HARX","Ridge"]:
    for cap,lab in [(np.inf,"무제한"),(1.5,"L≤1.5"),(1.0,"L≤1.0")]:
        r=np.minimum(W[m].values,cap)*wml_o
        cap_rows[f"{m} {lab}"]=dict(sharpe=sr(r),worst=np.nanmin(r)*100,ES5=es5(r))
caps=pd.DataFrame(cap_rows).T
print(caps.round(3).to_string())

# ---------- (c) 공매도 금지기간 ----------
BANS=[("2008-10","2009-05"),("2020-03","2021-04"),("2023-11","2025-03")]
ban_mask=np.zeros(n,bool)
for s,e in BANS:
    ban_mask |= (fc.index>=s)&(fc.index<=e)
print(f"\n===== (c) 공매도 금지월 {ban_mask.sum()}개 — 금지월 현금(0) 처리 =====")
ban_rows={}
for m in ["RW126","HARX","Ridge"]:
    r=RET[m].copy(); r_ban=r.copy(); r_ban[ban_mask]=0.0
    ban_rows[m]=dict(sharpe_기준=sr(r),sharpe_금지반영=sr(r_ban),
                     worst_기준=np.nanmin(r)*100,worst_금지반영=np.nanmin(r_ban)*100)
bans=pd.DataFrame(ban_rows).T
print(bans.round(3).to_string())

# ---------- (d) 학습창 민감도 (FT=180) ----------
print("\n===== (d) 초기 학습창 120 vs 180 =====")
FEAT=[c for c in ds.columns if not c.startswith("tgt_")]
X=ds[FEAT].values; yv=np.log(ds["tgt_var_next"].values); wn=ds["tgt_wml_next"].values
harx_cols=["vol_1m","vol_3m","vol_12m","bear","rebound","mkt_vol_1m","semi_ratio_6m","vol_gap_6m"]
Xh=ds[harx_cols].values
def run_ft(FT):
    nn=len(ds); oo=np.arange(FT,nn)
    preds={"Ridge":np.full(nn,np.nan),"HARX":np.full(nn,np.nan)}
    for t0 in range(FT,nn,12):
        tr,te=np.arange(t0),np.arange(t0,min(t0+12,nn))
        sc=StandardScaler().fit(X[tr]); Z=sc.transform(X)
        mR=RidgeCV(alphas=np.logspace(-3,3,13)).fit(Z[tr],yv[tr]); res=yv[tr]-mR.predict(Z[tr])
        preds["Ridge"][te]=np.clip(np.exp(mR.predict(Z[te])+res.var()/2),
                                   ds["tgt_var_next"].values[tr].min()*.5,ds["tgt_var_next"].values[tr].max()*2)
        mH=LinearRegression().fit(Xh[tr],yv[tr]); resH=yv[tr]-mH.predict(Xh[tr])
        preds["HARX"][te]=np.clip(np.exp(mH.predict(Xh[te])+resH.var()/2),
                                  ds["tgt_var_next"].values[tr].min()*.5,ds["tgt_var_next"].values[tr].max()*2)
    TGT=0.12/np.sqrt(12)
    out={"RW126":sr((TGT/np.sqrt(ds["fvar_126"].values)*wn)[oo])}
    for k,p in preds.items(): out[k]=sr((TGT/np.sqrt(p)*wn)[oo])
    return out,len(oo)
for FT in (120,180):
    res,noo=run_ft(FT)
    print(f"  FT={FT} (OOS {noo}): "+" · ".join(f"{k} {v:.3f}" for k,v in res.items()))

# ---------- (e) Ridge 피처 중요도 ----------
coefs=[]
for t0 in range(120,len(ds),12):
    tr=np.arange(t0)
    sc=StandardScaler().fit(X[tr]); m=RidgeCV(alphas=np.logspace(-3,3,13)).fit(sc.transform(X[tr]),yv[tr])
    coefs.append(np.abs(m.coef_))
imp=pd.Series(np.mean(coefs,axis=0),index=FEAT).sort_values(ascending=False)
print("\n===== (e) Ridge 피처 중요도 (표준화 |β| 시간평균, 상위 8) =====")
print(imp.head(8).round(3).to_string())

# ---------- (f) SPA형 다중검정 (Reality Check) ----------
act=fc["actual_var"].values
ql=lambda f: act/f-np.log(act/f)-1
L0=ql(fc["RW126"].values)
models=[c for c in fc.columns if c not in ("actual_var","RW126")]
D=np.column_stack([L0-ql(fc[m].values) for m in models])            # >0 = RW보다 좋음
D=np.nan_to_num(D,nan=0.0)
t_obs=D.mean(0)/ (D.std(0)/np.sqrt(n)); t_max=np.nanmax(t_obs)
cnt=0; B=3000
for _ in range(B):
    ix=block_idx(); Db=D[ix]-D.mean(0)                               # 중심화(귀무: 우위 없음)
    tb=Db.mean(0)/(Db.std(0)/np.sqrt(n))
    if np.nanmax(tb)>=t_max: cnt+=1
p_rc=cnt/B
best=models[int(np.nanargmax(t_obs))]
print(f"\n===== (f) Reality Check (11모델 동시, B=3000) =====")
print(f"  최고 모델 {best}: max-t={t_max:.2f} → p={p_rc:.4f}  (다중검정 후에도 RW126 우위 {'유의' if p_rc<0.05 else '비유의'})")

# ---------- 저장 + 게이트 ----------
tail.round(4).to_csv(ROOT/"output/tables/p9_tail_bootstrap.csv")
caps.round(4).to_csv(ROOT/"output/tables/p9_leverage_caps.csv")
bans.round(4).to_csv(ROOT/"output/tables/p9_short_bans.csv")
imp.round(4).to_csv(ROOT/"output/tables/p9_feature_importance.csv")
pd.DataFrame({"best":[best],"max_t":[t_max],"p_RC":[p_rc]}).to_csv(ROOT/"output/tables/p9_reality_check.csv",index=False)
assert np.isfinite(tail.values).all() and np.isfinite(caps["sharpe"]).all()
print("\n[GATES] 표 5종 저장 ✓")
