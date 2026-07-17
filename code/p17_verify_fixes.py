"""
P17 — 개정고 재심(minor_revision) 잔여 2건의 실질 보강.

(A) 메커니즘 추론력: 단일 팩터 위험-수익 기울기 t=−0.90은 약함 →
    WML 기울기 − 타팩터 평균 기울기의 paired 블록부트스트랩 CI (교차팩터 대비의 검정력).
    + WML의 Q5−Q1 스프레드가 타팩터보다 유의하게 음인가.
(B) 공매도 대차비용: 헤드라인이 숏다리 크래시 보험이므로 대차수수료(숏 명목의 연율 캐리)를
    거래비용과 별도로 부과 → 순ES5·순CRRA(γ10) 증분과 대차 손익분기(%/yr).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path("/mnt/20t/졸업논문")
W=pd.read_csv(ROOT/"data/processed/p3_weights.csv",index_col=0); W.index=pd.PeriodIndex(W.index,freq="M")
ds=pd.read_csv(ROOT/"data/processed/ml_dataset.csv",index_col=0); ds.index=pd.PeriodIndex(ds.index,freq="M")
mw=pd.read_csv(ROOT/"data/processed/wml_monthly.csv",parse_dates=["month"]).set_index("month"); mw.index=mw.index.to_period("M")
ff=pd.read_csv(ROOT/"data/processed/ff_own_monthly.csv",parse_dates=["month"]).set_index("month"); ff.index=ff.index.to_period("M")
wml=ds["tgt_wml_next"].reindex(W.index).values
RET={m:W[m].values*wml for m in W.columns}
n=len(W)
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
def es(r,q=0.05): r=r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1,int(q*len(r)))])*100
def crra_ce_ann(r,g=10):
    x=1+r[np.isfinite(r)]
    if (x<=0).any(): return np.nan
    ce=(np.mean(x**(1-g)))**(1/(1-g))-1
    return ((1+ce)**12-1)*100
def bidx(m):
    return np.concatenate([np.arange(s,s+12)%m for s in rng.integers(0,m,size=int(np.ceil(m/12)))])[:m]

# ================= (A) 메커니즘 교차팩터 추론 =================
print("===== (A) 조건부 위험–수익: WML vs 타팩터 기울기 차 (paired 블록부트스트랩) =====")
series={"WML":mw["wml"],"시장":ff["rmrf"],"SMB":ff["smb"],"HML":ff["hml"]}
# 공통 표본에서 표준화 위험(직전 6개월 실현분산)·사후수익
frames={}
for nm,s in series.items():
    s=s.dropna(); risk=s.rolling(6).var().shift(1)
    df=pd.concat([s.rename("r"),risk.rename("rv")],axis=1).dropna()
    frames[nm]=df
common=sorted(set.intersection(*[set(f.index) for f in frames.values()]))
common=pd.PeriodIndex(common)
def slope_on(nm,idx):
    df=frames[nm].loc[common][["r","rv"]].iloc[idx]
    z=(df["rv"]-df["rv"].mean())/df["rv"].std()
    b=np.polyfit(z,df["r"].values*100,1)[0]; return b
m=len(common)
obs={nm:slope_on(nm,np.arange(m)) for nm in series}
diff_obs=obs["WML"]-np.mean([obs[k] for k in ["시장","SMB","HML"]])
B=5000; diffs=np.empty(B); wsl=np.empty(B)
for b in range(B):
    ix=bidx(m)
    sl={nm:slope_on(nm,ix) for nm in series}
    wsl[b]=sl["WML"]; diffs[b]=sl["WML"]-np.mean([sl[k] for k in ["시장","SMB","HML"]])
lo,hi=np.percentile(diffs,[2.5,97.5]); wlo,whi=np.percentile(wsl,[2.5,97.5])
print(f"  WML 기울기 {obs['WML']:+.2f}%p (부트 95% CI [{wlo:+.2f},{whi:+.2f}])")
print(f"  WML − 평균(타3팩터) = {diff_obs:+.2f}%p · 95% CI [{lo:+.2f},{hi:+.2f}] · P(차≥0)={float((diffs>=0).mean()):.3f}")
print(f"  → WML이 유의하게 더 음(−): {'예' if hi<0 else '아니오(방향적)'}")

# ================= (B) 공매도 대차비용 =================
print("\n===== (B) 대차수수료(숏 명목 연율) 부과 후 순ES5·순CRRA(γ10) =====")
TO=0.28851583211895276; c_trade=0.0030   # 왕복 거래비용 30bp 고정
def net_with_borrow(mo, f_ann):
    L=W[mo].values; dL=np.abs(np.diff(L,prepend=L[0]))
    trade=c_trade*(2*TO*L+2*dL)
    borrow=(f_ann/12.0)*L          # 숏 명목 ≈ L (self-financing)
    return L*wml-trade-borrow
rows=[]
for f in (0.0,0.010,0.020,0.030):
    d={"f_bp_yr":int(f*1e4)}
    for mo in ["RW126","ENS","HARX"]:
        r=net_with_borrow(mo,f); d[f"{mo}_ES5"]=es(r); d[f"{mo}_CE10"]=crra_ce_ann(r,10)
    d["ENS_dES5"]=es(net_with_borrow("ENS",f))-es(net_with_borrow("RW126",f))
    d["ENS_dCE10"]=crra_ce_ann(net_with_borrow("ENS",f),10)-crra_ce_ann(net_with_borrow("RW126",f),10)
    d["HARX_dES5"]=es(net_with_borrow("HARX",f))-es(net_with_borrow("RW126",f))
    rows.append(d)
    print(f"  대차 {int(f*1e4):>3}bp/yr: ENS ΔES5={d['ENS_dES5']:+.2f}%p ΔCE10={d['ENS_dCE10']:+.2f}%p · HARX ΔES5={d['HARX_dES5']:+.2f}%p")
bt=pd.DataFrame(rows)
# 대차 손익분기: ENS ΔES5 개선이 0이 되는 f (숏비용이 꼬리이득을 상쇄)
fs=np.linspace(0,0.10,201); de=[es(net_with_borrow("ENS",f))-es(net_with_borrow("RW126",f)) for f in fs]
de=np.array(de); be=next((fs[i] for i in range(len(fs)) if de[i]<=0), np.inf)
print(f"  ENS 꼬리(ΔES5) 대차 손익분기 ≈ {be*1e4:.0f}bp/yr" if np.isfinite(be) else "  ENS ΔES5는 대차 10%/yr까지도 양(+)")
# 유의성: 대차 200bp에서 ENS ΔES5 p
f=0.020; rn={mo:net_with_borrow(mo,f) for mo in ["RW126","ENS","HARX"]}
p_ens=float((np.array([es(rn["ENS"][ix:=bidx(n)])-es(rn["RW126"][ix]) for _ in range(2000)])<=0).mean())
print(f"  [대차 200bp] ENS ΔES5={es(rn['ENS'])-es(rn['RW126']):+.2f}%p p={p_ens:.3f}")

pd.DataFrame({"metric":["diff_obs","diff_lo","diff_hi","P_diff_ge0","borrow_be_bp"],
             "value":[diff_obs,lo,hi,float((diffs>=0).mean()),be*1e4 if np.isfinite(be) else -1]}
            ).to_csv(ROOT/"output/tables/p17_mechanism_test.csv",index=False)
bt.round(4).to_csv(ROOT/"output/tables/p17_borrow_cost.csv",index=False)

assert obs["WML"]<0, "WML 기울기 음 아님"
print("\n[GATES] WML 기울기 음 ✓ · 표 2종 저장 ✓")
