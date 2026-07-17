"""
P16 — 블라인드 심사(major_revision ×3) 계량 대응 배터리.
전부 기존 처리자료(p3_weights·ml_dataset·ff_own_monthly)로 재계산. seed=42 고정, paired 블록부트스트랩.

(1) Romano–Wolf step-down: ΔES5의 모형별 FWER 보정 p — ENS 고유 p를 준다.
    (기존 fw_p=0.002는 max-t 주도가 ExpandingMean이라 ENS 주장에 쓸 수 없음 — 심사1·2 적중)
(2) 개별 p 정본화: 블록12·B=5000·seed42로 ΔES5 개별 p 단일화 (0.034 vs 0.0398 불일치 해소)
(3) ES 신뢰수준 스윕(1/5/10%) + 유효 꼬리 관측 수 병기
(4) CRRA 확실성등가: MV-CE는 분산만 가격화 → CRRA로 꼬리까지 가격화 (표 6 보강, §7.3 '직접 확인' 교정)
(5) 메커니즘 정식화(§5.2): 팩터별 r_{t+1} ~ 위험_t NW회귀 + 분위별 사후수익 (음의 위험-수익은 모멘텀만?)
(6) 하위구간 분해: 전체표본 0.17 vs 표본외 0.565 괴리 분해 + −46.6% 최악월 시점 규명 (프레임 균열 대응)
(7) net-of-cost ES5·CE: 한국 증권거래세 반영 순수익 꼬리·후생 재계산 (헤드라인 net 검증)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path("/mnt/20t/졸업논문")
W=pd.read_csv(ROOT/"data/processed/p3_weights.csv",index_col=0); W.index=pd.PeriodIndex(W.index,freq="M")
ds=pd.read_csv(ROOT/"data/processed/ml_dataset.csv",index_col=0); ds.index=pd.PeriodIndex(ds.index,freq="M")
ff=pd.read_csv(ROOT/"data/processed/ff_own_monthly.csv",parse_dates=["month"]).set_index("month"); ff.index=ff.index.to_period("M")
wml=ds["tgt_wml_next"].reindex(W.index).values
RET={m:W[m].values*wml for m in W.columns}
n=len(W); MODELS=list(W.columns)
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
def es(r,q): r=r[np.isfinite(r)]; k=max(1,int(q*len(r))); return np.mean(np.sort(r)[:k])*100
def bidx():
    return np.concatenate([np.arange(s,s+12)%n for s in rng.integers(0,n,size=int(np.ceil(n/12)))])[:n]

# ================= (1)+(2) Romano–Wolf step-down + 개별 p (공용 부트스트랩) =================
B=5000
cand=[m for m in MODELS if m!="RW126"]
d_obs={m:es(RET[m],0.05)-es(RET["RW126"],0.05) for m in cand}
D=np.zeros((B,len(cand)))
for b in range(B):
    ix=bidx(); base=es(RET["RW126"][ix],0.05)
    D[b]=[es(RET[m][ix],0.05)-base for m in cand]
sd={m:D[:,i].std() for i,m in enumerate(cand)}
indiv_p={m:float((D[:,i]<=0).mean()) for i,m in enumerate(cand)}          # 개별(비보정) p
t_obs={m:d_obs[m]/sd[m] for m in cand}
Dc=(D-np.array([d_obs[m] for m in cand]))/np.array([sd[m] for m in cand]) # 중심화·studentize
# step-down (t_obs 내림차순)
order=sorted(cand,key=lambda m:-t_obs[m]); rw_p={}; remaining=list(range(len(cand))); prev=0.0
idx_of={m:i for i,m in enumerate(cand)}
for m in order:
    cols=remaining
    maxnull=Dc[:,cols].max(1)
    p=float((maxnull>=t_obs[m]).mean()); p=max(prev,p); rw_p[m]=p; prev=p
    remaining.remove(idx_of[m])
print("===== (1) Romano–Wolf step-down (ΔES5, 11개 가족, B=5000) =====")
print(f"  {'모형':>13} {'ΔES5':>7} {'개별p':>7} {'RW보정p':>8}")
for m in order:
    print(f"  {m:>13} {d_obs[m]:+7.3f} {indiv_p[m]:7.3f} {rw_p[m]:8.3f}")
print(f"  → ENS 개별 p={indiv_p['ENS']:.3f} · RW보정 p={rw_p['ENS']:.3f} | HARX 개별 p={indiv_p['HARX']:.3f} · RW보정 p={rw_p['HARX']:.3f}")
print(f"  ※ 최대 ΔES5 주도 모형 = ExpandingMean({d_obs['ExpandingMean']:+.2f}, 사실상 디레버리지) — 기존 fw_p=0.002는 이 귀무의 값")

# ================= (3) ES 신뢰수준 스윕 =================
print("\n===== (3) ES 신뢰수준 스윕 (개별 블록부트스트랩 p) =====")
rows=[]
for q in (0.01,0.05,0.10):
    k=int(q*n); base=es(RET["RW126"],q)
    for m in ["ENS","HARX","Ridge"]:
        do=es(RET[m],q)-base
        bs=np.array([es(RET[m][ix:=bidx()],q)-es(RET["RW126"][ix],q) for _ in range(2000)])
        rows.append(dict(q=f"{int(q*100)}%",n_tail=k,model=m,base=base,dES=do,p=float((bs<=0).mean())))
        print(f"  ES{int(q*100)}%(꼬리 {k}관측) {m:>5}: base={base:+.2f} ΔES={do:+.2f}%p p={rows[-1]['p']:.3f}")
essweep=pd.DataFrame(rows)

# ================= (4) CRRA 확실성등가 (꼬리까지 가격화) =================
print("\n===== (4) CRRA 확실성등가 (연%, 월수익 복리 기반) =====")
def crra_ce_ann(r,g):
    x=1+r[np.isfinite(r)]
    if (x<=0).any(): return np.nan
    ce_m=(np.mean(x**(1-g)))**(1/(1-g))-1 if g!=1 else np.exp(np.mean(np.log(x)))-1
    return ((1+ce_m)**12-1)*100
crra={}
for g in (2,5,10,20):
    row={m:crra_ce_ann(RET[m],g) for m in ["ExpandingMean","RW126","ENS","HARX"]}
    # ExpandingMean은 관리대용 아님 → 무관리는 raw wml
    row["무관리"]=crra_ce_ann(wml,g)
    crra[g]=row
    mgmt=row["RW126"]-row["무관리"]; ml=row["ENS"]-row["RW126"]
    print(f"  γ={g:>2}: 무관리 {row['무관리']:+7.1f} · RW126 {row['RW126']:+7.1f} · ENS {row['ENS']:+7.1f} "
          f"| 관리가치 {mgmt:+.1f} · ML증분 {ml:+.2f}")
crra_tab=pd.DataFrame(crra).T

# ================= (5) 메커니즘: 팩터별 조건부 위험-수익 =================
print("\n===== (5) 조건부 위험–수익: r_{t+1} ~ 위험_t (NW-3), 팩터별 =====")
def rv6(x): return x.rolling(6).var()
facs={"WML":ds["tgt_wml_next"].copy()}   # placeholder; 아래서 정합 구성
# 월별 수익 시계열 구성
mw=pd.read_csv(ROOT/"data/processed/wml_monthly.csv",parse_dates=["month"]).set_index("month"); mw.index=mw.index.to_period("M")
series={"WML":mw["wml"], "시장":ff["rmrf"], "SMB":ff["smb"], "HML":ff["hml"]}
mech_rows={}; quint={}
for nm,s in series.items():
    s=s.dropna(); risk=s.rolling(6).var().shift(1)  # t까지 정보 → t+1 예측
    df=pd.concat([s.rename("r"),risk.rename("rv")],axis=1).dropna()
    X=np.column_stack([np.ones(len(df)),(df["rv"]-df["rv"].mean())/df["rv"].std()])
    y=df["r"].values*100
    beta=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X@beta
    S=np.zeros((2,2))
    for l in range(4):
        wgt=1-l/4
        for t in range(l,len(df)):
            u=resid[t]*X[t]; v=resid[t-l]*X[t-l]; S+=wgt*(np.outer(u,v)+(np.outer(v,u) if l else 0))
    XtXi=np.linalg.inv(X.T@X); V=XtXi@S@XtXi; tstat=beta/np.sqrt(np.diag(V))
    # 분위별 사후수익
    q5=pd.qcut(df["rv"],5,labels=False)
    qm=[df["r"][q5==i].mean()*1200 for i in range(5)]
    mech_rows[nm]=dict(slope=beta[1],t_slope=tstat[1],prem_ann=s.mean()*1200)
    quint[nm]=qm
    print(f"  {nm:>4}: 위험기울기 {beta[1]:+6.2f}%p (t={tstat[1]:+.2f}) · 프리미엄 {s.mean()*1200:+.2f}%/yr · "
          f"분위1→5 사후수익 {qm[0]:+.1f}→{qm[4]:+.1f}")
mech=pd.DataFrame(mech_rows).T
quint_tab=pd.DataFrame(quint,index=[f"Q{i+1}" for i in range(5)]).T

# ================= (6) 하위구간 분해 + 최악월 규명 =================
print("\n===== (6) WML 하위구간 분해 (전체표본 0.17 vs 표본외 0.565) =====")
mwv=mw["wml"].dropna()
segs={"1991–2000":("1991","2000"),"2001–2010":("2001","2010"),"2011–2026":("2011","2026"),
      "전체(1991–2026)":("1991","2026"),"표본외 raw(2001–2026)":("2001-12","2026")}
seg_rows={}
for nm,(a,b) in segs.items():
    x=mwv.loc[a:b]
    seg_rows[nm]=dict(n=len(x),sharpe=sr(x.values),skew=stats.skew(x.values),worst=x.min()*100,worst_month=str(x.idxmin()))
    print(f"  {nm:>20}: n={len(x):>3} 샤프 {sr(x.values):+.2f} 왜도 {stats.skew(x.values):+.2f} 최악 {x.min()*100:+.1f}% ({x.idxmin()})")
# 표본외 무관리(스케일 전 wml) 최악월
raw_oos=pd.Series(wml,index=W.index)
print(f"  [표본외 무관리 최악월] {raw_oos.idxmin()} = {raw_oos.min()*100:.1f}%  (표 5의 −46.6% 사건)")
seg=pd.DataFrame(seg_rows).T

# ================= (7) net-of-cost ES5·CE (증권거래세 반영) =================
print("\n===== (7) net-of-cost: ES5·CRRA-CE (왕복비용=수수료+거래세) =====")
TO=0.28851583211895276    # 다리별 월 회전율
# 순수익: r_net = L*wml - c*(2*TO*L + 2*|ΔL|) ; c=왕복 총비용(거래세 0.23%+수수료 등 가정)
def net_ret(m,c):
    L=W[m].values; dL=np.abs(np.diff(L,prepend=L[0]))
    cost=c*(2*TO*L+2*dL)
    return L*wml-cost
for c,tag in [(0.0030,"30bp"),(0.0050,"50bp(세0.23+α)")]:
    print(f"  [c={tag}]  ", end="")
    for m in ["RW126","ENS","HARX"]:
        rn=net_ret(m,c); print(f"{m}: 샤프 {sr(rn):.3f} ES5 {es(rn,0.05):+.2f} CE(γ10) {crra_ce_ann(rn,10):+.1f}  ", end="")
    print()
# net ΔES5 (ENS,HARX vs RW126) at 30bp with p
c=0.0030; net={m:net_ret(m,c) for m in ["RW126","ENS","HARX"]}
netrows={}
for m in ["ENS","HARX"]:
    do=es(net[m],0.05)-es(net["RW126"],0.05)
    bs=np.array([es(net[m][ix:=bidx()],0.05)-es(net["RW126"][ix],0.05) for _ in range(2000)])
    netrows[m]=dict(dES5_net=do,p=float((bs<=0).mean()),sharpe_net=sr(net[m]))
    print(f"  net(30bp) {m}: ΔES5={do:+.2f}%p p={netrows[m]['p']:.3f}")

# ---- 저장 ----
pd.DataFrame({"model":order,"dES5":[d_obs[m] for m in order],
              "indiv_p":[indiv_p[m] for m in order],"rw_p":[rw_p[m] for m in order]}
             ).to_csv(ROOT/"output/tables/p16_romano_wolf.csv",index=False)
essweep.round(4).to_csv(ROOT/"output/tables/p16_es_sweep.csv",index=False)
crra_tab.round(3).to_csv(ROOT/"output/tables/p16_crra_ce.csv")
mech.round(4).to_csv(ROOT/"output/tables/p16_conditional_risk_return.csv")
quint_tab.round(2).to_csv(ROOT/"output/tables/p16_risk_quintiles.csv")
seg.to_csv(ROOT/"output/tables/p16_subperiods.csv")
pd.DataFrame(netrows).T.round(4).to_csv(ROOT/"output/tables/p16_net_of_cost.csv")

# ---- 게이트 ----
assert 0.02<=indiv_p["ENS"]<=0.06, f"ENS 개별p 예상밖 {indiv_p['ENS']}"
assert rw_p["ENS"]>=indiv_p["ENS"], "RW보정p가 개별p보다 작을 수 없음"
assert mech.loc["WML","slope"]<0 and mech.loc["WML","t_slope"]<mech.loc["시장","t_slope"], "WML 음의 위험-수익 미확인"
print("\n[GATES] ENS 개별p 범위 ✓ · RW≥개별 ✓ · WML 음의 위험-수익 ✓ · 표 7종 저장 ✓")
