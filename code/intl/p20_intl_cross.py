"""
다시장 횡단면: "팩터 강도 → ML 꼬리가치" 법칙 검증.
5개 시장(일·유럽·북미·아태·미) KF 모멘텀 팩터(2x3, 일별) + 한국(decile) 보조점.
각 시장 동일 파이프라인(WML 자체 변동성 피처만 — 시장데이터 불필요, 비교가능성↑).
가설: 약한 팩터일수록 ML의 ΔES5(꼬리 개선)가 크다 → 횡단면 기울기 음(−).
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
D="/mnt/20t/tmp/claude-1001/-mnt-20t-----/da2b9872-77c2-46d2-a247-cd601338a0be/scratchpad/intl/"
SEED=42; rng=np.random.default_rng(SEED)

def load_wml(fp):
    L=open(fp,encoding="latin1").read().splitlines()
    i0=next(i for i,l in enumerate(L) if l.strip().startswith(",") and ("WML" in l or "Mom" in l))
    r=[]
    for l in L[i0+1:]:
        t=l.split(","); k=t[0].strip()
        if k.isdigit() and len(k)==8:
            try: r.append([k,float(t[1])])
            except: break
        elif r: break
    s=pd.Series(dict(r)); s.index=pd.to_datetime(s.index,format="%Y%m%d")
    return s.replace([-99.99,-999.0],np.nan).dropna()/100.0

sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
def es5(r): r=r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1,int(0.05*len(r)))])*100

def run_market(wd, label):
    wm=(1+wd).groupby(wd.index.to_period("M")).prod()-1
    g=wd.groupby(wd.index.to_period("M"))
    RV=g.apply(lambda x:(x**2).sum()); SEMI=g.apply(lambda x:(x[x<0]**2).sum())
    RW=(21*(wd**2).rolling(126,min_periods=126).mean()); RWme=RW.groupby(RW.index.to_period("M")).last()
    df=pd.DataFrame(index=RV.index)
    df["rv"]=RV; df["l1"]=np.log(RV); df["l3"]=np.log(RV.rolling(3).mean())
    df["l6"]=np.log(RV.rolling(6).mean()); df["l12"]=np.log(RV.rolling(12).mean())
    df["semir"]=(SEMI/RV).clip(0,1); df["rw"]=RWme
    df["tgt"]=np.log(RV.shift(-1)); df["rvn"]=RV.shift(-1); df["wmln"]=wm.reindex(df.index).shift(-1)
    df=df.dropna(subset=["l12","tgt","wmln","rw"]).copy()
    n=len(df); INIT=120;
    if n<INIT+60: return None
    FEAT=["l1","l3","l6","l12","semir"]; HARF=["l1","l3","l12"]
    X=df[FEAT].values; y=df["tgt"].values; rvcol=df["rv"].values
    def fp(cols,model):
        idx=[FEAT.index(c) for c in cols]
        pred=np.full(n,np.nan)
        for s in range(INIT,n,12):
            lo,hi=np.log(rvcol[:s].min()*0.5),np.log(rvcol[:s].max()*2)
            model.fit(X[:s][:,idx],y[:s]); te=slice(s,min(s+12,n))
            pred[te]=np.exp(np.clip(model.predict(X[te][:,idx]),lo,hi))
        return pred
    oos=np.arange(INIT,n); rvn=df["rvn"].values; wn=df["wmln"].values
    fc={}
    fc["ExpandingMean"]=np.array([rvcol[:t].mean() if t>0 else np.nan for t in range(n)])
    fc["RW126"]=df["rw"].values
    ew=np.full(n,np.nan); ew[0]=rvcol[0]
    for t in range(1,n): ew[t]=0.94*ew[t-1]+0.06*rvcol[t-1]
    fc["EWMA"]=ew
    fc["HAR"]=fp(HARF,LinearRegression())
    fc["Ridge"]=fp(FEAT,RidgeCV(alphas=np.logspace(-3,3,20)))
    fc["HGB"]=fp(FEAT,HistGradientBoostingRegressor(max_depth=3,max_iter=150,random_state=SEED))
    fc["ENS"]=np.nanmean([fc["HAR"],fc["Ridge"],fc["HGB"]],axis=0)
    TGT=0.12/np.sqrt(12); RET={m:(TGT/np.sqrt(f)*wn)[oos] for m,f in fc.items()}
    raw=wn[oos]
    em=fc["ExpandingMean"]
    def r2(f): m=oos; return (1-np.sum((rvn[m]-f[m])**2)/np.sum((rvn[m]-em[m])**2))*100
    # ENS ΔES5 vs RW126 + 블록부트스트랩
    nO=len(oos); B=3000
    def bidx(): return np.concatenate([np.arange(s,s+12)%nO for s in rng.integers(0,nO,size=int(np.ceil(nO/12)))])[:nO]
    base=es5(RET["RW126"]); dES=es5(RET["ENS"])-base
    bs=np.array([es5(RET["ENS"][ix:=bidx()])-es5(RET["RW126"][ix]) for _ in range(B)])
    p=float((bs<=0).mean())
    # 관리(RW126−raw) ΔES5
    dmgmt=es5(RET["RW126"])-es5(raw)
    return dict(market=label, n_oos=nO, raw_sharpe=sr(raw), rw_sharpe=sr(RET["RW126"]),
                ens_sharpe=sr(RET["ENS"]), ens_R2=r2(fc["ENS"]),
                mgmt_dES5=dmgmt, ml_dES5=dES, ml_dES5_p=p)

MK=[("Japan",D+"Japan_MOM_Factor_Daily.csv"),("Europe",D+"Europe_MOM_Factor_Daily.csv"),
    ("NorthAmerica",D+"North_America_MOM_Factor_Daily.csv"),
    ("AsiaPacxJP",D+"Asia_Pacific_ex_Japan_MOM_Factor_Daily.csv"),
    ("US",D+"F-F_Momentum_Factor_daily.csv")]
rows=[]
for lab,fp_ in MK:
    r=run_market(load_wml(fp_),lab)
    if r: rows.append(r); print(f"  {lab:>13}: rawSh {r['raw_sharpe']:+.2f} | 관리ΔES5 {r['mgmt_dES5']:+.1f} | ML ΔES5 {r['ml_dES5']:+.2f} (p={r['ml_dES5_p']:.3f}) | ENS R² {r['ens_R2']:.0f}%")

# 한국 보조점 (프로젝트 decile)
try:
    W=pd.read_csv("/mnt/20t/졸업논문/data/processed/p3_weights.csv",index_col=0)
    ds=pd.read_csv("/mnt/20t/졸업논문/data/processed/ml_dataset.csv",index_col=0)
    wmlk=ds["tgt_wml_next"].reindex(W.index).values
    RETk={m:(W[m].values*wmlk) for m in ["RW126","ENS"]}
    rawk=wmlk[np.isfinite(wmlk)]
    kr=dict(market="Korea(decile)",n_oos=len(rawk),raw_sharpe=sr(rawk),rw_sharpe=np.nan,ens_sharpe=np.nan,
            ens_R2=np.nan,mgmt_dES5=np.nan,ml_dES5=es5(RETk["ENS"][np.isfinite(RETk["ENS"])])-es5(RETk["RW126"][np.isfinite(RETk["RW126"])]),ml_dES5_p=0.037)
    rows.append(kr); print(f"  {'Korea(decile)':>13}: rawSh {kr['raw_sharpe']:+.2f} | ML ΔES5 {kr['ml_dES5']:+.2f} (p≈0.037)")
except Exception as e: print("Korea 보조점 skip:",e)

df=pd.DataFrame(rows)
# ★횡단면 회귀: ML ΔES5 ~ raw_sharpe (KF 5개 시장, 동일구성)
kf=df[df.market!="Korea(decile)"]
lr=stats.linregress(kf["raw_sharpe"],kf["ml_dES5"])
print(f"\n===== ★횡단면: ML ΔES5 = {lr.intercept:+.2f} + ({lr.slope:+.2f})×raw샤프 =====")
print(f"  기울기 {lr.slope:+.3f} (r={lr.rvalue:+.2f}, p={lr.pvalue:.3f}, n={len(kf)}) — 음이면 '약할수록 ML 꼬리가치 큼' 확증")
print(f"  스피어만 순위상관 ρ={stats.spearmanr(kf['raw_sharpe'],kf['ml_dES5']).statistic:+.2f}")
df.round(4).to_csv(D+"intl_cross.csv",index=False)
print("\n[SAVE] intl_cross.csv")
