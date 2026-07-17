"""
공정 다시장 횡단면: E20 진단(피처셋 아티팩트) 반영 → 시장상태 피처 포함 재검정.
각 시장: WML 자체변동성 + 그 시장의 시장상태(bear·시장변동성·드로다운·시장수익) 피처.
Korea 승자 HARX와 동일 구성. 질문: full 피처면 국제 시장에서도 약할수록 ML 꼬리↑?
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
D="/mnt/20t/tmp/claude-1001/-mnt-20t-----/da2b9872-77c2-46d2-a247-cd601338a0be/scratchpad/intl/"
SEED=42; rng=np.random.default_rng(SEED)

def _rows(fp, ncol):
    L=open(fp,encoding="latin1").read().splitlines()
    i0=next(i for i,l in enumerate(L) if l.strip().startswith(",") and any(k in l for k in ("WML","Mom","Mkt-RF")))
    r=[]
    for l in L[i0+1:]:
        t=l.split(","); k=t[0].strip()
        if k.isdigit() and len(k)==8:
            try: r.append([k]+[float(x) for x in t[1:1+ncol]]);
            except: break
        elif r: break
    return r
def load_wml(fp):
    r=_rows(fp,1); s=pd.Series({k:v[0] for k,v in ((x[0],x[1:]) for x in r)})
    s.index=pd.to_datetime(s.index,format="%Y%m%d"); return s.replace([-99.99,-999.0],np.nan).dropna()/100.0
def load_mkt(fp):   # Mkt-RF,SMB,HML,RF → 시장수익 = Mkt-RF+RF
    r=_rows(fp,4); df=pd.DataFrame([x[1:] for x in r],index=[x[0] for x in r],columns=["mktrf","smb","hml","rf"])
    df.index=pd.to_datetime(df.index,format="%Y%m%d"); df=df.replace([-99.99,-999.0],np.nan).dropna()/100.0
    return (df["mktrf"]+df["rf"])

sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
def es5(r): r=r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1,int(0.05*len(r)))])*100

def run(wd, mkt_d, label):
    wm=(1+wd).groupby(wd.index.to_period("M")).prod()-1
    g=wd.groupby(wd.index.to_period("M")); RV=g.apply(lambda x:(x**2).sum()); SEMI=g.apply(lambda x:(x[x<0]**2).sum())
    RW=(21*(wd**2).rolling(126,min_periods=126).mean()); RWme=RW.groupby(RW.index.to_period("M")).last()
    mm=(1+mkt_d).groupby(mkt_d.index.to_period("M")).prod()-1   # 월 시장수익
    df=pd.DataFrame(index=RV.index)
    df["rv"]=RV; df["l1"]=np.log(RV); df["l3"]=np.log(RV.rolling(3).mean())
    df["l6"]=np.log(RV.rolling(6).mean()); df["l12"]=np.log(RV.rolling(12).mean()); df["semir"]=(SEMI/RV).clip(0,1)
    df["mret1"]=mm.reindex(df.index); df["mret12"]=mm.reindex(df.index).rolling(12).sum()
    df["mvol6"]=mm.reindex(df.index).rolling(6).std()
    cm=(1+mm).cumprod(); df["mdd"]=(cm/cm.cummax()-1).reindex(df.index)
    df["bear"]=(mm.reindex(df.index).rolling(24).sum()<0).astype(float)
    df["rw"]=RWme; df["tgt"]=np.log(RV.shift(-1)); df["rvn"]=RV.shift(-1); df["wmln"]=wm.reindex(df.index).shift(-1)
    FEAT=["l1","l3","l6","l12","semir","mret1","mret12","mvol6","mdd","bear"]
    HARF=["l1","l3","l12"]; HARXF=HARF+["bear","mvol6","mdd"]
    df=df.dropna(subset=FEAT+["tgt","wmln","rw"]).copy(); n=len(df)
    if n<180: return None
    X=df[FEAT].values; y=df["tgt"].values; rvcol=df["rv"].values; INIT=120
    def fp(cols,model):
        idx=[FEAT.index(c) for c in cols]; pred=np.full(n,np.nan)
        for s in range(INIT,n,12):
            lo,hi=np.log(rvcol[:s].min()*0.5),np.log(rvcol[:s].max()*2)
            model.fit(X[:s][:,idx],y[:s]); te=slice(s,min(s+12,n))
            pred[te]=np.exp(np.clip(model.predict(X[te][:,idx]),lo,hi))
        return pred
    oos=np.arange(INIT,n); rvn=df["rvn"].values; wn=df["wmln"].values
    fc={"RW126":df["rw"].values}
    fc["ExpandingMean"]=np.array([rvcol[:t].mean() if t>0 else np.nan for t in range(n)])
    fc["HAR"]=fp(HARF,LinearRegression()); fc["HARX"]=fp(HARXF,LinearRegression())
    fc["Ridge"]=fp(FEAT,RidgeCV(alphas=np.logspace(-3,3,20)))
    fc["HGB"]=fp(FEAT,HistGradientBoostingRegressor(max_depth=3,max_iter=150,random_state=SEED))
    fc["ENS"]=np.nanmean([fc["HAR"],fc["Ridge"],fc["HGB"]],axis=0)
    TGT=0.12/np.sqrt(12); RET={m:(TGT/np.sqrt(f)*wn)[oos] for m,f in fc.items()}; raw=wn[oos]
    nO=len(oos); B=3000
    def bidx(): return np.concatenate([np.arange(s,s+12)%nO for s in rng.integers(0,nO,size=int(np.ceil(nO/12)))])[:nO]
    def dp(m):
        do=es5(RET[m])-es5(RET["RW126"]); bs=np.array([es5(RET[m][ix:=bidx()])-es5(RET["RW126"][ix]) for _ in range(B)])
        return do, float((bs<=0).mean())
    de_ens,p_ens=dp("ENS"); de_hx,p_hx=dp("HARX")
    return dict(market=label,n_oos=nO,raw_sharpe=sr(raw),ens_dES5=de_ens,ens_p=p_ens,harx_dES5=de_hx,harx_p=p_hx)

MK=[("Japan","Japan_MOM_Factor_Daily.csv","Japan_3_Factors_Daily.csv"),
    ("Europe","Europe_MOM_Factor_Daily.csv","Europe_3_Factors_Daily.csv"),
    ("NorthAmerica","North_America_MOM_Factor_Daily.csv","North_America_3_Factors_Daily.csv"),
    ("AsiaPacxJP","Asia_Pacific_ex_Japan_MOM_Factor_Daily.csv","Asia_Pacific_ex_Japan_3_Factors_Daily.csv")]
rows=[]
for lab,wf,mf in MK:
    r=run(load_wml(D+wf),load_mkt(D+mf),lab)
    if r: rows.append(r); print(f"  {lab:>13}: rawSh {r['raw_sharpe']:+.2f} | ENS ΔES5 {r['ens_dES5']:+.2f}(p={r['ens_p']:.2f}) | HARX ΔES5 {r['harx_dES5']:+.2f}(p={r['harx_p']:.2f})")
# 한국 앵커 (full-feature 프로젝트값)
rows.append(dict(market="Korea",n_oos=292,raw_sharpe=0.565,ens_dES5=0.86,ens_p=0.037,harx_dES5=1.34,harx_p=0.003))
print(f"  {'Korea(full)':>13}: rawSh +0.57 | ENS ΔES5 +0.86(p=0.04) | HARX ΔES5 +1.34(p=0.00)")
df=pd.DataFrame(rows)
for col in ["ens_dES5","harx_dES5"]:
    lr=stats.linregress(df["raw_sharpe"],df[col])
    print(f"\n[횡단면 {col}] 기울기 {lr.slope:+.3f} r={lr.rvalue:+.2f} p={lr.pvalue:.3f} · Spearman ρ={stats.spearmanr(df['raw_sharpe'],df[col]).statistic:+.2f} (n={len(df)})")
df.round(4).to_csv(D+"fair_cross.csv",index=False); print("\n[SAVE] fair_cross.csv")
