"""
P23 (전략 M2) — US 소형주 quintile WML: 국가와 개인지배·고유분산을 분리하는 티어 결정 실험.

[사전지정 — 실행 전 커밋]
- 질문: 미국 안에서 소형주(ME1) quintile 극단 WML — 개인투자자 비중·고유위험이 크고 모멘텀이
  상대적으로 약한 하위그룹 — 은 한국형 ML 꼬리 이득을 보이는가? 대형주(ME5)는 within-market 대조군.
- 가설: 효과의 원천이 X2(고유분산)/X4(개인지배)라면 US 소형에서 양성·대형에서 음성 — 국가와 분리 식별.
  효과가 한국×극단구성 상호작용(E22)이라면 US 소형에서도 음성.
- 데이터: KF 25 Size-Momentum(5×5) 일별 VW (1926–2026) — WML_ME1=SMALL HiPRIOR−SMALL LoPRIOR,
  WML_ME5=BIG HiPRIOR−BIG LoPRIOR. 시장상태 피처는 F-F 일별 3팩터(Mkt-RF+RF).
- 파이프라인: P21/P22와 동일 (HAR/HARX/Ridge/HGB/ENS, RW126 벤치마크, ΔES5 paired 블록부트 B=3000, seed42).
- 성공기준(사전지정, ME1 ENS 기준):
  (a) ΔES5 ≥ +0.5%p & p<0.10 → 양성: X2/X4 분리 식별, docs/08의 JFQA 상향 트리거
  (b) ΔES5 < +0.3%p 또는 p≥0.10 → 음성: 한국×구성 상호작용 경계 증거 +1
  (c) 그 외 → 미확정
- 게이트: ME1 raw 왜도<0(크래시 존재) · OOS n≥600 · ME1 vs ME5 파이프라인 동일성(코드 공유).
- 원장 E23. 논문 반영 없음(§6-2).
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

SEED=42; rng=np.random.default_rng(SEED)
D="/mnt/20t/tmp/claude-1001/-mnt-20t-----/da2b9872-77c2-46d2-a247-cd601338a0be/scratchpad/intl/"
OUT="/mnt/20t/졸업논문/output/tables/"

def load_25(fp):
    L=open(fp,encoding="latin1").read().splitlines()
    i0=next(i for i,l in enumerate(L) if "Average Value Weighted Returns -- Daily" in l)
    hdr=[h.strip() for h in L[i0+1].split(",")]
    rows=[]
    for l in L[i0+2:]:
        t=l.split(","); k=t[0].strip()
        if k.isdigit() and len(k)==8:
            try: rows.append([k]+[float(x) for x in t[1:len(hdr)]])
            except: break
        elif rows: break
    df=pd.DataFrame(rows,columns=["d"]+hdr[1:]).set_index("d")
    df.index=pd.to_datetime(df.index,format="%Y%m%d")
    return df.replace([-99.99,-999.0],np.nan)/100.0
P25=load_25(D+"25_Portfolios_ME_Prior_12_2_Daily.csv")
print("[25포트] 열:",list(P25.columns)[:3],"...",list(P25.columns)[-2:], P25.index[0].date(), P25.index[-1].date())
wml={"ME1(소형)":(P25["SMALL HiPRIOR"]-P25["SMALL LoPRIOR"]).dropna(),
     "ME5(대형)":(P25["BIG HiPRIOR"]-P25["BIG LoPRIOR"]).dropna()}

def load_mkt():
    L=open(D+"F-F_Research_Data_Factors_daily.csv",encoding="latin1").read().splitlines()
    i0=next(i for i,l in enumerate(L) if l.strip().startswith(",Mkt-RF"))
    rows=[]
    for l in L[i0+1:]:
        t=l.split(","); k=t[0].strip()
        if k.isdigit() and len(k)==8:
            try: rows.append([k,float(t[1]),float(t[4])])
            except: break
        elif rows: break
    df=pd.DataFrame(rows,columns=["d","mktrf","rf"]).set_index("d")
    df.index=pd.to_datetime(df.index,format="%Y%m%d")
    return (df["mktrf"]+df["rf"])/100.0
mktd=load_mkt()

Pm=lambda i:i.to_period("M")
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
def es5(r): r=r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1,int(0.05*len(r)))])*100

def run(wd,label):
    wm=(1+wd).groupby(Pm(wd.index)).prod()-1
    g=wd.groupby(Pm(wd.index)); RV=g.apply(lambda x:(x**2).sum()); SEMI=g.apply(lambda x:(x[x<0]**2).sum())
    RW=(21*(wd**2).rolling(126,min_periods=126).mean()); RWme=RW.groupby(Pm(RW.index)).last()
    mm=(1+mktd).groupby(Pm(mktd.index)).prod()-1
    df=pd.DataFrame(index=RV.index)
    df["rv"]=RV; df["l1"]=np.log(RV); df["l3"]=np.log(RV.rolling(3).mean())
    df["l6"]=np.log(RV.rolling(6).mean()); df["l12"]=np.log(RV.rolling(12).mean()); df["semir"]=(SEMI/RV).clip(0,1)
    df["mret1"]=mm.reindex(df.index); df["mret12"]=mm.reindex(df.index).rolling(12).sum()
    df["mvol6"]=mm.reindex(df.index).rolling(6).std()
    cm=(1+mm).cumprod(); df["mdd"]=(cm/cm.cummax()-1).reindex(df.index)
    df["bear"]=(mm.reindex(df.index).rolling(24).sum()<0).astype(float)
    df["rw"]=RWme; df["tgt"]=np.log(RV.shift(-1)); df["wmln"]=wm.reindex(df.index).shift(-1)
    FEAT=["l1","l3","l6","l12","semir","mret1","mret12","mvol6","mdd","bear"]
    HARF=["l1","l3","l12"]; HARXF=HARF+["bear","mvol6","mdd"]
    df=df.dropna(subset=FEAT+["tgt","wmln","rw"]).copy(); n=len(df); INIT=120
    X=df[FEAT].values; y=df["tgt"].values; rvcol=df["rv"].values
    def fp(cols,model):
        idx=[FEAT.index(c) for c in cols]; pred=np.full(n,np.nan)
        for s in range(INIT,n,12):
            lo,hi=np.log(rvcol[:s].min()*0.5),np.log(rvcol[:s].max()*2)
            model.fit(X[:s][:,idx],y[:s]); te=slice(s,min(s+12,n))
            pred[te]=np.exp(np.clip(model.predict(X[te][:,idx]),lo,hi))
        return pred
    oos=np.arange(INIT,n); wn=df["wmln"].values
    fc={"RW126":df["rw"].values,
        "HAR":fp(HARF,LinearRegression()), "HARX":fp(HARXF,LinearRegression()),
        "Ridge":fp(FEAT,RidgeCV(alphas=np.logspace(-3,3,20))),
        "HGB":fp(FEAT,HistGradientBoostingRegressor(max_depth=3,max_iter=150,random_state=SEED))}
    fc["ENS"]=np.nanmean([fc["HAR"],fc["Ridge"],fc["HGB"]],axis=0)
    TGT=0.12/np.sqrt(12); RET={m:(TGT/np.sqrt(f)*wn)[oos] for m,f in fc.items()}; rawr=wn[oos]
    nO=len(oos); B=3000
    def bidx(): return np.concatenate([np.arange(s,s+12)%nO for s in rng.integers(0,nO,size=int(np.ceil(nO/12)))])[:nO]
    res={}
    print(f"\n=== {label}: raw 샤프 {sr(rawr):+.2f} 왜도 {stats.skew(rawr[np.isfinite(rawr)]):+.2f} 최악 {np.nanmin(rawr)*100:.0f}% (OOS n={nO}) ===")
    for m in ["ENS","HARX","Ridge","HAR"]:
        do=es5(RET[m])-es5(RET["RW126"])
        bs=np.array([es5(RET[m][ix:=bidx()])-es5(RET["RW126"][ix]) for _ in range(B)])
        res[m]=dict(dES5=do,p=float((bs<=0).mean()),sharpe=sr(RET[m]))
        print(f"  {m:>5}: ΔES5 {do:+.2f}%p (p={res[m]['p']:.3f}) 샤프 {sr(RET[m]):.3f}")
    print(f"  관리: ΔES5 {es5(RET['RW126'])-es5(rawr):+.2f}%p · 샤프 {sr(rawr):.2f}→{sr(RET['RW126']):.2f}")
    return res, nO, stats.skew(rawr[np.isfinite(rawr)])

out={}
for lab,wd in wml.items():
    out[lab],nO,sk=run(wd,lab)

de,p=out["ME1(소형)"]["ENS"]["dES5"],out["ME1(소형)"]["ENS"]["p"]
verdict="(a) 양성 — X2/X4 분리 식별, JFQA 트리거" if (de>=0.5 and p<0.10) else \
        "(b) 음성 — 한국×구성 상호작용 경계 증거 +1" if (de<0.3 or p>=0.10) else "(c) 미확정"
print(f"\n[사전지정 판정, ME1 ENS] ΔES5={de:+.2f} p={p:.3f} → {verdict}")
rows=[]
for lab,r in out.items():
    for m,v in r.items(): rows.append(dict(group=lab,model=m,**v))
pd.DataFrame(rows).round(4).to_csv(OUT+"p23_us_smallcap.csv",index=False)
assert nO>=600, "OOS 부족"
print("[GATES] n_oos ✓ · 표 저장 ✓")
