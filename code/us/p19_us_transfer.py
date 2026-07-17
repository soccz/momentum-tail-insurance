"""
미국 병렬 2단계: US WML에 한국과 동일한 예측 사다리 → 전이곡선.
핵심 질문: 강한 팩터(미국)에서 ML 예측 개선이 샤프로 전이되는가, 아니면 여기서도 꼬리뿐인가?
설계는 한국(p2/p3/p16)과 동일 — 스케일 규칙 고정, 예측기만 교체. 월별 실현분산 예측.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
D = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "us") + "/"  # 공개 데이터 — 레포 상대경로
SEED=42; rng=np.random.default_rng(SEED)

# ---- 데이터 ----
wd=pd.read_csv(D+"us_wml_daily.csv",index_col=0,parse_dates=True).iloc[:,0]
wm=pd.read_csv(D+"us_wml_monthly.csv",index_col=0,parse_dates=True).iloc[:,0]
wm.index=wm.index.to_period("M")
def load_ff():
    lines=open(D+"F-F_Research_Data_Factors.csv").read().splitlines()
    i0=next(i for i,l in enumerate(lines) if l.strip().startswith(",Mkt-RF"))
    rows=[]
    for l in lines[i0+1:]:
        t=l.split(","); k=t[0].strip()
        if k.isdigit() and len(k)==6:
            try: rows.append([k]+[float(x) for x in t[1:5]])
            except: break
        elif rows: break
    df=pd.DataFrame(rows,columns=["m","mktrf","smb","hml","rf"]).set_index("m")
    df.index=pd.PeriodIndex(df.index,freq="M"); return df/100.0*100  # keep pct→ decimal below
ff=load_ff(); ff=ff/100.0   # to decimal
mkt=ff["mktrf"]+ff["rf"]     # market total

# ---- 월별 실현분산(일별 제곱합) + 다운사이드 ----
g=wd.groupby(wd.index.to_period("M"))
RV=g.apply(lambda x:(x**2).sum())                      # 월별 실현분산 (월단위)
SEMI=g.apply(lambda x:(x[x<0]**2).sum())               # 하방 준분산
RW126=(21*(wd**2).rolling(126,min_periods=126).mean()) # BS15 예보(월변동성)
RW126_me=RW126.groupby(RW126.index.to_period("M")).last()

# ---- 피처 프레임 (전부 t까지 정보, 타깃=t+1 실현분산) ----
df=pd.DataFrame(index=RV.index)
df["rv"]=RV
df["l1"]=np.log(RV)
df["l3"]=np.log(RV.rolling(3).mean())
df["l6"]=np.log(RV.rolling(6).mean())
df["l12"]=np.log(RV.rolling(12).mean())
df["semir"]=(SEMI/RV).clip(0,1)
df["mvol6"]=mkt.rolling(6).std()
df["mret1"]=mkt
df["mret12"]=mkt.rolling(12).sum()
cummkt=(1+mkt).cumprod(); df["mdd"]=cummkt/cummkt.cummax()-1
df["bear"]=(mkt.rolling(24).sum()<0).astype(float)
df["rw126"]=RW126_me
df["target"]=np.log(RV.shift(-1))                       # 로그 실현분산 t+1
df["rv_next"]=RV.shift(-1)
df["wml_next"]=wm.reindex(df.index).shift(-1)
df=df.dropna(subset=["l12","mret12","bear","target","wml_next","rw126"]).copy()
n=len(df); print(f"[US 피처] {df.index[0]}~{df.index[-1]} n={n}")

FEAT=["l1","l3","l6","l12","semir","mvol6","mret1","mret12","mdd","bear"]
HARF=["l1","l3","l12"]; HARXF=HARF+["bear","mvol6","mdd","semir"]
INIT=120; STEP=12
oos=np.arange(INIT,n)
X=df[FEAT].values; ylog=df["target"].values; rv_next=df["rv_next"].values
def fit_predict(cols,model):
    idx=[FEAT.index(c) for c in cols] if cols else list(range(len(FEAT)))
    pred=np.full(n,np.nan)
    for s in range(INIT,n,STEP):
        tr=slice(0,s); Xtr=X[tr][:,idx]; ytr=ylog[tr]
        lo,hi=np.log(df["rv"].values[tr].min()*0.5), np.log(df["rv"].values[tr].max()*2)
        model.fit(Xtr,ytr)
        te=slice(s,min(s+STEP,n)); p=model.predict(X[te][:,idx])
        pred[te]=np.exp(np.clip(p,lo,hi))
    return pred

fc={}
# 비회귀 기준
fc["ExpandingMean"]=np.array([df["rv"].values[:t].mean() for t in range(n)])
fc["RW126"]=df["rw126"].values
ew=np.full(n,np.nan); lam=0.94; ew[0]=df["rv"].values[0]
for t in range(1,n): ew[t]=lam*ew[t-1]+(1-lam)*df["rv"].values[t-1]
fc["EWMA"]=ew
# 회귀·ML
from sklearn.linear_model import LinearRegression
fc["HAR"]=fit_predict(HARF, LinearRegression())
fc["HARX"]=fit_predict(HARXF, LinearRegression())
fc["Ridge"]=fit_predict(FEAT, RidgeCV(alphas=np.logspace(-3,3,20)))
fc["Lasso"]=fit_predict(FEAT, LassoCV(n_alphas=20,max_iter=5000,random_state=SEED))
fc["RF"]=fit_predict(FEAT, RandomForestRegressor(n_estimators=200,max_depth=5,random_state=SEED,n_jobs=-1))
fc["HGB"]=fit_predict(FEAT, HistGradientBoostingRegressor(max_depth=3,max_iter=200,random_state=SEED))
fc["ENS"]=np.nanmean([fc["HAR"],fc["Ridge"],fc["HGB"]],axis=0)

# ---- 평가 ----
act=rv_next
def qlike(f):
    m=oos; r=act[m]/f[m]; return np.mean(r-np.log(r)-1)
em=fc["ExpandingMean"]
def oosr2(f):
    m=oos; sse=np.sum((act[m]-f[m])**2); sst=np.sum((act[m]-em[m])**2); return (1-sse/sst)*100
TGT=0.12/np.sqrt(12)
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
def es5(r): r=r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1,int(0.05*len(r)))])*100
wn=df["wml_next"].values
RET={}; rows={}
for mdl,f in fc.items():
    L=TGT/np.sqrt(f); r=L*wn; RET[mdl]=r[oos]
    rows[mdl]=dict(QLIKE=qlike(f),OOS_R2=oosr2(f),Sharpe=sr(r[oos]),ES5=es5(r[oos]),
                   skew=stats.skew(r[oos][np.isfinite(r[oos])]),worst=np.nanmin(r[oos])*100)
raw=wn[oos]; rows["무관리"]=dict(QLIKE=np.nan,OOS_R2=np.nan,Sharpe=sr(raw),ES5=es5(raw),
                                skew=stats.skew(raw[np.isfinite(raw)]),worst=np.nanmin(raw)*100)
res=pd.DataFrame(rows).T
print(f"\n===== US 전이곡선 리더보드 (OOS {len(oos)}개월) =====")
print(res[["QLIKE","OOS_R2","Sharpe","ES5","skew","worst"]].round(3).to_string())

# ---- 전이 통계 ----
core=res.drop("무관리").dropna(subset=["OOS_R2"]); pos=core[core["OOS_R2"]>0]
sl=stats.linregress(pos["OOS_R2"],pos["Sharpe"])
print(f"\n[US 전이] 샤프–R² 기울기 {sl.slope:+.5f}/%p (R²>0 {len(pos)}개, r={sl.rvalue:.2f})")
base=es5(RET["RW126"])
for m in ["ENS","HARX","Ridge","HAR"]:
    print(f"  {m}: ΔES5 vs RW126 = {es5(RET[m])-base:+.2f}%p")
# γ-스윕 CRRA (관리가치·ML증분)
def crra(r,g):
    x=1+r[np.isfinite(r)]
    if (x<=0).any(): return np.nan
    return (((np.mean(x**(1-g)))**(1/(1-g)))**12-1)*100
print("\n[US CRRA CE 증분 (연%)]")
for g in (2,5,10,20):
    mgmt=crra(RET["RW126"],g)-crra(raw,g); ml=crra(RET["ENS"],g)-crra(RET["RW126"],g)
    print(f"  γ={g:>2}: 관리가치 {mgmt:+.1f} · ML증분 {ml:+.2f}")
res.round(4).to_csv(D+"us_leaderboard.csv")

# ---- 미국 꼬리 유의성: paired 블록부트스트랩 (한국과 동일 방법, blk=12, B=5000) ----
nO=len(oos); B=5000
def bidx():
    return np.concatenate([np.arange(s,s+12)%nO for s in rng.integers(0,nO,size=int(np.ceil(nO/12)))])[:nO]
print("\n===== 미국 꼬리 ΔES5 유의성 (paired 블록부트스트랩, B=5000) =====")
baseR=RET["RW126"]
for m in ["ENS","HARX","Ridge","HAR"]:
    do=es5(RET[m])-es5(baseR)
    bs=np.array([es5(RET[m][ix:=bidx()])-es5(baseR[ix]) for _ in range(B)])
    p_impr=float((bs<=0).mean())   # P(개선≤0)
    print(f"  {m:>5}: ΔES5 {do:+.2f}%p · P(≤0)={p_impr:.3f} {'(개선 유의)' if p_impr<0.05 else '(무의미/악화)'}")
# 관리 자체(RW126 vs 무관리) 꼬리 유의성 — 강한 시장에서 관리의 꼬리 가치
raw_oos=wn[oos]
do_m=es5(baseR)-es5(raw_oos)
bs_m=np.array([es5(baseR[ix:=bidx()])-es5(raw_oos[ix]) for _ in range(B)])
print(f"  관리(RW126−무관리) ΔES5 {do_m:+.2f}%p · P(≤0)={float((bs_m<=0).mean()):.3f}")
print("\n[SAVE] us_leaderboard.csv")
