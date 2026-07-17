"""
P22 (전략 M1) — 한국 2×3 size-momentum 절제: 구성(X1) confound의 국가-고정 식별.

[사전지정 — 실행 전 커밋]
- 질문: 한국 ML 꼬리 이득(+0.86%p, decile 기준)이 KF식 2×3 분산 팩터 구성에서도 생존하는가?
- 가설: 효과가 극단 decile 구성에 산다면(X1), 2×3에서는 죽어야 한다(국제 2×3 전부 음성과 정합).
- 방법: FnGuide 패널 → 매월말 시총 중위 2분할 × 모멘텀(t−12~t−2) 30/70 → 6개 VW 포트폴리오,
  WML_2x3 = (BigWin+SmallWin)/2 − (BigLose+SmallLose)/2, buy-and-hold 일별.
  이후 fair_cross(P21)와 동일 파이프라인: WML 자체변동성 + 한국 시장상태 피처(ff_own_daily rmrf),
  HAR/HARX/Ridge/HGB/ENS, RW126 벤치마크, ΔES5 paired 블록부트스트랩(blk=12, B=3000, seed42).
- 성공기준(사전지정):
  (a) ΔES5(ENS) < +0.3%p & p > 0.10 → 구성(X1)이 조절변수로 식별 (효과는 극단 구성에 삶)
  (b) ΔES5(ENS) ≥ +0.5%p & p < 0.10 → X1 기각 (한국 특수성은 구성이 아님 → X4/X5로)
  (c) 그 외 → 미확정으로 보고
- 게이트: 2×3 WML 원시 corr(decile WML) > 0.5 · 표본외 n ≥ 250 · raw 샤프 유한.
- 원장 E22. 논문 반영 없음(헌법 §6-2).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path("/mnt/20t/졸업논문")
MOM="/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet"
Pm=lambda i:i.to_period("M")

print("[panel] loading...",flush=True)
raw=pd.read_parquet(MOM,columns=["Date","Code","AdjClose","MarketCap"]); raw=raw[raw["AdjClose"]>0]
price=raw.pivot_table(index="Date",columns="Code",values="AdjClose",aggfunc="last").sort_index().loc["1990":]
cap=raw.pivot_table(index="Date",columns="Code",values="MarketCap",aggfunc="last").reindex_like(price)
rets=price.pct_change(fill_method=None); rets=rets.where(rets.abs()<=0.5)   # 기준 필터 θ=0.5 (본편과 동일)
me=price.groupby(Pm(price.index)).apply(lambda x:x.index[-1]); me=pd.DatetimeIndex(me.values)
print(f"[panel] {price.shape[1]}종목, {price.index[0].date()}~{price.index[-1].date()}",flush=True)

# ---- 2×3 구성 (매월 리밸런스, VW, buy-and-hold) ----
print("[build] 2x3 size-momentum...",flush=True)
port_daily={k:{} for k in ("BW","BL","SW","SL")}
wml_m={}
for k in range(12,len(me)-1):
    T=me[k]
    sig=price.loc[me[k-1]]/price.loc[me[k-12]]-1          # t−12~t−2 (최근월 스킵)
    w=cap.loc[T]; ok=sig.notna()&w.notna()&(w>0)
    if ok.sum()<60: continue
    s,wv=sig[ok],w[ok]
    med=wv.median(); lo,hi=s.quantile(0.3),s.quantile(0.7)
    grp={"BW":(wv>=med)&(s>=hi), "BL":(wv>=med)&(s<=lo), "SW":(wv<med)&(s>=hi), "SL":(wv<med)&(s<=lo)}
    hold=price.index[(price.index>T)&(price.index<=me[k+1])]
    if not len(hold): continue
    mret={}
    for g,mask in grp.items():
        mem=s.index[mask]
        if len(mem)<5: mret=None; break
        w0=wv[mem]/wv[mem].sum()
        gcum=(1+rets.loc[hold,mem].fillna(0)).cumprod()
        v=gcum.mul(w0,axis=1).sum(axis=1)
        rr=v/v.shift(1)-1; rr.iloc[0]=v.iloc[0]-1
        port_daily[g][me[k+1]]=rr; mret[g]=(1+rr).prod()-1
    if mret is None: continue
    wml_m[Pm(pd.DatetimeIndex([me[k+1]]))[0]]=0.5*(mret["BW"]+mret["SW"])-0.5*(mret["BL"]+mret["SL"])
D={g:pd.concat(port_daily[g].values()) for g in port_daily}
wd=(0.5*(D["BW"]+D["SW"])-0.5*(D["BL"]+D["SL"])).dropna(); wd=wd[~wd.index.duplicated()]
wm=pd.Series(wml_m).sort_index()
print(f"[build] 2x3 WML: 일별 {len(wd)}, 월별 {len(wm)}",flush=True)

# ---- 게이트 1: decile WML과의 정합 ----
dec=pd.read_csv(ROOT/"data/processed/wml_monthly.csv",parse_dates=["month"]).set_index("month")["wml"]
dec.index=dec.index.to_period("M")
common=wm.index.intersection(dec.index)
corr=np.corrcoef(wm.loc[common],dec.loc[common])[0,1]
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
print(f"[gate1] corr(2x3, decile)={corr:.3f} · 2x3 raw 샤프 {sr(wm.values):+.3f} (decile {sr(dec.loc[common].values):+.3f}) · 왜도 {stats.skew(wm.dropna()):+.2f}")

# ---- 동일 파이프라인 (P21 fair_cross와 동일) ----
mktd=pd.read_csv(ROOT/"data/processed/ff_own_daily.csv",parse_dates=["Date"]).set_index("Date")["rmrf"].dropna()
def es5(r): r=r[np.isfinite(r)]; return np.mean(np.sort(r)[:max(1,int(0.05*len(r)))])*100
g=wd.groupby(Pm(wd.index))
RV=g.apply(lambda x:(x**2).sum()); SEMI=g.apply(lambda x:(x[x<0]**2).sum())
RW=(21*(wd**2).rolling(126,min_periods=126).mean()); RWme=RW.groupby(Pm(RW.index)).last()
mm=(1+mktd).groupby(Pm(mktd.index)).prod()-1
df=pd.DataFrame(index=RV.index)
df["rv"]=RV; df["l1"]=np.log(RV); df["l3"]=np.log(RV.rolling(3).mean())
df["l6"]=np.log(RV.rolling(6).mean()); df["l12"]=np.log(RV.rolling(12).mean()); df["semir"]=(SEMI/RV).clip(0,1)
df["mret1"]=mm.reindex(df.index); df["mret12"]=mm.reindex(df.index).rolling(12).sum()
df["mvol6"]=mm.reindex(df.index).rolling(6).std()
cm=(1+mm).cumprod(); df["mdd"]=(cm/cm.cummax()-1).reindex(df.index)
df["bear"]=(mm.reindex(df.index).rolling(24).sum()<0).astype(float)
df["rw"]=RWme; df["tgt"]=np.log(RV.shift(-1)); df["rvn"]=RV.shift(-1)
wm_p=wm.copy(); df["wmln"]=wm_p.reindex(df.index).shift(-1)
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
fc={"RW126":df["rw"].values}
fc["HAR"]=fp(HARF,LinearRegression()); fc["HARX"]=fp(HARXF,LinearRegression())
fc["Ridge"]=fp(FEAT,RidgeCV(alphas=np.logspace(-3,3,20)))
fc["HGB"]=fp(FEAT,HistGradientBoostingRegressor(max_depth=3,max_iter=150,random_state=SEED))
fc["ENS"]=np.nanmean([fc["HAR"],fc["Ridge"],fc["HGB"]],axis=0)
TGT=0.12/np.sqrt(12); RET={m:(TGT/np.sqrt(f)*wn)[oos] for m,f in fc.items()}; rawr=wn[oos]
nO=len(oos); B=3000
def bidx(): return np.concatenate([np.arange(s,s+12)%nO for s in rng.integers(0,nO,size=int(np.ceil(nO/12)))])[:nO]
rows={}
for m in ["ENS","HARX","Ridge","HAR"]:
    do=es5(RET[m])-es5(RET["RW126"])
    bs=np.array([es5(RET[m][ix:=bidx()])-es5(RET["RW126"][ix]) for _ in range(B)])
    rows[m]=dict(dES5=do,p=float((bs<=0).mean()),sharpe=sr(RET[m]))
    print(f"  {m:>5}: ΔES5 {do:+.2f}%p (p={rows[m]['p']:.3f}) 샤프 {sr(RET[m]):.3f}")
dmg=es5(RET["RW126"])-es5(rawr)
print(f"  관리(RW126−무관리): ΔES5 {dmg:+.2f}%p · raw 샤프 {sr(rawr):+.3f} → RW {sr(RET['RW126']):+.3f}")

# ---- 사전지정 판정 ----
de,p=rows["ENS"]["dES5"],rows["ENS"]["p"]
verdict="(a) X1 식별: 효과는 극단 decile 구성에 삶" if (de<0.3 and p>0.10) else \
        "(b) X1 기각: 2×3에서도 생존 — 구성이 아님" if (de>=0.5 and p<0.10) else "(c) 미확정"
print(f"\n[사전지정 판정] ENS ΔES5={de:+.2f} p={p:.3f} → {verdict}")
out=pd.DataFrame(rows).T; out.loc["mgmt","dES5"]=dmg
out.round(4).to_csv(ROOT/"output/tables/p22_kr_2x3.csv")
wd.rename("wml_2x3").to_csv(ROOT/"data/processed/wml_2x3_daily.csv")
wm.rename("wml_2x3").to_csv(ROOT/"data/processed/wml_2x3_monthly.csv")
assert corr>0.5 and nO>=250 and np.isfinite(sr(rawr)), "게이트 실패"
print(f"[GATES] corr>{0.5} ✓ · n_oos={nO} ✓ · 표 저장 ✓")
