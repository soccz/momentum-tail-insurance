"""
P7 — WML 구성 강건성 그리드 (심사 방어의 핵심 표) + 실제 leg 회전율 기반 풀비용.

변형 6종 (패널 1회 로드): base(VW·10분위·스킵) / 5분위 / 동일가중 / KOSPI만 / 시총상위50% / 스킵없음(12-0)
각 변형에서: RAW → BS15(RW126) → Ridge-lite(자기 변동성 피처) 샤프·최악월.
질문: "결과가 구성 선택에 의존하는가?" — 전 변형에서 관리>RAW(꼬리), ML≥RW면 강건.

+ base 변형의 승자/패자 멤버십 오버랩으로 월 회전율 → 풀비용 순샤프 = L·WML − c·(2·TO·L + 2·|ΔL|).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

SEED=42
ROOT=Path("/mnt/20t/졸업논문")
MOM="/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet"
LST="/mnt/20t/study/mom_paper_test/data/processed/listings.csv"
Pm=lambda i:i.to_period("M")
TGT=0.12/np.sqrt(12); FT,RT=120,12
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)

print("[panel] loading...",flush=True)
raw=pd.read_parquet(MOM,columns=["Date","Code","AdjClose","MarketCap"]); raw=raw[raw["AdjClose"]>0]
price=raw.pivot_table(index="Date",columns="Code",values="AdjClose",aggfunc="last").sort_index().loc["1990":]
cap=raw.pivot_table(index="Date",columns="Code",values="MarketCap",aggfunc="last").reindex_like(price)
rets=price.pct_change(fill_method=None); rets=rets.where(rets.abs()<=0.5)
me=price.groupby(Pm(price.index)).apply(lambda x:x.index[-1]); me=pd.DatetimeIndex(me.values)
mkt=pd.read_csv(LST)[["Code","Market"]].dropna(); kospi=set(mkt[mkt["Market"]=="유가증권"]["Code"].astype(str).str.zfill(6))
print(f"[panel] {price.shape} | KOSPI 매핑 {len(kospi)}종목",flush=True)

def vw_hold(w0,rw):
    g=(1+rw.fillna(0)).cumprod(); v=g.mul(w0,axis=1).sum(axis=1)
    r=v/v.shift(1)-1; r.iloc[0]=v.iloc[0]-1; return r

def build_variant(nbins=10, weighting="vw", universe="all", skip=True, track_turnover=False):
    win_d,los_d,wml_m=[],[],{}; prev_w,prev_l=set(),set(); tos=[]
    for k in range(12,len(me)-1):
        T=me[k]
        p_now=price.loc[me[k-1]] if skip else price.loc[T]
        sig=p_now/price.loc[me[k-12]]-1
        w=cap.loc[T]; ok=sig.notna()&w.notna()&(w>0)
        if universe=="kospi": ok&=pd.Series(sig.index.isin(kospi),index=sig.index)
        if universe=="sizetop": ok&=(w>=w[ok].median())
        if ok.sum()<max(30,nbins*3): continue
        dec=pd.qcut(sig[ok],nbins,labels=False,duplicates="drop")
        Wn,Ls=dec.index[dec==dec.max()],dec.index[dec==dec.min()]
        hold=price.index[(price.index>T)&(price.index<=me[k+1])]
        if not len(hold): continue
        if weighting=="vw": w0w,w0l=w[Wn]/w[Wn].sum(),w[Ls]/w[Ls].sum()
        else: w0w=pd.Series(1/len(Wn),index=Wn); w0l=pd.Series(1/len(Ls),index=Ls)
        rw_=vw_hold(w0w,rets.loc[hold,Wn]); rl_=vw_hold(w0l,rets.loc[hold,Ls])
        win_d.append(rw_); los_d.append(rl_)
        wml_m[Pm(pd.DatetimeIndex([me[k+1]]))[0]]=(1+rw_).prod()-(1+rl_).prod()
        if track_turnover:
            cw,cl=set(Wn),set(Ls)
            if prev_w: tos.append(1-len(cw&prev_w)/len(prev_w)*0.5-len(cl&prev_l)/len(prev_l)*0.5)
            prev_w,prev_l=cw,cl
    dW,dL=pd.concat(win_d),pd.concat(los_d)
    daily=(dW-dL); daily=daily[~daily.index.duplicated()]
    monthly=pd.Series(wml_m).sort_index()
    return daily,monthly,(np.mean(tos) if tos else np.nan)

def eval_variant(daily,monthly):
    sq=daily**2
    fvar=(21*sq.rolling(126,min_periods=126).mean()).groupby(Pm(sq.index)).last()
    feats=pd.DataFrame({f"vol_{l}":(np.sqrt(252/w_*sq.rolling(w_,min_periods=w_).sum())*100).groupby(Pm(sq.index)).last()
                        for l,w_ in [("1m",21),("3m",63),("6m",126),("12m",252)]})
    tgtv=sq.groupby(Pm(sq.index)).sum().shift(-1)
    wn=monthly.shift(-1)
    df=pd.concat([fvar.rename("fvar"),feats,tgtv.rename("v"),wn.rename("r")],axis=1).dropna()
    X=df[feats.columns].values; y=np.log(df["v"].values); n=len(df); oo=np.arange(FT,n)
    pred=np.full(n,np.nan)
    for t0 in range(FT,n,RT):
        tr,te=np.arange(t0),np.arange(t0,min(t0+RT,n))
        sc=StandardScaler().fit(X[tr]); Z=sc.transform(X)
        m=RidgeCV(alphas=np.logspace(-3,3,13)).fit(Z[tr],y[tr]); resid=y[tr]-m.predict(Z[tr])
        pred[te]=np.clip(np.exp(m.predict(Z[te])+resid.var()/2),df["v"].values[tr].min()*.5,df["v"].values[tr].max()*2)
    r=df["r"].values
    out={}
    out["RAW"]=(sr(r[oo]),np.nanmin(r[oo])*100)
    out["RW126"]=(sr((TGT/np.sqrt(df["fvar"].values)*r)[oo]),np.nanmin((TGT/np.sqrt(df["fvar"].values)*r)[oo])*100)
    out["Ridge"]=(sr((TGT/np.sqrt(pred)*r)[oo]),np.nanmin((TGT/np.sqrt(pred)*r)[oo])*100)
    return out,len(oo)

VARIANTS=[("base VW·10분위·스킵",dict()),("5분위",dict(nbins=5)),("동일가중 EW",dict(weighting="ew")),
          ("KOSPI만",dict(universe="kospi")),("시총상위50%",dict(universe="sizetop")),("스킵없음 12-0",dict(skip=False))]
rows={}; to_base=np.nan
for nm,kw in VARIANTS:
    d,mo,to=build_variant(track_turnover=(nm.startswith("base")),**kw)
    if nm.startswith("base"): to_base=to
    res,n_oo=eval_variant(d,mo)
    rows[nm]={f"{k}_{x}":v[i] for k,v in res.items() for i,x in enumerate(("샤프","최악월"))} | {"n_oos":n_oo}
    print(f"  {nm:<16} RAW {res['RAW'][0]:.2f} → RW {res['RW126'][0]:.2f} → ML {res['Ridge'][0]:.2f} | "
          f"최악 {res['RAW'][1]:.0f}→{res['RW126'][1]:.0f}→{res['Ridge'][1]:.0f}%",flush=True)
grid=pd.DataFrame(rows).T
print("\n===== P7 구성 강건성 그리드 =====");print(grid.round(2).to_string())

# ---------- 실제 회전율 풀비용 ----------
print(f"\n[회전율] base 변형 leg 평균 월회전율 TO ≈ {to_base:.2%} (멤버십 교체 기준, 미국 논문 74~75%와 비교)")
W=pd.read_csv(ROOT/"data/processed/p3_weights.csv",index_col=0); W.index=pd.PeriodIndex(W.index,freq="M")
ds=pd.read_csv(ROOT/"data/processed/ml_dataset.csv",index_col=0); ds.index=pd.PeriodIndex(ds.index,freq="M")
wn=ds["tgt_wml_next"].reindex(W.index).values
fc_rows={}
for m in ["RW126","HARX","Ridge","ENS"]:
    L=W[m].values; dL=np.abs(np.diff(L,prepend=L[0]))
    row={}
    for c_bp in [0,30,60,100]:
        net=L*wn - c_bp/1e4*(2*to_base*L + 2*dL)
        row[f"{c_bp}bp"]=sr(net)
    fc_rows[m]=row
fcost=pd.DataFrame(fc_rows).T
print("\n===== 풀비용 순샤프 (기초 WML 회전 포함: c×(2·TO·L+2·|ΔL|)) =====")
print(fcost.round(3).to_string())

grid.round(3).to_csv(ROOT/"output/tables/p7_construction_grid.csv")
fcost.round(4).to_csv(ROOT/"output/tables/p7_full_cost.csv")
pd.DataFrame({"leg_turnover_monthly":[to_base]}).to_csv(ROOT/"output/tables/p7_turnover.csv",index=False)
assert np.isfinite(grid.filter(like="샤프").values).all()
print("\n[GATES] 6변형 전부 산출 ✓ · 표 3종 저장 ✓")
