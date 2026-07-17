"""
P13 — 상폐 처리 3-way + 이상치 필터 감사 (심사자 must-fix #8, #10).

상폐: del-a(가치동결=현행) / del-b30(마지막 거래 다음날 −30%) / del-b100(−100% 파산가정)
  → raw WML·RW126·꼬리 헤드라인이 처리 방식에 흔들리는가 ("약한 팩터" 전제의 생사)
필터: |일별수익|>θ 클립의 적중 분포(연도별) + θ∈{0.3, 0.5, 1.0, 없음} 스윕
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path("/mnt/20t/졸업논문")
MOM="/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet"
Pm=lambda i:i.to_period("M"); TGT=0.12/np.sqrt(12); FT=120
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)

print("[panel] loading...",flush=True)
raw=pd.read_parquet(MOM,columns=["Date","Code","AdjClose","MarketCap"]); raw=raw[raw["AdjClose"]>0]
price=raw.pivot_table(index="Date",columns="Code",values="AdjClose",aggfunc="last").sort_index().loc["1990":]
cap=raw.pivot_table(index="Date",columns="Code",values="MarketCap",aggfunc="last").reindex_like(price)
rets0=price.pct_change(fill_method=None)
me=price.groupby(Pm(price.index)).apply(lambda x:x.index[-1]); me=pd.DatetimeIndex(me.values)

# 상폐 이벤트: 마지막 유효가 다음 거래일 위치
last_valid=price.apply(lambda s: s.last_valid_index())
didx=price.index
delist_pos={}
for c,lv in last_valid.items():
    if pd.isna(lv) or lv>=didx[-30]: continue          # 표본 말미는 상폐 아님
    j=didx.get_loc(lv)
    if j+1<len(didx): delist_pos[c]=j+1
print(f"[delist] 상폐(마지막가가 표본 종료 30일 이전) 종목 {len(delist_pos)}개")

def make_rets(theta, delist_ret=None):
    r=rets0.where(rets0.abs()<=theta) if theta else rets0.copy()
    if delist_ret is not None:
        for c,j in delist_pos.items():
            r.iloc[j, r.columns.get_loc(c)]=delist_ret   # 마지막가 다음날 강제 수익
    return r

def build(rets):
    win,los,wm={},{},{}
    for k in range(12,len(me)-1):
        T=me[k]
        sig=price.loc[me[k-1]]/price.loc[me[k-12]]-1
        w=cap.loc[T]; ok=sig.notna()&w.notna()&(w>0)
        if ok.sum()<30: continue
        dec=pd.qcut(sig[ok],10,labels=False,duplicates="drop")
        Wn,Ls=dec.index[dec==dec.max()],dec.index[dec==dec.min()]
        hold=price.index[(price.index>T)&(price.index<=me[k+1])]
        if not len(hold): continue
        out=[]
        for mem in (Wn,Ls):
            w0=w[mem]/w[mem].sum()
            g=(1+rets.loc[hold,mem].fillna(0)).cumprod()
            v=g.mul(w0,axis=1).sum(axis=1)
            rr=v/v.shift(1)-1; rr.iloc[0]=v.iloc[0]-1; out.append(rr)
        win[me[k+1]],los[me[k+1]]=out
        wm[Pm(pd.DatetimeIndex([me[k+1]]))[0]]=(1+out[0]).prod()-(1+out[1]).prod()
    dW=pd.concat(win.values()); dL=pd.concat(los.values())
    daily=(dW-dL); daily=daily[~daily.index.duplicated()]
    return daily,pd.Series(wm).sort_index()

def evaluate(daily,monthly,label):
    sq=daily**2
    fvar=(21*sq.rolling(126,min_periods=126).mean()).groupby(Pm(sq.index)).last()
    r=monthly.shift(-1)
    df=pd.concat([fvar.rename("f"),r.rename("r")],axis=1).dropna()
    oo=np.arange(FT,len(df)); rr=df["r"].values; L=TGT/np.sqrt(df["f"].values)
    es5=lambda x: np.mean(np.sort(x)[:max(1,int(.05*len(x)))])*100
    return dict(라벨=label,
        raw_sharpe=sr(rr[oo]), raw_skew=stats.skew(monthly.dropna()), raw_worst=monthly.min()*100,
        rw_sharpe=sr((L*rr)[oo]), rw_ES5=es5((L*rr)[oo]))

print("\n===== (A) 상폐 처리 3-way (θ=0.5 고정) =====")
rows=[]
for lab,dr in [("del-a 동결(현행)",None),("del-b −30%",-0.30),("del-b −100%",-1.00)]:
    d,m=build(make_rets(0.5,dr)); rows.append(evaluate(d,m,lab)); print(f"  {rows[-1]}",flush=True)
print("\n===== (B) 이상치 필터 θ 스윕 (del-a 고정) =====")
for lab,th in [("θ=0.3",0.3),("θ=0.5(현행)",None),("θ=1.0",1.0),("필터 없음",np.inf)]:
    if th is None: continue                              # 현행은 (A) 첫 행과 동일
    d,m=build(make_rets(th)); rows.append(evaluate(d,m,lab)); print(f"  {rows[-1]}",flush=True)
res=pd.DataFrame(rows).set_index("라벨")
print("\n",res.round(3).to_string())

# 필터 적중 감사
hit=(rets0.abs()>0.5)&rets0.notna()
by_year=hit.sum(axis=1).groupby(price.index.year).sum()
print("\n[필터 감사] |r|>0.5 적중: 총",int(hit.values.sum()),"건 | 상위 연도:",
      dict(by_year.sort_values(ascending=False).head(5)))
res.round(4).to_csv(ROOT/"output/tables/p13_delisting_filter.csv")
by_year.to_csv(ROOT/"output/tables/p13_filter_hits_by_year.csv")
assert np.isfinite(res["raw_sharpe"]).all()
print("[GATES] 전 변형 산출 ✓")
