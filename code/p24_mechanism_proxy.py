"""
P24 (전략 M3) — 메커니즘 대리검정: "RV가 꼬리의 나쁜 조기경보인 정도"의 시장 간 비교.

[사전지정 — 실행 전 커밋]
- 가설: ML(시장상태 조건부) 꼬리 이득은 "RV 지속성만으론 꼬리를 못 잡는 정도"가 큰 시장에서 발생.
  한국 decile이 타 시장 대비 이 열화가 크면, E21 국제 미재현이 반증이 아니라 메커니즘 검증 증거가 됨.
- 대리지표 3종 (월별 시계열, 시장별):
  P1 vol-of-vol: std(Δlog RV) — 위험 수준 자체의 불안정성
  P2 곰장 비선형성: r²_{t+1} = a + b·RV_t + c·(RV_t×bear_t) 에서 |t(c)| — RW126(선형 지속)이 못 잡고
     상태조건 모형(HARX)만 잡는 성분의 크기
  P3 꼬리 기습률: 최악 5% 월 중 직전 σ̂(RW126)가 표본 중앙값 이하였던 비율 — RV 조기경보 실패율
- 대상 8계열: 한국 decile · 한국 2×3(내부 대조) · US decile · 일본/유럽/북미/아태 2×3 · US 2×3(팩터)
- 성공기준(사전지정): 한국 decile이 3지표 중 **2개 이상에서 8계열 중 1위**(또는 뚜렷한 상위) →
  메커니즘 대리 확인. 아니면 docs/08 중단조건 발동: 메커니즘 승격 포기, '방향적' 서술 유지.
- 원장 E24. 논문 반영 없음(§6-2).
"""
import numpy as np, pandas as pd
from scipy import stats
D="/mnt/20t/tmp/claude-1001/-mnt-20t-----/da2b9872-77c2-46d2-a247-cd601338a0be/scratchpad/intl/"
ROOT="/mnt/20t/졸업논문/"
Pm=lambda i:i.to_period("M")

def load_kf_wml(fp):
    L=open(fp,encoding="latin1").read().splitlines()
    i0=next(i for i,l in enumerate(L) if l.strip().startswith(",") and ("WML" in l or "Mom" in l))
    r=[]
    for l in L[i0+1:]:
        t=l.split(","); k=t[0].strip()
        if k.isdigit() and len(k)==8:
            try: r.append((k,float(t[1])))
            except: break
        elif r: break
    s=pd.Series(dict(r)); s.index=pd.to_datetime(s.index,format="%Y%m%d")
    return s.replace([-99.99,-999.0],np.nan).dropna()/100.0
def load_mkt_kf(fp,cols=4):
    L=open(fp,encoding="latin1").read().splitlines()
    i0=next(i for i,l in enumerate(L) if l.strip().startswith(",") and "Mkt-RF" in l)
    r=[]
    for l in L[i0+1:]:
        t=l.split(","); k=t[0].strip()
        if k.isdigit() and len(k)==8:
            try: r.append((k,float(t[1])+float(t[cols])))
            except: break
        elif r: break
    s=pd.Series(dict(r)); s.index=pd.to_datetime(s.index,format="%Y%m%d")
    return s.replace([-99.99,-999.0],np.nan).dropna()/100.0

kr_dec=pd.read_csv(ROOT+"data/processed/wml_daily.csv",parse_dates=["Date"]).set_index("Date")["wml"].dropna()
kr_23=pd.read_csv(ROOT+"data/processed/wml_2x3_daily.csv",parse_dates=[0],index_col=0).iloc[:,0].dropna()
kr_mkt=pd.read_csv(ROOT+"data/processed/ff_own_daily.csv",parse_dates=["Date"]).set_index("Date")["rmrf"].dropna()
us_dec=pd.read_csv("/mnt/20t/졸업논문/data/us/us_wml_daily.csv",parse_dates=[0],index_col=0).iloc[:,0].dropna()
us_mkt=load_mkt_kf(D+"F-F_Research_Data_Factors_daily.csv")
SERIES={
 "Korea decile":(kr_dec,kr_mkt), "Korea 2x3":(kr_23,kr_mkt),
 "US decile":(us_dec,us_mkt), "US 2x3(factor)":(load_kf_wml(D+"F-F_Momentum_Factor_daily.csv"),us_mkt),
 "Japan 2x3":(load_kf_wml(D+"Japan_MOM_Factor_Daily.csv"),load_mkt_kf(D+"Japan_3_Factors_Daily.csv")),
 "Europe 2x3":(load_kf_wml(D+"Europe_MOM_Factor_Daily.csv"),load_mkt_kf(D+"Europe_3_Factors_Daily.csv")),
 "NorthAm 2x3":(load_kf_wml(D+"North_America_MOM_Factor_Daily.csv"),load_mkt_kf(D+"North_America_3_Factors_Daily.csv")),
 "AsiaPac 2x3":(load_kf_wml(D+"Asia_Pacific_ex_Japan_MOM_Factor_Daily.csv"),load_mkt_kf(D+"Asia_Pacific_ex_Japan_3_Factors_Daily.csv")),
}
rows={}
for nm,(wd,md) in SERIES.items():
    g=wd.groupby(Pm(wd.index)); RV=g.apply(lambda x:(x**2).sum())
    wm=(1+wd).groupby(Pm(wd.index)).prod()-1
    mm=(1+md).groupby(Pm(md.index)).prod()-1
    bear=(mm.rolling(24).sum()<0).astype(float)
    df=pd.concat([RV.rename("rv"),wm.rename("r"),bear.rename("bear")],axis=1).dropna()
    df["r2n"]=(df["r"]**2).shift(-1); df=df.dropna()
    # P1 vol-of-vol
    vv=np.log(df["rv"]).diff().std()
    # P2 곰장 비선형성 |t(c)|
    X=np.column_stack([np.ones(len(df)),df["rv"],df["rv"]*df["bear"]])
    beta,res_,rank,_=np.linalg.lstsq(X,df["r2n"],rcond=None)
    resid=df["r2n"].values-X@beta
    XtXi=np.linalg.inv(X.T@X); s2=(resid**2).mean(); se=np.sqrt(np.diag(XtXi)*s2*len(df)/(len(df)-3))
    tc=abs(beta[2]/se[2])
    # P3 꼬리 기습률: 최악 5% 월 직전 RV(σ̂ 대용=전월 RV)가 중앙값 이하 비율
    rv_lag=df["rv"].shift(1)
    worst=df["r"].nsmallest(max(3,int(0.05*len(df)))).index
    surprise=float((rv_lag.loc[worst]<=df["rv"].median()).mean())
    rows[nm]=dict(n=len(df),volofvol=vv,bear_nonlin_t=tc,tail_surprise=surprise)
    print(f"  {nm:>15}: vol-of-vol {vv:.3f} · |t(RV×bear)| {tc:.2f} · 기습률 {surprise:.2f} (n={len(df)})")
tab=pd.DataFrame(rows).T
ranks=tab[["volofvol","bear_nonlin_t","tail_surprise"]].rank(ascending=False)
tab["rank_sum"]=ranks.sum(axis=1)
kr_rank=ranks.loc["Korea decile"]
wins=int((kr_rank==1).sum())
print(f"\n[순위] Korea decile: vol-of-vol {int(kr_rank['volofvol'])}위 · 비선형성 {int(kr_rank['bear_nonlin_t'])}위 · 기습률 {int(kr_rank['tail_surprise'])}위 → 1위 {wins}개")
verdict="메커니즘 대리 확인 (한국 decile이 2+개 지표 1위)" if wins>=2 else "중단조건 발동: 메커니즘 승격 포기, '방향적' 서술 유지"
print(f"[사전지정 판정] {verdict}")
tab.round(4).to_csv(ROOT+"output/tables/p24_mechanism_proxy.csv")
print("[GATES] 8계열 산출 ✓ · 표 저장 ✓")
