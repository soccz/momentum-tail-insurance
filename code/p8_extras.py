"""
P8 — 추가 실험 묶음 (기존 산출물만 사용, 패널 불필요).

(a) RV 창 길이 스윕: 21/63/126/252일 롤링 스케일 — BS15의 "6개월" 선택 민감도 (논문 각주14 검증)
(b) 크래시 분류기 프로토타입: P(WML_{t+1}<-10%)를 로짓/HGB로 → AUC + 오버레이 전략 (라벨 희소성 정직 보고)
(c) 경제적 효용: 평균-분산 확실성등가 CE=μ−(γ/2)σ² (γ=5), 연율 %
(d) 누적곡선 그림 (발표 자산): RAW vs BS15 vs HARX vs Ridge-semi
(e) clean 표본(2005+) 재검: 결과가 1990년대 데이터 품질에 의존하지 않는가
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_k=[f.fname for f in fm.fontManager.ttflist if 'Nanum' in f.name or 'Noto Sans CJK' in f.name]
if _k: plt.rcParams['font.family']=fm.FontProperties(fname=_k[0]).get_name()
plt.rcParams['axes.unicode_minus']=False

SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path("/mnt/20t/졸업논문")
ds=pd.read_csv(ROOT/"data/processed/ml_dataset.csv",index_col=0); ds.index=pd.PeriodIndex(ds.index,freq="M"); ds=ds.dropna()
fc=pd.read_csv(ROOT/"data/processed/ml_forecasts.csv",index_col=0); fc.index=pd.PeriodIndex(fc.index,freq="M")
W=pd.read_csv(ROOT/"data/processed/p3_weights.csv",index_col=0); W.index=pd.PeriodIndex(W.index,freq="M")
FEAT=[c for c in ds.columns if not c.startswith("tgt_")]
X_all=ds[FEAT].values; wml=ds["tgt_wml_next"].values
n=len(ds); FT,RT=120,12; oos=np.arange(FT,n); TGT=0.12/np.sqrt(12)
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
worst=lambda r: np.nanmin(r)*100

def perf_row(r):
    r=r[np.isfinite(r)]
    return dict(sharpe=sr(r), worst=worst(r), skew=stats.skew(r),
                CE5=(np.nanmean(r)-2.5*np.nanvar(r))*12*100)   # γ=5 확실성등가 (연 %)

# ---------- (a) RV 창 길이 스윕 ----------
print("===== (a) RV 창 길이 (BS15 '6개월' 민감도) =====")
win_rows={}
for lab in ["1m","3m","6m","12m"]:
    fvar=(ds[f"vol_{lab}"].values/100)**2/12
    win_rows[f"RW-{lab}"]=perf_row((TGT/np.sqrt(fvar)*wml)[oos])
wins=pd.DataFrame(win_rows).T
print(wins.round(3).to_string())

# ---------- (b) 크래시 분류기 ----------
print("\n===== (b) 크래시 분류기 P(WML<-10%) — 라벨 희소성 정직 보고 =====")
y_cr=(wml<-0.10).astype(int)
print(f"라벨: 전체 {y_cr.sum()}건 / OOS {y_cr[oos].sum()}건 (전체 {n}개월)")
proba={"Logit":np.full(n,np.nan),"HGBc":np.full(n,np.nan)}
for t0 in range(FT,n,RT):
    tr,te=np.arange(t0),np.arange(t0,min(t0+RT,n))
    if y_cr[tr].sum()<5: continue
    sc=StandardScaler().fit(X_all[tr]); Z=sc.transform(X_all)
    proba["Logit"][te]=LogisticRegression(max_iter=2000,C=0.5).fit(Z[tr],y_cr[tr]).predict_proba(Z[te])[:,1]
    proba["HGBc"][te]=HistGradientBoostingClassifier(random_state=SEED,max_depth=3,learning_rate=0.05)\
                        .fit(X_all[tr],y_cr[tr]).predict_proba(X_all[te])[:,1]
cls_rows={}
for k,p in proba.items():
    ok=np.isfinite(p[oos])
    auc=roc_auc_score(y_cr[oos][ok],p[oos][ok])
    Lov=W["RW126"].reindex(ds.index).values[oos]*np.clip(1-p[oos],0.2,1.0)
    r_ov=Lov*wml[oos]
    cls_rows[k]=dict(AUC=auc, **perf_row(r_ov))
cls_rows["기준 RW126"]=dict(AUC=np.nan, **perf_row(W["RW126"].reindex(ds.index).values[oos]*wml[oos]))
cls=pd.DataFrame(cls_rows).T
print(cls.round(3).to_string())

# ---------- (c) 효용 이득 표 (γ=5) ----------
print("\n===== (c) 확실성등가 CE (γ=5, 연 %) — 경제적 유의성 =====")
strat={"RAW":wml[oos]}
for m in ["RW126","HARX","Ridge","ENS"]:
    strat[m]=W[m].reindex(ds.index).values[oos]*wml[oos]
ce=pd.DataFrame({k:perf_row(v) for k,v in strat.items()}).T
ce["ΔCE_vs_RAW"]=ce["CE5"]-ce.loc["RAW","CE5"]
print(ce.round(2).to_string())

# ---------- (d) 누적곡선 그림 ----------
idx=ds.index[oos].to_timestamp("M")
fig,ax=plt.subplots(figsize=(10,5))
for nm,r,c,dsh in [("RAW 무관리",strat["RAW"],"#96887a",":"),
                   ("BS15 (RW126)",strat["RW126"],"#2f6d8f","-"),
                   ("HARX",strat["HARX"],"#c4724e","-"),
                   ("Ridge",strat["Ridge"],"#c9a24a","--")]:
    ax.plot(idx,(1+pd.Series(r,index=idx).fillna(0)).cumprod(),label=f"{nm} (샤프 {sr(r):.2f})",color=c,ls=dsh,lw=1.6)
ax.set_yscale("log"); ax.legend(frameon=False); ax.set_ylabel("누적 ($1, 로그)")
ax.set_title("OOS 누적성과 — 위험관리·ML의 가치 (2001-12~2026-03)",loc="left")
fig.tight_layout(); fig.savefig(ROOT/"output/figures/p8_cumulative.png",dpi=130,bbox_inches="tight")

# ---------- (e) clean 표본(2005+) 재검 ----------
print("\n===== (e) clean 표본(2005+) — 90년대 의존성 점검 =====")
wdc=pd.read_csv(ROOT/"data/processed/wml_daily_clean.csv",parse_dates=["Date"]).set_index("Date")
wmc=pd.read_csv(ROOT/"data/processed/wml_monthly_clean.csv",parse_dates=["month"]).set_index("month")
ffc=pd.read_csv(ROOT/"data/processed/ff_own_daily_clean.csv",parse_dates=["Date"]).set_index("Date")
Pm=lambda i:i.to_period("M")
r=wdc["wml"]; rm=ffc["rmrf"].reindex(r.index); sq=r**2
f=pd.DataFrame(index=r.index)
for lab,w_ in [("1m",21),("3m",63),("6m",126),("12m",252)]:
    f[f"vol_{lab}"]=np.sqrt(252/w_*sq.rolling(w_,min_periods=w_).sum())*100
f["semineg_6m"]=np.sqrt(252/126*(r.where(r<0)**2).fillna(0).rolling(126).sum())*100
f["mkt_vol_1m"]=np.sqrt(252/21*(rm**2).rolling(21).sum())*100
mi=(1+rm.fillna(0)).cumprod()
f["bear"]=(mi.pct_change(504)<0).astype(float); f["rebound"]=((f["bear"]>0)&(mi.pct_change(21)>0)).astype(float)
f["fvar_126"]=21*sq.rolling(126,min_periods=126).mean()
fme=f.groupby(Pm(f.index)).last()
tgtv=sq.groupby(Pm(sq.index)).sum().shift(-1)
wnext=wmc["wml"].copy(); wnext.index=Pm(wnext.index); wnext=wnext.shift(-1)
dfc=pd.concat([fme,tgtv.rename("var_next"),wnext.rename("wml_next")],axis=1).dropna()
Xc=dfc[[c for c in fme.columns]].values; yv=np.log(dfc["var_next"].values); wn=dfc["wml_next"].values
nc=len(dfc); FTc=96; oc=np.arange(FTc,nc)
pred=np.full(nc,np.nan)
for t0 in range(FTc,nc,12):
    tr,te=np.arange(t0),np.arange(t0,min(t0+12,nc))
    scl=StandardScaler().fit(Xc[tr]); Z=scl.transform(Xc)
    m=RidgeCV(alphas=np.logspace(-3,3,13)).fit(Z[tr],yv[tr])
    resid=yv[tr]-m.predict(Z[tr])
    pred[te]=np.clip(np.exp(m.predict(Z[te])+resid.var()/2),
                     dfc["var_next"].values[tr].min()*0.5,dfc["var_next"].values[tr].max()*2)
rw_c=TGT/np.sqrt(dfc["fvar_126"].values[oc])*wn[oc]
ml_c=TGT/np.sqrt(pred[oc])*wn[oc]
raw_c=wn[oc]
clean=pd.DataFrame({"RAW":perf_row(raw_c),"RW126":perf_row(rw_c),"Ridge":perf_row(ml_c)}).T
print(f"clean n={nc} (OOS {len(oc)}개월, {dfc.index[oc[0]]}~{dfc.index[oc[-1]]})")
print(clean.round(3).to_string())

# ---------- 저장 + 게이트 ----------
wins.round(4).to_csv(ROOT/"output/tables/p8_windows.csv")
cls.round(4).to_csv(ROOT/"output/tables/p8_crash_classifier.csv")
ce.round(4).to_csv(ROOT/"output/tables/p8_utility.csv")
clean.round(4).to_csv(ROOT/"output/tables/p8_clean_sample.csv")
assert np.isfinite(wins["sharpe"]).all() and np.isfinite(clean["sharpe"]).all()
print("\n[GATES] 표 4종+그림 1종 저장 ✓")
