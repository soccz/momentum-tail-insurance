"""
P10 — 심사자 잔소리 선제 처리 5종.

(a) 곰장/비곰장 조건부 성과: 이득이 어느 국면에서 오는가 (타이밍 해부의 공식 표)
(b) Jobson-Korkie(Memmel 보정) 샤프 차이 검정 — 부트스트랩의 모수적 보완
(c) 1998-10 제외 시 full-sample 재현 통계 — 단일 관측 지배 점검 (C2 방어)
(d) ΔCE(γ=5) 부트스트랩 CI — 경제적 이득의 공식 추론
(e) 분기 리밸런스(스케일 3개월 고정) — 저빈도 실행 변형
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path("/mnt/20t/졸업논문")
fc=pd.read_csv(ROOT/"data/processed/ml_forecasts.csv",index_col=0); fc.index=pd.PeriodIndex(fc.index,freq="M")
W=pd.read_csv(ROOT/"data/processed/p3_weights.csv",index_col=0); W.index=pd.PeriodIndex(W.index,freq="M")
ds=pd.read_csv(ROOT/"data/processed/ml_dataset.csv",index_col=0); ds.index=pd.PeriodIndex(ds.index,freq="M")
wml=ds["tgt_wml_next"].reindex(fc.index).values
bear=ds["bear"].reindex(fc.index).values.astype(bool)
n=len(fc); RET={m:W[m].values*wml for m in W.columns}
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
ce5=lambda r: (np.nanmean(r)-2.5*np.nanvar(r))*12*100

# ---------- (a) 곰장 조건부 ----------
print(f"===== (a) 곰장({bear.sum()}개월)/비곰장({(~bear).sum()}) 조건부 — 연율 평균수익% =====")
rows={}
for m in ["RW126","HARX","Ridge"]:
    rows[m]={"곰장 평균%":np.nanmean(RET[m][bear])*1200,"비곰장 평균%":np.nanmean(RET[m][~bear])*1200,
             "곰장 샤프":sr(RET[m][bear]),"비곰장 샤프":sr(RET[m][~bear])}
rows["RAW"]={"곰장 평균%":np.nanmean(wml[bear])*1200,"비곰장 평균%":np.nanmean(wml[~bear])*1200,
             "곰장 샤프":sr(wml[bear]),"비곰장 샤프":sr(wml[~bear])}
bearT=pd.DataFrame(rows).T
print(bearT.round(2).to_string())

# ---------- (b) Jobson-Korkie–Memmel ----------
print("\n===== (b) JK-Memmel 샤프 차이 검정 vs RW126 =====")
def jkm(r1,r0):
    ok=np.isfinite(r1)&np.isfinite(r0); a,b=r1[ok],r0[ok]; T=len(a)
    s1,s0=a.std(),b.std(); m1,m0=a.mean(),b.mean(); rho=np.corrcoef(a,b)[0,1]
    sh1,sh0=m1/s1,m0/s0
    v=(1/T)*(2-2*rho+0.5*(sh1**2+sh0**2-2*sh1*sh0*rho**2))
    z=(sh1-sh0)/np.sqrt(v); return z,2*(1-stats.norm.cdf(abs(z)))
jk_rows={}
for m in ["HARX","Ridge","ENS","EWMA","HAR"]:
    z,p=jkm(RET[m],RET["RW126"]); jk_rows[m]=dict(z=z,p=p)
    print(f"  {m:>6}: z={z:+.2f}  p={p:.3f}")
jk=pd.DataFrame(jk_rows).T

# ---------- (c) 1998-10 제외 full-sample ----------
print("\n===== (c) 1998-10 제외 시 full-sample WML 통계 (C2 단일관측 점검) =====")
wm_full=pd.read_csv(ROOT/"data/processed/wml_monthly.csv",parse_dates=["month"]).set_index("month")["wml"]
def full_stats(s):
    return dict(sharpe=np.sqrt(12)*s.mean()/s.std(),skew=stats.skew(s,bias=False),
                kurt=stats.kurtosis(s,bias=False),worst=s.min()*100)
a=full_stats(wm_full); b=full_stats(wm_full.drop(pd.Timestamp("1998-10-31")))
lwo=pd.DataFrame({"전체":a,"1998-10 제외":b}).T
print(lwo.round(2).to_string())
print("→ 왜도·첨도는 크게 완화되지만 부호 유지 — '크래시 위험 실재' 주장은 유지하되 단일사건 의존을 본문에 명시")

# ---------- (d) ΔCE 부트스트랩 ----------
print("\n===== (d) ΔCE(γ=5, 연%p) 부트스트랩 vs RW126 (B=5000) =====")
dce_rows={}
for m in ["HARX","Ridge","ENS"]:
    d=[]
    for _ in range(5000):
        ix=np.concatenate([np.arange(s,s+12)%n for s in rng.integers(0,n,size=int(np.ceil(n/12)))])[:n]
        d.append(ce5(RET[m][ix])-ce5(RET["RW126"][ix]))
    d=np.array(d); dce_rows[m]=dict(dCE=ce5(RET[m])-ce5(RET["RW126"]),ci_lo=np.percentile(d,2.5),
                                    ci_hi=np.percentile(d,97.5),P_le0=float((d<=0).mean()))
    r=dce_rows[m]; print(f"  {m:>6}: ΔCE={r['dCE']:+.2f}%p  CI[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]  P(≤0)={r['P_le0']:.3f}")
dce=pd.DataFrame(dce_rows).T

# ---------- (e) 분기 리밸런스 ----------
print("\n===== (e) 스케일 분기 고정 (실행 저빈도 변형) =====")
q_rows={}
for m in ["RW126","HARX","Ridge"]:
    Lq=W[m].values.copy()
    for i in range(n):
        if i%3: Lq[i]=Lq[i-1]                      # 분기 첫 달 값 유지
    r=Lq*wml; dL=np.abs(np.diff(Lq,prepend=Lq[0]))
    q_rows[m]=dict(sharpe_월=sr(RET[m]),sharpe_분기=sr(r),
                   회전ΔL_월=np.abs(np.diff(W[m].values,prepend=W[m].values[0])).mean(),
                   회전ΔL_분기=dL.mean())
qtab=pd.DataFrame(q_rows).T
print(qtab.round(3).to_string())

for df,f in [(bearT,"p10_bear_conditional"),(jk,"p10_jkm_test"),(lwo,"p10_leave_199810_out"),
             (dce,"p10_dce_bootstrap"),(qtab,"p10_quarterly")]:
    df.round(4).to_csv(ROOT/f"output/tables/{f}.csv")
print("\n[GATES] 표 5종 저장 ✓")
