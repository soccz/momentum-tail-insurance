"""
P11 — 심사자 공격 대응: 계량 수정 4종 (전부 기존 OOS 산출물로 계산).

(1) ΔES5 family-wise 검정: 11모델 동시 중심화 max-t (paired 블록부트스트랩) — 선택적 추론 방어
(2) Clark-West 중첩모형 보정 (fvar_126이 피처에 포함 → DM 부적합 지점 보완)
(3) Giacomini-White식 조건부 검정: QLIKE 손실차를 곰장 더미에 회귀 (설계 §1④ 약속 이행)
(4) 전이곡선 기울기의 조인트 부트스트랩 CI (9점 완전종속 문제 해결 — 점별 OLS p 철회)
※ 부트스트랩은 전부 paired(동일 월 인덱스를 양쪽에 적용).
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
bear=ds["bear"].reindex(fc.index).values
act=fc["actual_var"].values; n=len(fc)
MODELS=[c for c in fc.columns if c not in ("actual_var",)]
RET={m:W[m].values*wml for m in W.columns}
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
es5=lambda r: np.mean(np.sort(r[np.isfinite(r)])[:max(1,int(0.05*len(r)))])*100
def bidx():
    return np.concatenate([np.arange(s,s+12)%n for s in rng.integers(0,n,size=int(np.ceil(n/12)))])[:n]

# ---------- (1) ΔES5 family-wise (11모델 동시) ----------
print("===== (1) ΔES5 family-wise max-t (paired, B=3000) =====")
cand=[m for m in W.columns if m!="RW126"]
d_obs=np.array([es5(RET[m])-es5(RET["RW126"]) for m in cand])
D=np.zeros((3000,len(cand)))
for b in range(3000):
    ix=bidx()
    base=es5(RET["RW126"][ix])
    D[b]=[es5(RET[m][ix])-base for m in cand]
Dc=D-D.mean(0)                                     # 중심화(귀무)
maxnull=Dc.max(1)
fw_p=float((maxnull>=d_obs.max()).mean())
best=cand[int(np.argmax(d_obs))]
indiv={m:float((D[:,i]<=0).mean()) for i,m in enumerate(cand)}
print(f"  최대 개선 {best} ΔES5={d_obs.max():+.2f}%p → family-wise p={fw_p:.4f}")
print(f"  개별 P(≤0): ENS={indiv.get('ENS',np.nan):.3f} · HARX={indiv.get('HARX',np.nan):.3f} · Ridge={indiv.get('Ridge',np.nan):.3f}")

# ---------- (2) Clark-West (MSPE-adj, 분산공간) ----------
print("\n===== (2) Clark-West vs RW126 (중첩모형 보정) =====")
cw_rows={}
e0=act-fc["RW126"].values
for m in ["HAR","HARX","Ridge","Lasso","RF","HGB","ENS","EWMA"]:
    e1=act-fc[m].values
    adj=e0**2-(e1**2-(fc["RW126"].values-fc[m].values)**2)
    ok=np.isfinite(adj); a=adj[ok]
    t=a.mean()/(a.std()/np.sqrt(len(a)))
    cw_rows[m]=dict(CW_t=t,p_1side=1-stats.norm.cdf(t))
    print(f"  {m:>6}: CW-t={t:+.2f}  p(우월)={1-stats.norm.cdf(t):.4f}")
cw=pd.DataFrame(cw_rows).T

# ---------- (3) GW식 조건부 (QLIKE 손실차 ~ 곰장) ----------
print("\n===== (3) 조건부 예측력 (손실차 = a + b·bear, HAC-3) =====")
ql=lambda f: act/f-np.log(act/f)-1
gw_rows={}
for m in ["HARX","Ridge","ENS"]:
    d=ql(fc["RW126"].values)-ql(fc[m].values)      # >0 = m 우월
    X=np.column_stack([np.ones(n),bear])
    beta=np.linalg.lstsq(X,d,rcond=None)[0]
    resid=d-X@beta
    # HAC(3) 분산
    S=np.zeros((2,2))
    for l in range(4):
        w=1-l/4
        for t in range(l,n):
            u=resid[t]*X[t]; v=resid[t-l]*X[t-l]
            S+=w*(np.outer(u,v)+(np.outer(v,u) if l else 0))
    XtX_inv=np.linalg.inv(X.T@X); V=XtX_inv@S@XtX_inv
    t_a,t_b=beta/np.sqrt(np.diag(V))
    gw_rows[m]=dict(a_무조건=beta[0],t_a=t_a,b_곰장추가=beta[1],t_b=t_b)
    print(f"  {m:>6}: 무조건 우위 a={beta[0]:+.4f}(t={t_a:+.2f}) · 곰장 추가 b={beta[1]:+.4f}(t={t_b:+.2f})")
gw=pd.DataFrame(gw_rows).T

# ---------- (4) 전이곡선 기울기 조인트 부트스트랩 ----------
print("\n===== (4) 전이곡선 기울기 — 조인트 부트스트랩 CI (B=2000) =====")
core=[m for m in ["RW126","EWMA","HAR","HARX","Ridge","Lasso","RF","HGB","ENS"]]
def slope_of(ix):
    a=act[ix]; sse_b=((a-fc["ExpandingMean"].values[ix])**2).sum()
    xs,ys=[],[]
    for m in core:
        f=fc[m].values[ix]
        xs.append((1-((a-f)**2).sum()/sse_b)*100)
        ys.append(sr(RET[m][ix]))
    return stats.linregress(xs,ys).slope
obs=slope_of(np.arange(n))
sl=np.array([slope_of(bidx()) for _ in range(2000)])
lo,hi=np.percentile(sl,[2.5,97.5])
print(f"  기울기 점추정 {obs:+.4f}/%p · 부트스트랩 95% CI [{lo:+.4f},{hi:+.4f}] · 0 포함={'예' if lo<=0<=hi else '아니오'}")
print(f"  |기울기|>0.005/%p 확률 = {float((np.abs(sl)>0.005).mean()):.3f} (경제적 유의 경계)")

pd.DataFrame({"metric":["fw_p_ES5","best","slope","slope_lo","slope_hi"],
              "value":[fw_p,best,obs,lo,hi]}).to_csv(ROOT/"output/tables/p11_fw_and_slope.csv",index=False)
cw.round(4).to_csv(ROOT/"output/tables/p11_clark_west.csv")
gw.round(4).to_csv(ROOT/"output/tables/p11_gw_conditional.csv")
print("\n[GATES] 표 3종 저장 ✓")
