"""P14 — 잔여 민감도 quick 배터리: 부트스트랩 블록길이·OOS 시작점·효용 γ·DSR 실효 N."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path("/mnt/20t/졸업논문")
fc=pd.read_csv(ROOT/"data/processed/ml_forecasts.csv",index_col=0); fc.index=pd.PeriodIndex(fc.index,freq="M")
W=pd.read_csv(ROOT/"data/processed/p3_weights.csv",index_col=0); W.index=pd.PeriodIndex(W.index,freq="M")
ds=pd.read_csv(ROOT/"data/processed/ml_dataset.csv",index_col=0); ds.index=pd.PeriodIndex(ds.index,freq="M")
wml=ds["tgt_wml_next"].reindex(fc.index).values; n=len(fc)
RET={m:W[m].values*wml for m in W.columns}
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
es5=lambda r: np.mean(np.sort(r[np.isfinite(r)])[:max(1,int(.05*len(r)))])*100

# (1) 블록길이 민감도: ENS ΔES5 paired P
print("===== (1) ENS ΔES5 부트스트랩 — 블록길이 {6,12,24} =====")
for blk in (6,12,24):
    d=[]
    for _ in range(3000):
        ix=np.concatenate([np.arange(s,s+blk)%n for s in rng.integers(0,n,size=int(np.ceil(n/blk)))])[:n]
        d.append(es5(RET["ENS"][ix])-es5(RET["RW126"][ix]))
    d=np.array(d); print(f"  blk={blk:>2}: ΔES5={es5(RET['ENS'])-es5(RET['RW126']):+.2f}%p  P(≤0)={float((d<=0).mean()):.3f}")

# (2) OOS 시작점: ml_dataset 재적합 없이 평가창만 변경 (예측은 동일 프로토콜 산출물)
print("\n===== (2) 평가창 시작 이동 (예측 고정, 평가만) =====")
for cut in ("2001-12","2004-01","2006-01"):
    m=fc.index>=cut
    print(f"  {cut}+ (n={m.sum()}): RW126 {sr(RET['RW126'][m]):.3f} · ENS {sr(RET['ENS'][m]):.3f} · HARX {sr(RET['HARX'][m]):.3f} · ΔES5(ENS) {es5(RET['ENS'][m])-es5(RET['RW126'][m]):+.2f}%p")

# (3) 효용 γ 스윕
print("\n===== (3) ΔCE(ENS vs RW126) — γ ∈ {2,5,10,20} =====")
for g in (2,5,10,20):
    ce=lambda r: (np.nanmean(r)-g/2*np.nanvar(r))*1200
    print(f"  γ={g:>2}: 관리(RW126 vs RAW) {ce(RET['RW126'])-ce(wml):+.1f}%p/yr · ML증분(ENS vs RW126) {ce(RET['ENS'])-ce(RET['RW126']):+.2f}%p/yr")

# (4) DSR 실효 시행수
print("\n===== (4) Deflated Sharpe — 실효 시행수 N ∈ {12, 40} =====")
srs=np.array([np.nanmean(RET[m])/np.nanstd(RET[m]) for m in W.columns])
best=W.columns[int(np.nanargmax(srs))]; SR=srs.max(); V=srs.var(); emc=0.5772156649
r=RET[best][np.isfinite(RET[best])]; g3,g4=stats.skew(r),stats.kurtosis(r,fisher=False)
for N in (12,40):
    sr0=np.sqrt(V)*((1-emc)*stats.norm.ppf(1-1/N)+emc*stats.norm.ppf(1-1/(N*np.e)))
    dsr=stats.norm.cdf(((SR-sr0)*np.sqrt(len(r)-1))/np.sqrt(1-g3*SR+(g4-1)/4*SR**2))
    print(f"  N={N:>2}: E[maxSR|noise]={sr0:.4f} → DSR({best})={dsr:.3f}")
print("\n[GATES] 완료 ✓")
