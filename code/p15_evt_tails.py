"""
P15 — EVT 극값 꼬리 검증 + 경제적 유의성 보강 (업그레이드 라운드).

동기: 헤드라인 ΔES5는 경험적 꼬리(292개월×5%≈14관측)에 얹혀 있음 → 극값이론(POT/GPD)으로
      모수적 꼬리를 추정해 관측 희소성 한계를 돌파. + "경제적 유의성 얇다" 공격 방어 3종.

(1) POT/GPD: 손실 초과분에 GPD 적합 → 모수적 ES5%·ES1%, 임계 80/85/90% 민감도
(2) ΔES(GPD) paired 블록부트스트랩 (B=2000, 블록12, seed42) — 경험적 ES 결과와 교차 확인
(3) Hill 꼬리지수 (k=20)
(4) 경제적 유의성: 꼬리위험 단위당 수익(mean/|ES5|) · 롤링 60개월 승률 · 손익분기 비용(bp)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path("/mnt/20t/졸업논문")
W=pd.read_csv(ROOT/"data/processed/p3_weights.csv",index_col=0); W.index=pd.PeriodIndex(W.index,freq="M")
ds=pd.read_csv(ROOT/"data/processed/ml_dataset.csv",index_col=0); ds.index=pd.PeriodIndex(ds.index,freq="M")
wml=ds["tgt_wml_next"].reindex(W.index).values
RET={m:W[m].values*wml for m in W.columns}
n=len(W); MODELS=["RW126","ENS","HARX","Ridge"]
sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
es_emp=lambda r,q=0.05: np.mean(np.sort(r[np.isfinite(r)])[:max(1,int(q*len(r)))])*100
def bidx():
    return np.concatenate([np.arange(s,s+12)%n for s in rng.integers(0,n,size=int(np.ceil(n/12)))])[:n]

# ---------- (1) POT/GPD 모수적 꼬리 ----------
def gpd_es(r, uq=0.90, p=0.05):
    """손실 ℓ=-r 의 uq-분위 임계 초과분에 GPD 적합 → 수익 공간 ES_p (%). ξ≥0.95면 NaN."""
    l=-r[np.isfinite(r)]; u=np.quantile(l,uq); exc=l[l>u]-u
    if len(exc)<10: return np.nan,np.nan,len(exc)
    xi,_,sg=stats.genpareto.fit(exc,floc=0)
    if xi>=0.95: return np.nan,xi,len(exc)
    zeta=len(exc)/len(l)
    var=u+(sg/xi)*((p/zeta)**(-xi)-1) if abs(xi)>1e-9 else u+sg*np.log(zeta/p)
    es=(var+sg-xi*u)/(1-xi) if abs(xi)>1e-9 else var+sg   # McNeil–Frey–Embrechts: ES=(VaR+σ−ξu)/(1−ξ)
    return -es*100, xi, len(exc)

print("===== (1) GPD 모수적 꼬리 (임계 민감도) =====")
rows=[]
for m in MODELS:
    row={"model":m,"ES5_emp":es_emp(RET[m]),"ES1_emp":es_emp(RET[m],0.01)}
    for uq in (0.80,0.85,0.90):
        e5,xi,ne=gpd_es(RET[m],uq,0.05); e1,_,_=gpd_es(RET[m],uq,0.01)
        row[f"ES5_gpd_u{int(uq*100)}"]=e5; row[f"ES1_gpd_u{int(uq*100)}"]=e1
        if uq==0.85: row["xi"]=xi; row["n_exc"]=ne
    rows.append(row); print(f"  {m:>6}: emp ES5={row['ES5_emp']:+.2f} | GPD(u85) ES5={row['ES5_gpd_u85']:+.2f} "
                            f"ES1={row['ES1_gpd_u85']:+.2f} ξ={row['xi']:+.3f} (초과 {row['n_exc']}개)")
gpd_tab=pd.DataFrame(rows).set_index("model")

# ---------- (2) ΔES(GPD) paired 블록부트스트랩 ----------
print("\n===== (2) ΔES(GPD, u=85%) paired 부트스트랩 (B=2000) =====")
B=2000; boot={m:{"d5":[],"d1":[]} for m in MODELS if m!="RW126"}; degen=0
for b in range(B):
    ix=bidx()
    b5,_,_=gpd_es(RET["RW126"][ix],0.85,0.05); b1,_,_=gpd_es(RET["RW126"][ix],0.85,0.01)
    if not np.isfinite(b5): degen+=1; continue
    for m in boot:
        e5,_,_=gpd_es(RET[m][ix],0.85,0.05); e1,_,_=gpd_es(RET[m][ix],0.85,0.01)
        boot[m]["d5"].append(e5-b5 if np.isfinite(e5) else np.nan)
        boot[m]["d1"].append(e1-b1 if np.isfinite(e1) else np.nan)
ev_rows={}
for m,d in boot.items():
    d5=np.array(d["d5"]); d5=d5[np.isfinite(d5)]; d1=np.array(d["d1"]); d1=d1[np.isfinite(d1)]
    obs5=gpd_tab.loc[m,"ES5_gpd_u85"]-gpd_tab.loc["RW126","ES5_gpd_u85"]
    obs1=gpd_tab.loc[m,"ES1_gpd_u85"]-gpd_tab.loc["RW126","ES1_gpd_u85"]
    ev_rows[m]=dict(dES5_gpd=obs5,P_le0_ES5=float((d5<=0).mean()),
                    dES1_gpd=obs1,P_le0_ES1=float((d1<=0).mean()),n_boot=len(d5))
    print(f"  {m:>6}: ΔES5(GPD)={obs5:+.2f}%p P(≤0)={ev_rows[m]['P_le0_ES5']:.3f} | "
          f"ΔES1(GPD)={obs1:+.2f}%p P(≤0)={ev_rows[m]['P_le0_ES1']:.3f}")
print(f"  (퇴화 재표집 {degen}/{B})")
evt=pd.DataFrame(ev_rows).T

# ---------- (3) Hill 꼬리지수 ----------
print("\n===== (3) Hill 꼬리지수 (k=20) =====")
for m in MODELS:
    l=np.sort(-RET[m][np.isfinite(RET[m])])[::-1]; k=20
    H=np.mean(np.log(l[:k]/l[k])); gpd_tab.loc[m,"hill_alpha"]=1/H
    print(f"  {m:>6}: α={1/H:.2f}")

# ---------- (4) 경제적 유의성 3종 ----------
print("\n===== (4a) 꼬리위험 단위당 수익 (연수익%/|ES5%|) =====")
for m in MODELS:
    mu=np.nanmean(RET[m])*1200
    gpd_tab.loc[m,"mean_ann"]=mu
    gpd_tab.loc[m,"ret_per_tail"]=mu/abs(gpd_tab.loc[m,"ES5_emp"])
    print(f"  {m:>6}: {mu:+.2f}%/yr ÷ |{gpd_tab.loc[m,'ES5_emp']:.2f}| = {gpd_tab.loc[m,'ret_per_tail']:.3f}")

print("\n===== (4b) 롤링 60개월 승률 vs RW126 =====")
wins={}
for m in [x for x in MODELS if x!="RW126"]:
    ws,we=[],[]
    for s in range(0,n-60+1):
        sl=slice(s,s+60)
        ws.append(sr(RET[m][sl])>=sr(RET["RW126"][sl]))
        we.append(es_emp(RET[m][sl])>=es_emp(RET["RW126"][sl]))
    wins[m]=dict(sharpe_win=np.mean(ws),es5_win=np.mean(we),n_win=len(ws))
    print(f"  {m:>6}: 샤프 승률 {np.mean(ws):.1%} · ES5 승률 {np.mean(we):.1%} ({len(ws)}창)")

print("\n===== (4c) 손익분기 비용 (p7_full_cost 보간) =====")
fc=pd.read_csv(ROOT/"output/tables/p7_full_cost.csv",index_col=0); fc.columns=[int(c[:-2]) for c in fc.columns]
be={}
for m in [x for x in fc.index if x!="RW126"]:
    diff=fc.loc[m]-fc.loc["RW126"]; cs=diff.index.values.astype(float); dv=diff.values
    be[m]=np.nan
    for i in range(len(cs)-1):
        if dv[i]>0>=dv[i+1]:
            be[m]=cs[i]+(cs[i+1]-cs[i])*dv[i]/(dv[i]-dv[i+1]); break
    if np.isnan(be[m]) and (dv>0).all(): be[m]=np.inf
    print(f"  {m:>6}: 손익분기 ≈ {be[m]:.0f}bp" if np.isfinite(be[m]) else f"  {m:>6}: 전 구간 우위(>100bp)")

gpd_tab.round(4).to_csv(ROOT/"output/tables/p15_evt_gpd.csv")
evt.round(4).to_csv(ROOT/"output/tables/p15_evt_bootstrap.csv")
pd.DataFrame(wins).T.round(4).to_csv(ROOT/"output/tables/p15_rolling_winrate.csv")
pd.Series(be,name="breakeven_bp").round(1).to_csv(ROOT/"output/tables/p15_breakeven_cost.csv")

# 게이트: GPD ES5가 경험 ES5의 ±40% 이내 (모수화 정합) · 부트스트랩 유효표본 ≥ 1500
assert all(abs(gpd_tab.loc[m,"ES5_gpd_u85"]/gpd_tab.loc[m,"ES5_emp"]-1)<0.4 for m in MODELS), "GPD-경험 괴리"
assert all(v["n_boot"]>=1500 for v in ev_rows.values()), "부트스트랩 퇴화 과다"
print("\n[GATES] GPD≈경험 정합 ✓ · 부트스트랩 유효 ✓ · 표 4종 저장 ✓")
