"""
P26 — A+ 마감 라운드(E33)의 심사 패치 3종.

(1) CW 보정 손실차 Reality Check — "중첩과 다중성을 동시에 처리한 검정이 없다"(2라운드 major) 봉합.
    p11의 Clark–West MSPE-adj 손실차(분산공간)를 11개 비교 전체에 만들고, 스튜던트화 최대통계량의
    원형 블록 부트스트랩(blk=12, B=5,000, seed 42, 재중심화)으로 FWER p를 계산 — Hansen SPA형.
(2) CRRA CE(γ=10) 잭나이프 — "최악 2~3개월 빼면 +1.75가 얼마나 남는가"(방어 렌즈) 봉합.
    무관리 WML 기준 최악 k개월(k=0..3)을 공통 제거 후 ΔCE(ENS−RW126) 재계산.
(3) MV-CE CSV 보존 — 표 6 MV열의 산출물 부재(수치 렌즈) 봉합. p14의 ce()를 문자 동일 재현해 저장.

산출물: output/tables/p26_cw_rc.csv · p26_ce_jackknife.csv · p14_mv_ce.csv (원장 E33).
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42; rng = np.random.default_rng(SEED)
ROOT = Path("/mnt/20t/졸업논문")
fc = pd.read_csv(ROOT / "data/processed/ml_forecasts.csv", index_col=0); fc.index = pd.PeriodIndex(fc.index, freq="M")
W = pd.read_csv(ROOT / "data/processed/p3_weights.csv", index_col=0); W.index = pd.PeriodIndex(W.index, freq="M")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0); ds.index = pd.PeriodIndex(ds.index, freq="M")
wml = ds["tgt_wml_next"].reindex(fc.index).values
act = fc["actual_var"].values
n = len(fc)
MODELS = [c for c in fc.columns if c not in ("actual_var", "RW126")]  # 11개 비교 (p9 RC와 동일 가족)

# ---------- (1) CW-조정 손실차의 스튜던트화 Reality Check ----------
e0 = act - fc["RW126"].values
ADJ = {}
for m in MODELS:
    e1 = act - fc[m].values
    ADJ[m] = e0**2 - (e1**2 - (fc["RW126"].values - fc[m].values)**2)   # p11과 동일 (CW MSPE-adj)
A = pd.DataFrame(ADJ).values                                            # (n, 11)
tstat = lambda X: X.mean(0) / (X.std(0, ddof=1) / np.sqrt(X.shape[0]))
t_obs = tstat(A); T_obs = t_obs.max()
B = 5000; blk = 12
Ac = A - A.mean(0)                                                      # 재중심화 (White 2000)
Tb = np.empty(B)
for b in range(B):
    ix = np.concatenate([np.arange(s, s + blk) % n for s in rng.integers(0, n, size=int(np.ceil(n / blk)))])[:n]
    Tb[b] = tstat(Ac[ix]).max()
p_rc = float((Tb >= T_obs).mean())
best = MODELS[int(t_obs.argmax())]
pd.DataFrame({"model": MODELS, "cw_t": t_obs}).set_index("model").round(4).assign(
    max_model=best, max_t=round(T_obs, 4), rc_p=p_rc).to_csv(ROOT / "output/tables/p26_cw_rc.csv")
print(f"[1] CW-조정 스튜던트화 RC: max-t={T_obs:.2f} ({best}) · FWER p={p_rc:.4f} (11개 비교, blk=12, B=5000)")

# ---------- (2) CRRA CE(γ=10) 잭나이프 ----------
g = 10
def ce_crra(r):
    r = r[np.isfinite(r)]
    m = np.mean((1 + r) ** (1 - g)) ** (1 / (1 - g)) - 1
    return ((1 + m) ** 12 - 1) * 100
RET = {m: W[m].values * wml for m in W.columns}
order = np.argsort(wml)                                                 # 무관리 WML 기준 최악월
rows = {}
for k in range(4):
    keep = np.ones(n, bool); keep[order[:k]] = False
    rows[k] = dict(dCE_ML=ce_crra(RET["ENS"][keep]) - ce_crra(RET["RW126"][keep]),
                   dropped="" if k == 0 else ",".join(str(fc.index[i]) for i in order[:k]))
jk = pd.DataFrame(rows).T
jk.to_csv(ROOT / "output/tables/p26_ce_jackknife.csv")
print("[2] CRRA CE(γ=10) ML증분 잭나이프 (무관리 최악 k개월 제거):")
for k in range(4):
    print(f"    k={k}: ΔCE={rows[k]['dCE_ML']:+.2f}%p/yr  {rows[k]['dropped']}")

# ---------- (3) MV-CE CSV (p14 (3)과 문자 동일) ----------
mv = {}
for gg in (2, 5, 10, 20):
    ce = lambda r: (np.nanmean(r) - gg / 2 * np.nanvar(r)) * 1200
    mv[gg] = dict(MV_mgmt=ce(RET["RW126"]) - ce(wml), MV_ML=ce(RET["ENS"]) - ce(RET["RW126"]))
mvdf = pd.DataFrame(mv).T.round(4)
mvdf.to_csv(ROOT / "output/tables/p14_mv_ce.csv")
print("[3] MV-CE 저장 (표 6 대조):")
print(mvdf.round(2).to_string())

# 게이트: 표 6 고정값 재현
assert abs(mv[5]["MV_mgmt"] - 12.5) < 0.15 and abs(mv[10]["MV_ML"] - 1.12) < 0.02, "표 6 MV 재현 실패"
print("\n[GATES] 표 6 MV열 재현 ✓ · 산출물 3종 저장 ✓")
