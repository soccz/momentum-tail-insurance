"""
P5 — 강건성·비용·통계적 엄밀성 (정직성 프로토콜의 본체).

(a) 페어드 블록 부트스트랩(원형, 블록12, B=5000): ΔSharpe CI — HARX·ENS vs RW126
(b) Deflated Sharpe (Bailey & López de Prado 2014): 후보 12개 중 최고 샤프의 다중검정 보정
(c) 거래비용 스윕: 순수익 = L·WML − c·2·|ΔL| (레버리지 조정 왕복, 양 leg), c∈{0,10,30,60,100}bp
(d) 하위표본: OOS 전반/후반 ΔSharpe 유지 여부
(e) 타깃 불변성: σ_target 6/12/20% → 샤프 불변 확인 (Eq 6 성질)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = Path("/mnt/20t/졸업논문")

fc = pd.read_csv(ROOT / "data/processed/ml_forecasts.csv", index_col=0); fc.index = pd.PeriodIndex(fc.index, freq="M")
W = pd.read_csv(ROOT / "data/processed/p3_weights.csv", index_col=0); W.index = pd.PeriodIndex(W.index, freq="M")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0); ds.index = pd.PeriodIndex(ds.index, freq="M")
wml = ds["tgt_wml_next"].reindex(fc.index).values
MODELS = [c for c in W.columns]
RET = {m: W[m].values * wml for m in MODELS}
n = len(wml)

sr = lambda r: np.sqrt(12) * np.nanmean(r) / np.nanstd(r)

# ---------- (a) 페어드 블록 부트스트랩 ΔSharpe ----------
def block_boot_dsr(r1, r0, B=5000, blk=12):
    diffs = np.empty(B)
    for b in range(B):
        idx = np.concatenate([np.arange(s, s + blk) % n
                              for s in rng.integers(0, n, size=int(np.ceil(n / blk)))])[:n]
        diffs[b] = sr(r1[idx]) - sr(r0[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi, (diffs <= 0).mean()

print("===== (a) 블록 부트스트랩 ΔSharpe vs RW126 (B=5000, blk=12) =====")
boot_rows = {}
for m in ["HARX", "ENS", "Ridge", "HAR", "EWMA"]:
    mu, lo, hi, p = block_boot_dsr(RET[m], RET["RW126"])
    boot_rows[m] = dict(dSharpe=sr(RET[m]) - sr(RET["RW126"]), boot_mean=mu, ci_lo=lo, ci_hi=hi, p_le0=p)
    print(f"  {m:>6}: Δ={boot_rows[m]['dSharpe']:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  P(Δ≤0)={p:.3f}")
boot = pd.DataFrame(boot_rows).T

# ---------- (b) Deflated Sharpe Ratio ----------
srs_m = np.array([np.nanmean(RET[m]) / np.nanstd(RET[m]) for m in MODELS])   # 비연율 월간 SR
best_i = int(np.nanargmax(srs_m)); best = MODELS[best_i]; SR = srs_m[best_i]
N = len(MODELS); V = srs_m.var()
emc = 0.5772156649
sr0 = np.sqrt(V) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * np.e)))
r_best = RET[best][~np.isnan(RET[best])]
g3, g4 = stats.skew(r_best), stats.kurtosis(r_best, fisher=False)
dsr = stats.norm.cdf(((SR - sr0) * np.sqrt(len(r_best) - 1)) /
                     np.sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR ** 2))
print(f"\n===== (b) Deflated Sharpe — 최고 모델 {best} =====")
print(f"  월간SR={SR:.4f}  E[maxSR|N=12,noise]={sr0:.4f}  → DSR(진짜일 확률)={dsr:.3f}")

# ---------- (c) 거래비용 스윕 ----------
print("\n===== (c) 순샤프 — 레버리지 조정비용 c×2×|ΔL| =====")
cost_rows = {}
for m in ["RW126", "EWMA", "HAR", "HARX", "Ridge", "ENS"]:
    L = W[m].values
    dL = np.abs(np.diff(L, prepend=L[0]))
    row = {}
    for c_bp in [0, 10, 30, 60, 100]:
        net = RET[m] - c_bp / 1e4 * 2 * dL
        row[f"{c_bp}bp"] = sr(net)
    cost_rows[m] = row
cost = pd.DataFrame(cost_rows).T
print(cost.round(3).to_string())

# ---------- (d) 하위표본 ----------
half = n // 2
print(f"\n===== (d) 하위표본 ΔSharpe vs RW126 (전반 {fc.index[0]}~{fc.index[half-1]} / 후반 ~{fc.index[-1]}) =====")
sub_rows = {}
for m in ["HARX", "ENS", "Ridge"]:
    sub_rows[m] = {"전반": sr(RET[m][:half]) - sr(RET["RW126"][:half]),
                   "후반": sr(RET[m][half:]) - sr(RET["RW126"][half:])}
sub = pd.DataFrame(sub_rows).T
print(sub.round(3).to_string())

# ---------- (e) 타깃 불변성 ----------
s12 = sr(RET["ENS"])
s06 = sr(0.5 * RET["ENS"]); s20 = sr(20 / 12 * RET["ENS"])
inv = max(abs(s06 - s12), abs(s20 - s12))
print(f"\n===== (e) σ_target 불변성: |ΔSharpe(6%,20% vs 12%)| = {inv:.2e} (≈0이어야) =====")

# ---------- 저장 + 게이트 ----------
boot.round(4).to_csv(ROOT / "output/tables/p5_bootstrap.csv")
cost.round(4).to_csv(ROOT / "output/tables/p5_cost_sweep.csv")
sub.round(4).to_csv(ROOT / "output/tables/p5_subsamples.csv")
pd.DataFrame({"best": [best], "SR_monthly": [SR], "E_max_SR0": [sr0], "DSR": [dsr]}).to_csv(
    ROOT / "output/tables/p5_deflated_sharpe.csv", index=False)
assert np.isfinite(boot["ci_lo"]).all() and inv < 1e-9
print("\n[GATES] 부트스트랩 유한 ✓ · 타깃 불변성 ✓ · 표 4종 저장 ✓")
