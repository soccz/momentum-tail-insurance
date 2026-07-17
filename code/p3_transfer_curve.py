"""
P3 — 전이곡선: 예측 정확도가 성과로 얼마나 전이되는가 (킬러 피겨).

각 예측기 m의 분산예측 F_m[t] → 스케일 L_t = (0.12/√12)/√F_m[t] → WML*_{t+1} = L_t·WML_{t+1}.
전 모델 동일 규칙(Eq 6) — 다른 것은 예측기뿐이므로 성과 차이는 전부 예측력에 귀속.

산출: 모델별 포트폴리오 성과표 + 전이곡선 그림(x=OOS R², y=샤프/최악월).
게이트: G1 순위상관(예측력↔샤프)>0 확인 출력 · G2 RW126 스케일=Eq(5) 재현 · G3 n=292 정렬.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_k=[f.fname for f in fm.fontManager.ttflist if 'Nanum' in f.name or 'Noto Sans CJK' in f.name]
if _k: plt.rcParams['font.family']=fm.FontProperties(fname=_k[0]).get_name()
plt.rcParams['axes.unicode_minus']=False
from scipy import stats

ROOT = Path("/mnt/20t/졸업논문")
fc = pd.read_csv(ROOT / "data/processed/ml_forecasts.csv", index_col=0)
fc.index = pd.PeriodIndex(fc.index, freq="M")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0)
ds.index = pd.PeriodIndex(ds.index, freq="M")
lb = pd.read_csv(ROOT / "output/tables/p2_leaderboard.csv", index_col=0)

wml_next = ds["tgt_wml_next"].reindex(fc.index)          # 행 t: t+1월 WML 수익
MODELS = [c for c in fc.columns if c != "actual_var"]
TGT_M = 0.12 / np.sqrt(12)

def perf(r):
    r = r.dropna(); mu, sd = r.mean(), r.std()
    t = mu / (sd / np.sqrt(len(r)))
    es5 = r.sort_values().head(max(1, int(0.05 * len(r)))).mean() * 100
    return dict(sharpe=np.sqrt(12) * mu / sd, t_stat=t, mean_ann=mu * 12 * 100,
                vol_ann=sd * np.sqrt(12) * 100, skew=stats.skew(r, bias=False),
                kurt=stats.kurtosis(r, bias=False), worst=r.min() * 100, ES5=es5,
                mdd=((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100)

rows, weights = {}, {}
rows["RAW(무관리)"] = perf(wml_next)
for m in MODELS:
    L = TGT_M / np.sqrt(fc[m])
    weights[m] = L
    rows[m] = perf(L * wml_next)
pf = pd.DataFrame(rows).T
pf = pf.join(lb[["QLIKE", "OOS_R2"]], how="left")

print("===== P3 포트폴리오 성과 (OOS %s~%s, n=%d) =====" % (fc.index[0], fc.index[-1], len(fc)))
print(pf.round(3).to_string())

# ---- 전이 통계: 예측력 → 샤프 ----
ok = pf.dropna(subset=["OOS_R2"])
core = ok[ok["OOS_R2"] > 0]                                # 퇴화 모델(음수 R²) 제외 회귀
slope, icpt, rr, pp, se = stats.linregress(core["OOS_R2"], core["sharpe"])
rho = stats.spearmanr(ok["OOS_R2"], ok["sharpe"]).statistic
print(f"\n[전이] 샤프 = {icpt:.3f} + {slope:.4f}×OOS_R²(%p)  (R²>0 모델 {len(core)}개, r={rr:.2f}, p={pp:.3f})")
print(f"[전이] +10%p 예측력 ≈ 샤프 +{slope*10:.3f} | 순위상관(전체 12모델) ρ={rho:.2f}")
for ycol, nm in [("worst", "최악월"), ("skew", "왜도"), ("mdd", "MDD")]:
    s2, i2, r2_, p2_, _ = stats.linregress(core["OOS_R2"], core[ycol])
    print(f"[꼬리전이] {nm} = {i2:.2f} + {s2:.4f}×R²  (r={r2_:.2f}, p={p2_:.3f})")

# ---- 킬러 피겨 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
XMIN = -8
plot_df = ok.copy(); plot_df["x"] = plot_df["OOS_R2"].clip(lower=XMIN)   # 퇴화 모델은 왼쪽 경계에 표시
for ax, ycol, ylab in [(axes[0], "sharpe", "연율 샤프비율"), (axes[1], "ES5", "월간 5% 기대꼬리손실 ES5 (%)")]:
    deg = plot_df["OOS_R2"] < XMIN
    ax.scatter(plot_df.loc[~deg, "x"], plot_df.loc[~deg, ycol], s=55, color="#c4724e", zorder=3)
    ax.scatter(plot_df.loc[deg, "x"], plot_df.loc[deg, ycol], s=55, marker="<", color="#96887a", zorder=3)
    for m, r in plot_df.iterrows():
        lbl = m + ("(축외)" if r["OOS_R2"] < XMIN else "")
        ax.annotate(lbl, (r["x"], r[ycol]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(XMIN - 2, plot_df["x"].max() + 6)
    if ycol == "sharpe":
        xs = np.linspace(core["OOS_R2"].min(), core["OOS_R2"].max(), 50)
        # 조인트 부트스트랩 95% CI (p11_fw_and_slope.csv) — 점별 p는 9점 종속으로 무효
        _sl = pd.read_csv(ROOT / "output/tables/p11_fw_and_slope.csv").set_index("metric")["value"]
        _lo, _hi = float(_sl["slope_lo"]), float(_sl["slope_hi"])
        ax.plot(xs, icpt + slope * xs, "--", color="#2f6d8f", lw=1.4,
                label=f"기울기 {slope:.4f}/%p (조인트 95% CI [{_lo:+.3f}, {_hi:+.3f}], 0 포함)")
        ax.axhline(rows["RAW(무관리)"]["sharpe"], color="gray", lw=.8, ls=":", label="RAW 무관리")
        ax.legend(frameon=False, fontsize=8.5)
    ax.set_xlabel("변동성 예측 OOS R² (%)"); ax.set_ylabel(ylab)
fig.suptitle("전이곡선 — 위험 예측이 좋아지면 성과가 얼마나 좋아지는가 (한국 WML, OOS 292개월)")
fig.tight_layout()
fig.savefig(ROOT / "output/figures/p3_transfer_curve.png", dpi=130, bbox_inches="tight")

# ---- 저장 + 게이트 ----
pf.round(4).to_csv(ROOT / "output/tables/p3_portfolio.csv")
pd.DataFrame(weights, index=fc.index).round(4).to_csv(ROOT / "data/processed/p3_weights.csv")

g2 = np.allclose(weights["RW126"], TGT_M / np.sqrt(ds["fvar_126"].reindex(fc.index)))
print(f"\n[G1] 순위상관 ρ={rho:.2f} (>0)  [G2] RW126 스케일=Eq(5) {g2}  [G3] n={len(fc)}")
assert rho > 0 and g2 and len(fc) == 292
print("[GATES] 통과 ✓  → output/figures/p3_transfer_curve.png")
