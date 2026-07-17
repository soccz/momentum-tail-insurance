"""
Reproduce Figures 1, 2, 3, 6 of Barroso & Santa-Clara (2015) with Korean data.

Fig 1  Momentum crashes    : cumulative WML vs RMRF in two turbulent periods
Fig 2  Realized volatility : annualized realized vol of momentum over time
Fig 3  Risk quintiles      : WML/RMRF next-12m vol, return, Sharpe by prior-6m-vol quintile
Fig 6  Benefit of RM       : cumulative WML vs WML* in the same two turbulent periods
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path("/mnt/20t/졸업논문/data/processed")
FIG = Path("/mnt/20t/졸업논문/output/figures")
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})

# Two Korean turbulent periods (analogous to Barroso's 1930s / 2000s).
# Main panel (1990+): the 1998 IMF-recovery momentum crash and the GFC.
# Clean panel (2005+): GFC and COVID era.
PERIODS_MAIN = [("1997-01", "2002-12", "Panel A: IMF recovery & dot-com 1997-2002"),
                ("2007-01", "2012-12", "Panel B: GFC 2007-2012")]
PERIODS_CLEAN = [("2007-01", "2012-12", "Panel A: GFC 2007-2012"),
                 ("2017-01", "2023-12", "Panel B: COVID era 2017-2023")]
PERIODS = PERIODS_MAIN


def load_monthly(suf: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"factors_monthly{suf}.csv", parse_dates=["month"]).set_index("month")


def load_daily(suf: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"factors_daily{suf}.csv", parse_dates=["Date"]).set_index("Date")


def cum(series: pd.Series) -> pd.Series:
    return (1 + series.dropna()).cumprod()


# ---------------------------------------------------------------- Fig 1
def fig1(f: pd.DataFrame, suf: str):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5))
    for ax, (s, e, title) in zip(axes, PERIODS):
        w = f.loc[s:e]
        cw, cr = cum(w["wml"]), cum(w["rmrf"])
        ax.plot(cr.index, cr.values, color="#1f4e79", lw=1.6, label="RMRF (market)")
        ax.plot(cw.index, cw.values, color="#c00000", lw=1.6, ls="--", label="WML (momentum)")
        for c, col in [(cw, "#c00000"), (cr, "#1f4e79")]:
            ax.annotate(f"${c.iloc[-1]:.2f}", (c.index[-1], c.iloc[-1]),
                        color=col, fontsize=8, va="center")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("Cumulative return ($1)")
        ax.axhline(1, color="gray", lw=0.6, alpha=0.5)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Fig 1. Momentum crashes (Korea): WML vs market in turbulent periods", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / f"fig1_momentum_crashes{suf}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig2(f: pd.DataFrame, suf: str):
    rv = f["rv_ann"].dropna()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(rv.index, rv.values, color="#1f4e79", lw=0.8)
    ax.fill_between(rv.index, rv.values, color="#1f4e79", alpha=0.12)
    ax.set_ylabel("Annualized realized vol (%)")
    ax.set_title(f"Fig 2. Realized volatility of Korean momentum "
                 f"(min {rv.min():.1f}%, max {rv.max():.1f}%)", fontsize=10, loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG / f"fig2_realized_vol{suf}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def _quintile_stats(daily: pd.Series, monthly: pd.Series, nq: int = 5) -> pd.DataFrame:
    """For each month: prior-126d realized variance (sort var), and next-12-month
    realized vol / cumulative return / Sharpe (outcomes). Then average by quintile."""
    me = monthly.index
    sq = (daily ** 2)
    rv6 = sq.rolling(126, min_periods=100).sum()                    # prior 6m variance (daily)
    rv6_me = rv6.groupby(rv6.index.to_period("M").to_timestamp("M")).last().reindex(me)

    fwd_vol, fwd_ret, fwd_sr = {}, {}, {}
    for i, t in enumerate(me):
        nxt = monthly.iloc[i + 1:i + 13].dropna()
        if len(nxt) < 12:
            continue
        mu, sd = nxt.mean(), nxt.std()
        fwd_vol[t] = sd * np.sqrt(12) * 100
        fwd_ret[t] = ((1 + nxt).prod() - 1) * 100
        fwd_sr[t] = np.sqrt(12) * mu / sd if sd > 0 else np.nan
    df = pd.DataFrame({"sortvar": rv6_me, "vol": pd.Series(fwd_vol),
                       "ret": pd.Series(fwd_ret), "sharpe": pd.Series(fwd_sr)}).dropna()
    df["q"] = pd.qcut(df["sortvar"], nq, labels=False) + 1
    return df.groupby("q")[["vol", "ret", "sharpe"]].mean()


def fig3(f: pd.DataFrame, daily: pd.DataFrame, suf: str):
    common = daily.dropna()
    stats = {}
    for name in ["wml", "rmrf"]:
        stats[name] = _quintile_stats(common[name], f[name].dropna())
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8))
    labels = [("vol", "Volatility (%)", "Panel A: Risk and risk"),
              ("ret", "Annual return (%)", "Panel B: Risk and return"),
              ("sharpe", "Sharpe ratio", "Panel C: Risk and Sharpe ratio")]
    x = np.arange(1, 6); wd = 0.38
    for ax, (col, ylab, title) in zip(axes, labels):
        ax.bar(x - wd/2, stats["wml"][col].reindex(x).values, wd, label="WML", color="#c9b18a")
        ax.bar(x + wd/2, stats["rmrf"][col].reindex(x).values, wd, label="RMRF", color="#2b2b2b")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel(ylab); ax.set_xticks(x); ax.set_xlabel("Quintile (prior 6-month vol)")
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 3. Performance conditional on prior 6-month realized volatility (Korea)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / f"fig3_risk_quintiles{suf}.png", bbox_inches="tight")
    plt.close(fig)
    # also dump the numbers
    for name in stats:
        stats[name].round(2).to_csv(FIG / f"fig3_{name}{suf}.csv")


# ---------------------------------------------------------------- Fig 6
def fig6(f: pd.DataFrame, suf: str):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5))
    for ax, (s, e, title) in zip(axes, PERIODS):
        w = f.loc[s:e]
        cw, cs = cum(w["wml"]), cum(w["wml_star"])
        ax.plot(cw.index, cw.values, color="#1f4e79", lw=1.6, label="WML (plain)")
        ax.plot(cs.index, cs.values, color="#c00000", lw=1.6, ls="--", label="WML* (risk-managed)")
        for c, col in [(cw, "#1f4e79"), (cs, "#c00000")]:
            ax.annotate(f"${c.iloc[-1]:.2f}", (c.index[-1], c.iloc[-1]),
                        color=col, fontsize=8, va="center")
        ax.set_title(title.replace("Panel", "Panel"), fontsize=10, loc="left")
        ax.set_ylabel("Cumulative return ($1)")
        ax.axhline(1, color="gray", lw=0.6, alpha=0.5)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Fig 6. Benefit of risk-management (Korea): WML vs WML*", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / f"fig6_risk_managed{suf}.png", bbox_inches="tight")
    plt.close(fig)


def run(suf: str):
    global PERIODS
    PERIODS = PERIODS_CLEAN if suf == "_clean" else PERIODS_MAIN
    f = load_monthly(suf)
    d = load_daily(suf)
    fig1(f, suf); fig2(f, suf); fig3(f, d, suf); fig6(f, suf)
    print(f"[figs] wrote fig1/2/3/6{suf}.png to {FIG}")
    # print Fig3 tables for verification
    for name in ["wml", "rmrf"]:
        t = pd.read_csv(FIG / f"fig3_{name}{suf}.csv", index_col=0)
        print(f"\nFig3 {name.upper()} by quintile:\n{t.round(2).to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["main", "clean"], default="main")
    args = ap.parse_args()
    run("" if args.panel == "main" else "_clean")
