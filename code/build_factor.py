"""
Build the Korean momentum factor (WML) to Barroso & Santa-Clara (2015) spec.

Design (locked in docs/01_reproduction_spec.md):
  - Universe : all stocks (KOSPI+KOSDAQ, survivorship-free, delisted included)
  - Breakpts : deciles (all-stock), D10 winner - D1 loser
  - Weight   : value-weight by formation-date market cap (buy-and-hold within month)
  - Signal   : cumret from end(M-12) to end(M-1)  == skip most-recent-month (12-1)
  - Rebalance: monthly (form at month-end M, hold month M+1)
  - Outliers : |daily stock return| > 0.50 -> NaN (data artifact guard)
  - Delisting: del-a (drop at last valid price, renormalize; no penalty)

Outputs (data/processed/):
  wml_daily.csv     Date, winner, loser, wml           (daily leg + long-short)
  wml_monthly.csv   month, winner, loser, wml, n_win, n_los
Usage:
  python build_factor.py            # primary panel (prices_adj, 2005+)
  python build_factor.py --long     # long robustness panel (mom_prices, 1979+)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/mnt/20t/졸업논문")
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = "/mnt/20t/study/mom_paper_test/data/processed/prices_adj.parquet"
LONG = "/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet"

RETURN_CAP = 0.50   # |daily return| beyond this = corporate-action artifact
N_BINS = 10         # deciles


def load_panel(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (adjclose_wide, mktcap_wide) indexed by Date, columns=Code."""
    cols = pd.read_parquet(path).columns
    ac = "AdjClose"
    df = pd.read_parquet(path, columns=["Date", "Code", ac, "MarketCap"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=[ac])
    df = df[df[ac] > 0]
    adj = df.pivot_table(index="Date", columns="Code", values=ac, aggfunc="last")
    cap = df.pivot_table(index="Date", columns="Code", values="MarketCap", aggfunc="last")
    adj = adj.sort_index()
    cap = cap.reindex_like(adj)
    return adj, cap


def month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in idx."""
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().values)


def leg_daily_vw(ret_win: pd.DataFrame, w0: pd.Series) -> pd.Series:
    """Buy-and-hold value-weighted daily returns over a holding window.

    ret_win : days x stocks daily returns (NaN where no data / halted / capped)
    w0      : formation-date weights (sum=1) for the leg's members
    Delisting (del-a): a stock with NaN return on day d is excluded from that day's
    return and from the denominator -> survivors auto-renormalize.
    """
    members = [c for c in w0.index if c in ret_win.columns]
    if not members:
        return pd.Series(dtype=float, index=ret_win.index)
    R = ret_win[members]
    w0 = w0[members]
    # value at END of each day = w0 * cumprod(1+r); NaN return -> value flat that day
    gross = (1.0 + R.fillna(0.0)).cumprod()
    V_end = gross.mul(w0, axis=1)
    V_prev = V_end.shift(1)
    V_prev.iloc[0] = w0.values            # start-of-first-day value = formation weight
    valid = R.notna()
    num = (V_prev.where(valid) * R).sum(axis=1, skipna=True)
    den = V_prev.where(valid).sum(axis=1, skipna=True)
    out = num / den
    return out.where(den > 0)


def build(path: str, suf: str, start: str | None, end: str | None) -> None:
    print(f"[load] {path}")
    adj, cap = load_panel(path)
    if start:
        adj = adj.loc[adj.index >= start]; cap = cap.loc[cap.index >= start]
    if end:
        adj = adj.loc[adj.index <= end]; cap = cap.loc[cap.index <= end]
    print(f"[panel] {adj.shape[0]} days x {adj.shape[1]} stocks, "
          f"{adj.index.min().date()}..{adj.index.max().date()}")

    ret = adj.pct_change(fill_method=None)
    ret = ret.where(ret.abs() <= RETURN_CAP)          # outlier guard

    me = month_end_index(adj.index)
    print(f"[months] {len(me)} month-ends")

    daily_win, daily_los = [], []
    monthly_rows = []
    for k in range(12, len(me) - 1):
        T = me[k]                       # formation = end of month M
        p_now = adj.loc[me[k - 1]]      # end(M-1)  -> skip most recent month M
        p_then = adj.loc[me[k - 12]]    # end(M-12)
        sig = (p_now / p_then) - 1.0
        capT = cap.loc[T]
        elig = sig.notna() & capT.notna() & (capT > 0)
        s = sig[elig]
        if len(s) < N_BINS * 3:         # need enough names to form deciles
            continue
        # deciles: 0 = loser, N_BINS-1 = winner
        try:
            q = pd.qcut(s, N_BINS, labels=False, duplicates="drop")
        except ValueError:
            continue
        win = s.index[q == q.max()]
        los = s.index[q == q.min()]

        hold = adj.index[(adj.index > T) & (adj.index <= me[k + 1])]
        if len(hold) == 0:
            continue
        rw = ret.reindex(hold)

        wW = capT[win] / capT[win].sum()
        wL = capT[los] / capT[los].sum()
        rW = leg_daily_vw(rw[list(win)], wW)
        rL = leg_daily_vw(rw[list(los)], wL)
        daily_win.append(rW); daily_los.append(rL)

        mW = float(np.prod(1.0 + rW.dropna()) - 1.0)
        mL = float(np.prod(1.0 + rL.dropna()) - 1.0)
        monthly_rows.append({
            "month": me[k + 1].to_period("M").to_timestamp("M"),
            "winner": mW, "loser": mL, "wml": mW - mL,
            "n_win": len(win), "n_los": len(los),
        })

    dW = pd.concat(daily_win).sort_index()
    dL = pd.concat(daily_los).sort_index()
    dW = dW[~dW.index.duplicated()]; dL = dL[~dL.index.duplicated()]
    daily = pd.DataFrame({"winner": dW, "loser": dL})
    daily["wml"] = daily["winner"] - daily["loser"]
    daily.index.name = "Date"
    daily = daily.dropna(subset=["wml"])

    monthly = pd.DataFrame(monthly_rows).set_index("month")

    daily.to_csv(OUT / f"wml_daily{suf}.csv")
    monthly.to_csv(OUT / f"wml_monthly{suf}.csv")
    print(f"[out] wml_daily{suf}.csv  {len(daily)} days  {daily.index.min().date()}..{daily.index.max().date()}")
    print(f"[out] wml_monthly{suf}.csv {len(monthly)} months")
    # quick sanity
    m = monthly["wml"]
    ann_ret = m.mean() * 12 * 100
    ann_vol = m.std() * np.sqrt(12) * 100
    from scipy import stats
    print(f"[sanity] WML monthly: ann_ret={ann_ret:.2f}%  ann_vol={ann_vol:.2f}%  "
          f"sharpe={ann_ret/ann_vol:.3f}  skew={stats.skew(m):.2f}  "
          f"exkurt={stats.kurtosis(m):.2f}  min={m.min()*100:.1f}%  max={m.max()*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["main", "clean"], default="main",
                    help="main = mom_prices 1990+ (long, primary); clean = prices_adj 2005+")
    args = ap.parse_args()
    if args.panel == "main":
        build(LONG, "", "1990-01-01", None)          # primary: long Korean panel
    else:
        build(PRIMARY, "_clean", None, None)          # robustness: clean 2005+ panel
