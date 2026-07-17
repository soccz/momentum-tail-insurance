"""
Build RMRF, SMB, HML from KOREAN STOCK DATA ONLY (no external FF3 file).

  RMRF : value-weighted return of all stocks (RF=0; no KR risk-free series on disk).
  SMB/HML : textbook Fama-French 2x3 construction from the same panel + book-to-market.
        - June(Y) rebalance. Size = market equity at end of June(Y), median split (S/B).
        - B/M = 1/PBR at end of Dec(Y-1); 30/70 breakpoints (L/M/H).
        - 6 value-weighted portfolios (SL,SM,SH,BL,BM,BH), held Jul(Y)..Jun(Y+1).
        - SMB = mean(S*) - mean(B*);  HML = mean(*H) - mean(*L).

Inputs:
  panel: mom_prices.parquet (main) or prices_adj.parquet (clean)  [AdjClose, MarketCap]
  book-to-market: data/processed/pbr_monthly.parquet  (Code, Date=month-end, PBR)
Outputs (data/processed/):
  ff_own_daily{suf}.csv    Date, rmrf, smb, hml
  ff_own_monthly{suf}.csv  month, rmrf, smb, hml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from build_factor import load_panel, month_end_index, leg_daily_vw, LONG, PRIMARY, RETURN_CAP

OUT = Path("/mnt/20t/졸업논문/data/processed")
PBR = OUT / "pbr_monthly.parquet"


def market_factor(ret: pd.DataFrame, cap: pd.DataFrame) -> pd.Series:
    """Daily value-weighted market return (weights = lagged market cap)."""
    w = cap.shift(1).reindex_like(ret)
    valid = ret.notna() & w.notna()
    num = (ret.where(valid) * w.where(valid)).sum(axis=1, skipna=True)
    den = w.where(valid).sum(axis=1, skipna=True)
    return (num / den).where(den > 0)


def ff_2x3(adj, cap, ret, pbr_wide) -> pd.DataFrame:
    """Daily SMB, HML from annual June-rebalanced 2x3 sorts."""
    me = month_end_index(adj.index)
    me_by_ym = {(d.year, d.month): d for d in me}
    years = sorted({d.year for d in me})
    smb_parts, hml_parts = [], []

    for Y in years:
        if (Y, 6) not in me_by_ym or (Y - 1, 12) not in me_by_ym:
            continue
        jun = me_by_ym[(Y, 6)]
        dec = me_by_ym[(Y - 1, 12)]
        size = cap.loc[jun]
        # B/M at Dec(Y-1) = 1/PBR
        if dec not in pbr_wide.index:
            continue
        bm = 1.0 / pbr_wide.loc[dec]
        elig = size.notna() & (size > 0) & bm.notna() & (bm > 0)
        s = size[elig]; b = bm[elig]
        if len(s) < 30:
            continue
        small = s < s.median()
        lo = b <= b.quantile(0.30)
        hi = b >= b.quantile(0.70)
        mid = (~lo) & (~hi)
        groups = {
            "SL": s.index[small & lo], "SM": s.index[small & mid], "SH": s.index[small & hi],
            "BL": s.index[~small & lo], "BM": s.index[~small & mid], "BH": s.index[~small & hi],
        }
        # holding window Jul(Y)..Jun(Y+1)
        end = me_by_ym.get((Y + 1, 6), adj.index[-1])
        hold = adj.index[(adj.index > jun) & (adj.index <= end)]
        if len(hold) == 0:
            continue
        rw = ret.reindex(hold)
        legret = {}
        for g, members in groups.items():
            members = [m for m in members if m in rw.columns]
            if not members:
                legret[g] = pd.Series(0.0, index=hold)
                continue
            w0 = size[members] / size[members].sum()
            legret[g] = leg_daily_vw(rw[members], w0).reindex(hold)
        L = pd.DataFrame(legret)
        smb = L[["SL", "SM", "SH"]].mean(axis=1) - L[["BL", "BM", "BH"]].mean(axis=1)
        hml = L[["SH", "BH"]].mean(axis=1) - L[["SL", "BL"]].mean(axis=1)
        smb_parts.append(smb); hml_parts.append(hml)

    smb = pd.concat(smb_parts).sort_index()
    hml = pd.concat(hml_parts).sort_index()
    smb = smb[~smb.index.duplicated()]; hml = hml[~hml.index.duplicated()]
    return pd.DataFrame({"smb": smb, "hml": hml})


def monthly_compound(daily: pd.Series) -> pd.Series:
    g = daily.dropna().groupby(daily.dropna().index.to_period("M").to_timestamp("M")).apply(
        lambda x: np.prod(1 + x) - 1)
    g.index.name = "month"
    return g


def run(panel: str, suf: str, start, end):
    adj, cap = load_panel(panel)
    if start: adj = adj.loc[adj.index >= start]; cap = cap.loc[cap.index >= start]
    if end: adj = adj.loc[adj.index <= end]; cap = cap.loc[cap.index <= end]
    ret = adj.pct_change(fill_method=None)
    ret = ret.where(ret.abs() <= RETURN_CAP)
    print(f"[panel] {adj.shape[0]}d x {adj.shape[1]} stocks {adj.index.min().date()}..{adj.index.max().date()}")

    rmrf = market_factor(ret, cap)
    daily = pd.DataFrame({"rmrf": rmrf})

    if PBR.exists():
        pbr = pd.read_parquet(PBR)
        pbr["Date"] = pd.to_datetime(pbr["Date"])
        pbr_wide = pbr.pivot_table(index="Date", columns="Code", values="PBR", aggfunc="last")
        ff = ff_2x3(adj, cap, ret, pbr_wide)
        daily = daily.join(ff, how="left")
        print(f"[ff] SMB/HML built {ff.index.min().date()}..{ff.index.max().date()}")
    else:
        print("[ff] pbr_monthly.parquet not ready -> RMRF only")

    daily.index.name = "Date"
    daily.to_csv(OUT / f"ff_own_daily{suf}.csv")
    monthly = pd.DataFrame({c: monthly_compound(daily[c]) for c in daily.columns})
    monthly.to_csv(OUT / f"ff_own_monthly{suf}.csv")

    for c in daily.columns:
        m = monthly[c].dropna()
        print(f"[{c}] ann_mean={m.mean()*12*100:+.1f}%  ann_vol={m.std()*np.sqrt(12)*100:.1f}%  "
              f"sharpe={np.sqrt(12)*m.mean()/m.std():.2f}  n={len(m)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["main", "clean"], default="main")
    args = ap.parse_args()
    if args.panel == "main":
        run(LONG, "", "1990-01-01", None)
    else:
        run(PRIMARY, "_clean", None, None)
