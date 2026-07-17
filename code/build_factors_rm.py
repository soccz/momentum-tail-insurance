"""
Assemble factor panels + realized volatility + risk-managed WML* (Barroso Eq 2,5,6).
Korean-data-only: factors come from build_ff_factors.py (no external FF3 file).

Inputs (data/processed/):
  wml_daily{suf}.csv, wml_monthly{suf}.csv     (build_factor.py)
  ff_own_daily{suf}.csv, ff_own_monthly{suf}.csv  (build_ff_factors.py: rmrf, smb, hml)
Outputs (data/processed/):
  factors_monthly{suf}.csv  month, rmrf, smb, hml, wml, wml_star, rv_ann, sigma_hat, scale
  factors_daily{suf}.csv    Date, rmrf, wml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/mnt/20t/졸업논문/data/processed")

TARGET_ANN = 0.12
TARGET_M = TARGET_ANN / np.sqrt(12)
RV21 = 21     # Eq 2 (Fig 2 diagnostic)
RV126 = 126   # Eq 5 (6-month forecast for scaling)


def me_stamp(idx: pd.DatetimeIndex) -> pd.Series:
    return idx.to_period("M").to_timestamp("M")


def run(suf: str) -> None:
    wd = pd.read_csv(OUT / f"wml_daily{suf}.csv", parse_dates=["Date"]).set_index("Date")
    wm = pd.read_csv(OUT / f"wml_monthly{suf}.csv", parse_dates=["month"]).set_index("month")
    ffm = pd.read_csv(OUT / f"ff_own_monthly{suf}.csv", parse_dates=["month"]).set_index("month")
    ffd = pd.read_csv(OUT / f"ff_own_daily{suf}.csv", parse_dates=["Date"]).set_index("Date")

    # ---- Eq 2: 21-day realized variance -> annualized realized vol (Fig 2) ----
    sq = wd["wml"] ** 2
    rv21_daily = sq.rolling(RV21, min_periods=RV21).sum()
    rv_ann = np.sqrt(12.0 * rv21_daily) * 100.0
    rv_ann_me = rv_ann.groupby(me_stamp(rv_ann.index)).last()

    # ---- Eq 5: 126-day forecast (monthly variance) as of each day ----
    fvar_daily = 21.0 * sq.rolling(RV126, min_periods=RV126).mean()
    fvar_me = fvar_daily.groupby(me_stamp(fvar_daily.index)).last()
    sigma2_hat = fvar_me.shift(1)                    # forecast for month t made at end of t-1
    sigma_hat = np.sqrt(sigma2_hat)

    # ---- Eq 6: risk-managed WML* ----
    scale = TARGET_M / sigma_hat
    wml = wm["wml"]
    wml_star = scale.reindex(wml.index) * wml

    idx = ffm.index.union(wml.index).sort_values()
    out = pd.DataFrame(index=idx)
    for c in ["rmrf", "smb", "hml"]:
        out[c] = ffm[c] if c in ffm.columns else np.nan
    out["wml"] = wml
    out["wml_star"] = wml_star
    out["rv_ann"] = rv_ann_me.reindex(idx)
    out["sigma_hat"] = sigma_hat.reindex(idx)
    out["scale"] = scale.reindex(idx)
    out.index.name = "month"
    out.to_csv(OUT / f"factors_monthly{suf}.csv")

    daily = pd.DataFrame({"rmrf": ffd["rmrf"], "wml": wd["wml"]})
    daily.index.name = "Date"
    daily.to_csv(OUT / f"factors_daily{suf}.csv")

    print(f"[out] factors_monthly{suf}.csv  {out.index.min().date()}..{out.index.max().date()}  n={len(out)}")
    common = out.dropna(subset=["wml", "wml_star"])
    print(f"[wml*] overlap n={len(common)}  {common.index.min().date()}..{common.index.max().date()}")
    print(f"[wml*] realized ann_vol WML={common.wml.std()*np.sqrt(12)*100:.1f}%  "
          f"WML*={common.wml_star.std()*np.sqrt(12)*100:.1f}%  (target 12%)")
    print(f"[rv]  realized vol range {out.rv_ann.min():.1f}%..{out.rv_ann.max():.1f}%  mean {out.rv_ann.mean():.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["main", "clean"], default="main")
    args = ap.parse_args()
    run("" if args.panel == "main" else "_clean")
