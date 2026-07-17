"""
Reproduce Table 1 (descriptive statistics) and Table 3 (economic gains from scaling)
for the Korean data, following Barroso & Santa-Clara (2015).

Stats on monthly returns (decimal):
  Max, Min  = max/min monthly return * 100
  Mean      = monthly mean * 12 * 100           (annualized, arithmetic)
  Std       = monthly std  * sqrt(12) * 100     (annualized)
  Kurtosis  = excess kurtosis (Fisher)
  Skewness  = skewness
  Sharpe    = sqrt(12) * mean/std               (annualized)
Table 3 also:
  Info ratio = sqrt(12) * mean(WML*_n - WML_n) / std(WML*_n - WML_n),
               where each series is divided by its own std (target-independent).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("/mnt/20t/졸업논문/data/processed")
TAB = Path("/mnt/20t/졸업논문/output/tables")
TAB.mkdir(parents=True, exist_ok=True)


def desc(r: pd.Series) -> dict:
    r = r.dropna()
    mu, sd = r.mean(), r.std()
    return {
        "Maximum": r.max() * 100,
        "Minimum": r.min() * 100,
        "Mean": mu * 12 * 100,
        "Std dev": sd * np.sqrt(12) * 100,
        "Kurtosis": stats.kurtosis(r, fisher=True, bias=False),
        "Skewness": stats.skew(r, bias=False),
        "Sharpe": np.sqrt(12) * mu / sd,
    }


def info_ratio(wml: pd.Series, wml_star: pd.Series) -> float:
    df = pd.concat([wml, wml_star], axis=1).dropna()
    a = df.iloc[:, 0] / df.iloc[:, 0].std()
    b = df.iloc[:, 1] / df.iloc[:, 1].std()
    diff = b - a
    return np.sqrt(12) * diff.mean() / diff.std()


def fmt_table(df: pd.DataFrame) -> str:
    return df.round(2).to_markdown()


def run(suf: str) -> None:
    f = pd.read_csv(OUT / f"factors_monthly{suf}.csv", parse_dates=["month"]).set_index("month")

    # ---------- Table 1: common sample where all 4 factors exist ----------
    t1cols = ["rmrf", "smb", "hml", "wml"]
    s1 = f[t1cols].dropna()
    rows = {name.upper(): desc(s1[name]) for name in t1cols}
    t1 = pd.DataFrame(rows).T[["Maximum", "Minimum", "Mean", "Std dev",
                               "Kurtosis", "Skewness", "Sharpe"]]
    t1.index = ["RMRF", "SMB", "HML", "WML"]

    # ---------- Table 3: WML vs WML* on common (scaled) sample ----------
    s3 = f[["wml", "wml_star"]].dropna()
    r3 = {"WML": desc(s3["wml"]), "WML*": desc(s3["wml_star"])}
    t3 = pd.DataFrame(r3).T[["Maximum", "Minimum", "Mean", "Std dev",
                             "Kurtosis", "Skewness", "Sharpe"]]
    t3["Info ratio"] = [np.nan, info_ratio(s3["wml"], s3["wml_star"])]

    p1 = f"{s1.index.min().strftime('%Y:%m')}-{s1.index.max().strftime('%Y:%m')} (n={len(s1)})"
    p3 = f"{s3.index.min().strftime('%Y:%m')}-{s3.index.max().strftime('%Y:%m')} (n={len(s3)})"
    print(f"\n===== Table 1  Descriptive statistics  [{p1}] =====")
    print(fmt_table(t1))
    print(f"\n===== Table 3  Economic gains from scaling  [{p3}] =====")
    print(fmt_table(t3))

    t1.round(3).to_csv(TAB / f"table1{suf}.csv")
    t3.round(3).to_csv(TAB / f"table3{suf}.csv")
    with open(TAB / f"tables{suf}.md", "w") as fh:
        fh.write(f"## Table 1 — Descriptive statistics (Korea)  [{p1}]\n\n{fmt_table(t1)}\n\n")
        fh.write(f"## Table 3 — Economic gains from scaling (Korea)  [{p3}]\n\n{fmt_table(t3)}\n")
    print(f"\n[out] {TAB}/table1{suf}.csv, table3{suf}.csv, tables{suf}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["main", "clean"], default="main")
    args = ap.parse_args()
    run("" if args.panel == "main" else "_clean")
