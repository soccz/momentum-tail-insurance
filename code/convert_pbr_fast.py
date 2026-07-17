"""
Fast PBR.xlsx -> month-end parquet via python-calamine (Rust reader).
Keeps only month-end columns and only codes present in the momentum panel
(mom_prices) to bound output size. Output: data/processed/pbr_monthly.parquet.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import python_calamine as pc

SRC = "/mnt/20t/study/mom_paper_test/data/PBR.xlsx"
MOM = "/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet"
OUT = Path("/mnt/20t/졸업논문/data/processed/pbr_monthly.parquet")


def norm(code):
    return str(code).lstrip("A").zfill(6)


def main():
    t0 = time.time()
    keep = set(norm(c) for c in pd.read_parquet(MOM, columns=["Code"])["Code"].unique())
    print(f"[universe] {len(keep)} momentum codes to keep", flush=True)

    print("[read] calamine to_python ...", flush=True)
    wb = pc.CalamineWorkbook.from_path(SRC)
    rows = wb.get_sheet_by_name("Sheet1").to_python(skip_empty_area=False)
    print(f"[read] {len(rows)} rows in {(time.time()-t0)/60:.1f}min", flush=True)

    # locate header row (contains '코드' and '코드명'), date columns start at first datetime
    import datetime as dt
    hdr = None
    for i, r in enumerate(rows[:40]):
        sv = [str(x) if x is not None else "" for x in r]
        if "코드" in sv and "코드명" in sv:
            code_c = sv.index("코드")
            date_c0 = next(j for j, x in enumerate(r)
                           if isinstance(x, (dt.date, dt.datetime)))
            hdr = i
            break
    dates = pd.to_datetime([rows[hdr][j] for j in range(date_c0, len(rows[hdr]))], errors="coerce")
    dser = pd.Series(range(len(dates)), index=dates)
    me_pos = list(dser.groupby([dates.year, dates.month]).last().values)
    me_dates = dates[me_pos]
    print(f"[header] row {hdr} code_c={code_c} date_c0={date_c0} ndates={len(dates)} "
          f"month-ends={len(me_pos)} {me_dates[0].date()}..{me_dates[-1].date()}", flush=True)

    series = {}
    for r in rows[hdr + 1:]:
        code = r[code_c]
        if code is None:
            continue
        c = norm(code)
        if c not in keep:
            continue
        vals = r[date_c0:]
        series[c] = [vals[p] if p < len(vals) else None for p in me_pos]
    print(f"[filter] kept {len(series)} codes; building long ({(time.time()-t0)/60:.1f}min)", flush=True)

    wide = pd.DataFrame.from_dict(series, orient="index", columns=me_dates)
    wide.index.name = "Code"
    long = wide.reset_index().melt(id_vars="Code", var_name="Date", value_name="PBR")
    long["PBR"] = pd.to_numeric(long["PBR"], errors="coerce")
    long["Date"] = pd.to_datetime(long["Date"])
    long = long.dropna(subset=["PBR"])
    long = long[long["PBR"] > 0]
    long.to_parquet(OUT, index=False)
    print(f"[out] {OUT}  {len(long):,} rows  {long.Code.nunique()} codes  "
          f"{long.Date.min().date()}..{long.Date.max().date()}  ({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
