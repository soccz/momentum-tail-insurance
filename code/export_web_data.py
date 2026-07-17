"""
Export a compact JSON bundle for the interactive HTML paper analyzer.
Computes (Korean data only): WML*, realized vol, risk quintiles, Table 1/3,
crash-window return series — for main (1990+) and clean (2005+) panels.
All four factors (RMRF/SMB/HML/WML) come from the canonical Korean-origin build
(build_ff_factors.py -> ff_own_monthly, full history) for consistency.
(build_ff_2x3 below is legacy/unused, kept for reference.)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("/mnt/20t/졸업논문/data/processed")
WEB = Path("/mnt/20t/졸업논문/output/web")
WEB.mkdir(parents=True, exist_ok=True)
ALLSTOCK = "/mnt/20t/main/FF3/krx_data_final/combined/ALL_STOCKS_HISTORY.parquet"

TARGET_M = 0.12 / np.sqrt(12)


def me_stamp(idx):
    return idx.to_period("M").to_timestamp("M")


def wml_star(wd, wm):
    sq = wd["wml"] ** 2
    fvar = 21.0 * sq.rolling(126, min_periods=126).mean()
    fvar_me = fvar.groupby(me_stamp(fvar.index)).last()
    sigma_hat = np.sqrt(fvar_me.shift(1))                 # monthly vol forecast (decimal)
    scale = (TARGET_M / sigma_hat).reindex(wm.index)
    return wm["wml"] * scale, scale, sigma_hat.reindex(wm.index)


def decompose(wml_d, rmrf_d, win=126):
    """Rolling market/specific variance decomposition of momentum (Eq 7)."""
    df = pd.concat([wml_d, rmrf_d], axis=1).dropna()
    df.columns = ["w", "m"]
    cov = df["w"].rolling(win).cov(df["m"])
    var_m = df["m"].rolling(win).var()
    var_w = df["w"].rolling(win).var()
    beta = cov / var_m
    mkt = (beta ** 2) * var_m
    frac = (mkt / var_w).clip(0, 1)
    out = pd.DataFrame({"frac": frac, "rvw": np.sqrt(var_w * 252) * 100}).dropna()
    return out.groupby(me_stamp(out.index)).last()


def realized_vol(wd):
    sq = wd["wml"] ** 2
    rv = np.sqrt(12.0 * sq.rolling(21, min_periods=21).sum()) * 100
    return rv.groupby(me_stamp(rv.index)).last()


def desc(r):
    r = r.dropna(); mu, sd = r.mean(), r.std()
    return dict(mx=r.max()*100, mn=r.min()*100, mean=mu*12*100, std=sd*np.sqrt(12)*100,
               kurt=float(stats.kurtosis(r, fisher=True, bias=False)),
               skew=float(stats.skew(r, bias=False)), sharpe=float(np.sqrt(12)*mu/sd))


def info_ratio(a, b):
    df = pd.concat([a, b], axis=1).dropna()
    x = df.iloc[:, 0]/df.iloc[:, 0].std(); y = df.iloc[:, 1]/df.iloc[:, 1].std()
    d = y - x
    return float(np.sqrt(12)*d.mean()/d.std())


def quintiles(daily_factor, monthly_factor, nq=5):
    sq = daily_factor ** 2
    rv6 = sq.rolling(126, min_periods=100).sum()
    rv6_me = rv6.groupby(me_stamp(rv6.index)).last().reindex(monthly_factor.index)
    me = monthly_factor.index
    rows = {}
    for i, t in enumerate(me):
        nxt = monthly_factor.iloc[i+1:i+13].dropna()
        if len(nxt) < 12: continue
        mu, sd = nxt.mean(), nxt.std()
        rows[t] = (rv6_me.get(t, np.nan), sd*np.sqrt(12)*100,
                   ((1+nxt).prod()-1)*100, np.sqrt(12)*mu/sd if sd > 0 else np.nan)
    df = pd.DataFrame(rows, index=["sv", "vol", "ret", "sharpe"]).T.dropna()
    df["q"] = pd.qcut(df["sv"], nq, labels=False) + 1
    g = df.groupby("q")[["vol", "ret", "sharpe"]].mean()
    return [dict(q=int(q), vol=round(r.vol, 2), ret=round(r.ret, 2), sharpe=round(r.sharpe, 3))
            for q, r in g.iterrows()]


def ser(s, r=5):
    s = s.dropna()
    return [[d.strftime("%Y-%m"), round(float(v), r)] for d, v in s.items()]


def build_ff_2x3():
    """Fast FF 2x3 SMB/HML from ALL_STOCKS_HISTORY (Close, PBR, MarketCap)."""
    df = pd.read_parquet(ALLSTOCK, columns=["Date", "Code", "Close", "PBR", "MarketCap"])
    df["Date"] = pd.to_datetime(df["Date"])
    adj = df.pivot_table(index="Date", columns="Code", values="Close", aggfunc="last").sort_index()
    cap = df.pivot_table(index="Date", columns="Code", values="MarketCap", aggfunc="last").reindex_like(adj)
    pbr = df.pivot_table(index="Date", columns="Code", values="PBR", aggfunc="last").reindex_like(adj)
    ret = adj.pct_change(fill_method=None); ret = ret.where(ret.abs() <= 0.5)
    me = pd.Series(adj.index, index=adj.index).groupby([adj.index.year, adj.index.month]).last()
    me = pd.DatetimeIndex(me.values); mbym = {(d.year, d.month): d for d in me}
    smb_p, hml_p = [], []
    for Y in sorted({d.year for d in me}):
        if (Y, 6) not in mbym or (Y-1, 12) not in mbym: continue
        jun, dec = mbym[(Y, 6)], mbym[(Y-1, 12)]
        size = cap.loc[jun]; bm = 1.0/pbr.loc[dec]
        el = size.notna() & (size > 0) & bm.notna() & (bm > 0)
        s, b = size[el], bm[el]
        if len(s) < 30: continue
        sm = s < s.median(); lo = b <= b.quantile(.3); hi = b >= b.quantile(.7); mid = ~lo & ~hi
        grp = {"SL": s.index[sm&lo], "SM": s.index[sm&mid], "SH": s.index[sm&hi],
               "BL": s.index[~sm&lo], "BM": s.index[~sm&mid], "BH": s.index[~sm&hi]}
        end = mbym.get((Y+1, 6), adj.index[-1])
        hold = adj.index[(adj.index > jun) & (adj.index <= end)]
        if len(hold) == 0: continue
        rw = ret.reindex(hold); leg = {}
        for g, mem in grp.items():
            mem = [m for m in mem if m in rw.columns]
            if not mem: leg[g] = pd.Series(0.0, index=hold); continue
            w0 = size[mem]/size[mem].sum()
            gross = (1+rw[mem].fillna(0)).cumprod(); Vp = gross.mul(w0, axis=1).shift(1)
            Vp.iloc[0] = w0.values; valid = rw[mem].notna()
            leg[g] = ((Vp.where(valid)*rw[mem]).sum(1) / Vp.where(valid).sum(1)).reindex(hold)
        L = pd.DataFrame(leg)
        smb_p.append(L[["SL","SM","SH"]].mean(1) - L[["BL","BM","BH"]].mean(1))
        hml_p.append(L[["SH","BH"]].mean(1) - L[["SL","BL"]].mean(1))
    smb = pd.concat(smb_p).sort_index(); hml = pd.concat(hml_p).sort_index()
    smb = smb[~smb.index.duplicated()]; hml = hml[~hml.index.duplicated()]
    mc = lambda x: x.dropna().groupby(me_stamp(x.dropna().index)).apply(lambda z: np.prod(1+z)-1)
    return mc(smb), mc(hml)


def panel_bundle(suf):
    wd = pd.read_csv(OUT/f"wml_daily{suf}.csv", parse_dates=["Date"]).set_index("Date")
    wm = pd.read_csv(OUT/f"wml_monthly{suf}.csv", parse_dates=["month"]).set_index("month")
    ffd = pd.read_csv(OUT/f"ff_own_daily{suf}.csv", parse_dates=["Date"]).set_index("Date")
    ffm = pd.read_csv(OUT/f"ff_own_monthly{suf}.csv", parse_dates=["month"]).set_index("month")
    ws, scale, sigma_hat = wml_star(wd, wm)
    rv = realized_vol(wd)
    dec = decompose(wd["wml"], ffd["rmrf"])
    common = pd.concat([wm["wml"], ws], axis=1).dropna()
    common.columns = ["wml", "ws"]
    t3 = {"WML": desc(common["wml"]), "WMLs": desc(common["ws"])}
    t3["WMLs"]["ir"] = info_ratio(common["wml"], common["ws"])
    worst = (wm["wml"].sort_values().head(6)*100).round(1)
    return {
        "wml": ser(wm["wml"]), "rmrf": ser(ffm["rmrf"]), "wml_star": ser(ws),
        "sigma_hat": ser(sigma_hat, 6),
        "rv": ser(rv, 1),
        "decomp": [[d.strftime("%Y-%m"), round(float(r.frac), 3), round(float(r.rvw), 1)] for d, r in dec.iterrows()],
        "decomp_mean_frac": round(float(dec["frac"].mean()), 3),
        "quint_wml": quintiles(wd["wml"], wm["wml"]),
        "quint_rmrf": quintiles(ffd["rmrf"], ffm["rmrf"]),
        "table3": t3,
        "worst": [[d.strftime("%Y-%m"), float(v)] for d, v in worst.items()],
        "range": [wm.index.min().strftime("%Y-%m"), wm.index.max().strftime("%Y-%m")],
        "n": len(wm),
    }


def main():
    out = {}
    out["main"] = panel_bundle("")
    out["clean"] = panel_bundle("_clean")

    # Table 1 (main long panel): all four factors from the SAME canonical Korean-origin
    # source (build_ff_factors.py -> ff_own_monthly; RMRF/SMB/HML) + wml_monthly (WML).
    ffm = pd.read_csv(OUT/"ff_own_monthly.csv", parse_dates=["month"]).set_index("month")
    wm = pd.read_csv(OUT/"wml_monthly.csv", parse_dates=["month"]).set_index("month")
    def with_range(s):
        s = s.dropna(); d = desc(s); d["range"] = f"{s.index.min():%Y}-{s.index.max():%Y}"; d["n"] = len(s); return d
    out["table1"] = {
        "RMRF": with_range(ffm["rmrf"]),
        "SMB": with_range(ffm["smb"]),
        "HML": with_range(ffm["hml"]),
        "WML": with_range(wm["wml"]),
    }
    # US originals (Barroso & Santa-Clara 2015, Table 1 & 3; 1927:03-2011:12)
    out["us"] = {
        "table1": {
            "RMRF": dict(mx=38.27, mn=-29.04, mean=7.33, std=18.96, kurt=7.35, skew=0.17, sharpe=0.39),
            "SMB": dict(mx=39.04, mn=-16.62, mean=2.99, std=11.52, kurt=21.99, skew=2.17, sharpe=0.26),
            "HML": dict(mx=35.48, mn=-13.45, mean=4.50, std=12.38, kurt=15.63, skew=1.84, sharpe=0.36),
            "WML": dict(mx=26.18, mn=-78.96, mean=14.46, std=27.53, kurt=18.24, skew=-2.47, sharpe=0.53),
        },
        "table3": {
            "WML": dict(mx=26.18, mn=-78.96, mean=14.46, std=27.53, kurt=18.24, skew=-2.47, sharpe=0.53),
            "WMLs": dict(mx=21.95, mn=-28.40, mean=16.50, std=16.95, kurt=2.68, skew=-0.42, sharpe=0.97, ir=0.78),
        },
    }
    (WEB/"data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"[out] {WEB}/data.json  ({(WEB/'data.json').stat().st_size/1024:.0f} KB)")
    print("Table1(KR):", {k: (round(v["sharpe"], 2), v["range"]) for k, v in out["table1"].items()})
    print("Table3(KR main): WML sharpe", round(out["main"]["table3"]["WML"]["sharpe"], 2),
          "-> WML* sharpe", round(out["main"]["table3"]["WMLs"]["sharpe"], 2),
          "IR", round(out["main"]["table3"]["WMLs"]["ir"], 2))
    print("Table3(KR clean): WML", round(out["clean"]["table3"]["WML"]["sharpe"], 2),
          "-> WML*", round(out["clean"]["table3"]["WMLs"]["sharpe"], 2),
          "IR", round(out["clean"]["table3"]["WMLs"]["ir"], 2))


if __name__ == "__main__":
    main()
