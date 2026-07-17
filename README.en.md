<div align="center">

# What Improves When Risk Forecasts Improve

### Machine-Learned Volatility and Tail Insurance in a Weak Momentum Market

[![License: MIT](https://img.shields.io/badge/License-MIT-1c1917.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-44403c.svg)](requirements.txt)
[![Experiments](https://img.shields.io/badge/Ledger-E00–E38-b45309.svg)](docs/05_results_ledger.md)
[![headline gates](https://img.shields.io/badge/headline_gates-11%2F11_PASS-15803d.svg)](tests/verify_headline_numbers.py)

**[🧭 Research journey (all successes, failures, retractions)](https://soccz.github.io/projects/momentum-journey/)** ·
**[🎛️ Interactive BS15 replication demo](https://soccz.github.io/projects/momentum-barroso/)** ·
**[🇰🇷 한국어](README.md)**

</div>

---

Volatility-scaled momentum (Barroso and Santa-Clara, 2015, *JFE*) runs on a single input: a risk forecast.
This project replaces that input with **twelve pre-specified predictors** — from rolling variance to HAR,
regularized regressions, tree ensembles, and neural networks — and measures what improves when the forecast does,
using 35 years of Korean stock data (1991–2026) including 1,842 delisted firms.

<div align="center">
<img src="output/figures/p3_transfer_curve.png" width="88%" alt="Transfer curve">
<br><sub><b>The transfer curve.</b> Better forecasts do not move the Sharpe ratio (left). The improvement shows up in the tail (right) — pairwise against the benchmark.</sub>
</div>

## Three findings

1. **Forecasts improve, Sharpe ratios do not follow.** The ensemble cuts QLIKE by 24% (Clark–West t=3.4),
   yet the slope of the Sharpe–accuracy transfer curve is indistinguishable from zero
   (joint block-bootstrap CI [−0.006, +0.006]).
2. **The improvement shows up in the tail.** Monthly expected shortfall improves by 0.86pp; the gain survives
   Romano–Wolf corrections at pre-specified predictors (HAR, Lasso), extends to the 1% extreme tail under EVT,
   and persists net of transaction taxes and borrowing fees.
3. **The insurance holds only in a narrow domain — and it is sold by predictor structure, not ML accuracy.**
   The gain exists only at the intersection of a weak factor and extreme decile construction
   (Korea–US difference test D=+2.01pp, p=0.001); most of it is captured by long-memory (HAR) and
   market-state structure, with zero incremental tail gain from ML combinations (ENS−HAR = −0.25pp, n.s.).

Across eight international/construction contrasts (US extreme deciles, US small-caps, Japan/Europe/North America/
Asia-Pacific 2×3, Korea 2×3 ablation), **forecasts improve everywhere (QLIKE 10–34%) but the tail responds only
in Korea's extreme-decile construction** — the contrapositive test that became the paper's boundary-condition section.

## Why this repository is unusual: the whole process is on the record

The **[results ledger](docs/05_results_ledger.md)** (E00–E38, append-only) preserves every failure, retraction,
and correction — neural nets collapsing (OOS R² of −439%), a discarded "resurrecting dead factors" frame,
a favorable-but-wrong family-wise p-value we retired, the pre-specified winner failing its own FWER correction,
and a cross-market "weak-factor law" that died in six markets. See the
[Hall of Failures](https://soccz.github.io/projects/momentum-journey/#graveyard).

The manuscript was closed by a **blind gauntlet**: five reviewer lenses (prose, journal referee, full numeric audit,
structure, thesis defense) × adversarial verification × seven rounds. Confirmed findings could only be closed by
**computation**, never by rewording — the final rounds alone added six new tests (p25–p30). Final verdict:
zero confirmed defects, zero numeric mismatches across ~200 quantitative claims.

## Quick start

```bash
pip install -r requirements.txt

# Headline gate: 11 core claims must reproduce from the canonical CSVs (re-runnable by anyone in seconds)
python tests/verify_headline_numbers.py

# Full analysis pipeline reproduces from this repository alone
# (factor-level derived series are included; only security-level raw data is excluded)
python code/p2_forecast_race.py          # 12-predictor forecast race
python code/p30_kr_us_difference.py      # Korea–US tail-increment difference test (live FF download)
```

Every `p*.py` uses seed=42 and prints its own verification gate on completion.
Paths are hard-coded to the original project root — edit the `ROOT` line to rerun.

## Data policy

| Data | Source | Here | Note |
|---|---|---|---|
| Korean security-level prices/caps/PBR | FnGuide DataGuide (subscription) | ❌ | Not redistributable — verify identity via SHA256 in [`reproduce/DATA_MANIFEST.txt`](reproduce/DATA_MANIFEST.txt) |
| Factor-level derived series (WML, factors, features, forecasts) | aggregated transforms of the above | ✅ | Portfolio-level only — same practice as Ken French publishing CRSP-derived factors |
| Risk-free rate (call rate) | Bank of Korea / FRED | ✅ | `reproduce/step0` re-downloads from FRED and matches to 1e-16 |
| US/international momentum & factors | Kenneth R. French Data Library | ✅ | Public — the entire international parallel reproduces from this repo |
| Full paper (Korean, v4.2) | — | ⏳ | Released after the thesis defense; until then every claim is verifiable via the [claims map](docs/06_claims_map.md) and [canonical CSVs](output/tables/) |

## Citation

```bibtex
@misc{soccz2026momentum,
  title  = {Momentum Tail Insurance: What Improves When Risk Forecasts Improve},
  author = {soccz},
  year   = {2026},
  url    = {https://github.com/soccz/momentum-tail-insurance},
  note   = {Working paper; full text to be released after thesis defense}
}
```

Original paper: Barroso, P., and P. Santa-Clara (2015), *Momentum has its moments*, JFE 116(1), 111–120.
Prior Korean BS15 replication: Son, Yoon, and Yoon (2017), Journal of Knowledge Studies 15(1).
