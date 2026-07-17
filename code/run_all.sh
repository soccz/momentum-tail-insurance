#!/usr/bin/env bash
# End-to-end reproduction pipeline (Korean-data-only).
# Usage: bash code/run_all.sh [main|clean|both]
set -euo pipefail
cd "$(dirname "$0")/.."
PANELS=${1:-both}
[ "$PANELS" = "both" ] && PANELS="main clean"

# one-time: PBR.xlsx -> pbr_monthly.parquet (book-to-market for HML)
[ -f data/processed/pbr_monthly.parquet ] || python3 code/convert_pbr.py

for p in $PANELS; do
  echo "########## PANEL = $p ##########"
  python3 code/build_factor.py     --panel "$p"   # WML (daily+monthly), VW deciles
  python3 code/build_ff_factors.py --panel "$p"   # RMRF, SMB, HML from KR data only
  python3 code/build_factors_rm.py --panel "$p"   # realized vol + WML* (Eq 2,5,6)
  python3 code/build_tables.py     --panel "$p"   # Table 1, 3
  python3 code/build_figures.py    --panel "$p"   # Fig 1, 2, 3, 6
done
echo "DONE. tables -> output/tables/  figures -> output/figures/"
