#!/usr/bin/env bash
# Time-split pipeline check (Section 2 of the report): train on the first 60% of each
# trace, test on the last 30%. Order: media -> fp -> infra -> web.
#
# Variants per category:
#   tokens            plain BranchNet-big (our version), the baseline
#   tage_raw          BranchNet-big + TAGE-SC-L internal state + fetch-safe raw features
#   tage_raw_shuffled CONTROL (fp only): feature rows permuted across instances
#
# Every stage is resumable; re-running this script continues where it stopped.
# Progress: results/cbp2025/<category>/pipeline.log; report: results/cbp2025/SUMMARY.md
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

# media first: the smallest category, used as the pipeline check
$PY scripts/cbp2025_pipeline.py --category media --traces 'new_data/media/*_trace.gz' \
    --min-exec 1000 --min-miss 50 --variants tokens,tage_raw,tage_raw_shuffled --max-epochs 10

COMMON="--max-epochs 8 --patience 2 --max-train-samples 500000 --max-val-samples 300000 --top-k 8"

$PY scripts/cbp2025_pipeline.py --category fp \
    --traces 'new_data/extracted/fp/*_trace.gz' \
    --variants tokens,tage_raw,tage_raw_shuffled $COMMON
$PY tools/cbp2025_summary.py > /dev/null

for cat in infra web; do
  $PY scripts/cbp2025_pipeline.py --category $cat \
      --traces "new_data/extracted/$cat/*_trace.gz" \
      --variants tokens,tage_raw $COMMON
  $PY tools/cbp2025_summary.py > /dev/null
done
echo "ALL DONE $(date)"
