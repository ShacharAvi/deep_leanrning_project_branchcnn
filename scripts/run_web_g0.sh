#!/bin/sh
# Specialists on web_g0 (report Sections 5.2, 5.3, 5.5, 6.2): pilot split, data-volume
# curve, pooled 43-trace coverage, dataflow producers. Prerequisites: ./build.sh, the web
# traces under new_data/extracted/web/, and the prep + grouping steps of the README.
# Every stage is resumable.
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
COMMON="--max-epochs 8 --patience 2"
ALL=tokens,tokens@a3,tage,raw_path,raw_committed,tage_raw,tage_raw@a3,tage_raw_shuffled,tokens_labelshuffle

# 1. pilot: 29 train / 9 validation / 9 test traces, every feature variant and control
$PY scripts/cbp2025_crosstrace.py --group web_g0 --stages setup
$PY scripts/cbp2025_crosstrace.py --group web_g0 --stages train,eval --variants $ALL \
    $COMMON --max-train-samples 1000000 --results-tag _learned
# 2. data-volume curve: the same 9 test traces, 1..16 training traces
$PY scripts/cbp2025_crosstrace.py --group web_g0 --variants tokens,tage_raw \
    --max-train-traces 1 2 4 8 16 $COMMON --max-train-samples 1000000
# 3. pooled coverage: 43 training traces (validation and test unchanged)
$PY scripts/cbp2025_crosstrace.py --group web_g0 --pool --variants tokens,tage_raw \
    $COMMON --max-train-samples 2000000
# 4. dataflow producers: screening alone, inside BranchNet-big, and as arbiter inputs
$PY tools/cbp2025_feature_screen.py --group web_g0
$PY scripts/cbp2025_crosstrace.py --group web_g0 --pool --results-tag _prod $COMMON \
    --max-train-samples 2000000 \
    --variants tage_raw_prod1024,tage_raw_prod256,tage_raw_prod1024_shuffled,tage_raw_prod1024_labelshuffle
V=tokens,tage_raw
$PY tools/cbp2025_shared_arbiter.py --group web_g0 --variants tokens,tage_raw,tage_raw_shuffled,tokens_labelshuffle
$PY tools/cbp2025_shared_arbiter.py --group web_g0_pool --variants $V
$PY tools/cbp2025_shared_arbiter.py --group web_g0_pool --variants $V --arbiter-extra prodfeat_L1024 --tag _arbprod1024
$PY tools/cbp2025_shared_arbiter.py --group web_g0_pool --variants $V --arbiter-extra prodfeat_L1024 --shuffle-extra --tag _arbprod1024_shuf
# 5. headroom (Section 6.2)
$PY tools/cbp2025_gap_analysis.py --group web_g0 --variant tage_raw
echo "WEB_G0 DONE $(date)"
