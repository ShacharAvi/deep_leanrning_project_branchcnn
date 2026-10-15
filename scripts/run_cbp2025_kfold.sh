#!/bin/sh
# k-fold over TRACES (Tarsa's protocol) for the two largest program groups.
# Fold 0 is the original split; folds 1-4 rotate which traces are test (the
# permutation is shifted by fold * n_test, so test sets are disjoint), and each
# fold re-selects its branches from ITS training traces only.
# Resumable; aggregate with tools/cbp2025_figures.py (writes results/cbp2025/kfold_table.json).
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
V=tokens,tage_raw,tokens_labelshuffle
for G in web_g0 infra_g0; do
  for K in 1 2 3 4; do
    $PY scripts/cbp2025_crosstrace.py --group $G --fold $K \
      --stages setup,train,eval --variants $V \
      --max-epochs 8 --patience 2 --max-train-samples 2000000
    OMP_NUM_THREADS=6 $PY tools/cbp2025_shared_arbiter.py \
      --group ${G}_f$K --variants $V
  done
done
echo "KFOLD DONE $(date)"
