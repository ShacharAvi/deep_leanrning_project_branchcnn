#!/bin/sh
# Specialists on BranchCNN's exact web splits (report Sections 5.2 and 5.4): fold 1 and
# fold 2 (4-5 training traces) and the 16-trace split. The split files are in
# configs/splits/. Afterwards run the BranchCNN side (README, "Track B") and then
# tools/combine_tracks.py.
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
V=tokens,tage_raw,tokens_labelshuffle
for k in 1 2; do
  $PY scripts/cbp2025_crosstrace.py --group bcnn_web_f$k \
      --split-json configs/splits/branchcnn_web_fold$k.json --min-traces 2 --variants $V \
      --max-epochs 8 --patience 2 --max-train-samples 2000000
done
$PY scripts/cbp2025_crosstrace.py --group bcnn_web_big \
    --split-json configs/splits/branchcnn_web_big.json --variants $V \
    --max-epochs 8 --patience 2 --max-train-samples 2000000
echo "HEAD-TO-HEAD DONE $(date)"
