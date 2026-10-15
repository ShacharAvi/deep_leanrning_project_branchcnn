#!/bin/sh
# CBP2016 TAGE-SC-L replay for every trace that has extracted features.
#
#   branchcnn/scripts/build_tage_logs.sh <trace dir> <features dir> <output dir>
#   e.g. branchcnn/scripts/build_tage_logs.sh new_data/extracted/web \
#            branchcnn/data/web/features branchcnn/data/web/tage
#
# Needs build/tage_cbp2025 (run ./build.sh at the repository root). Per trace it writes
# <name>.log.bin (4 bytes per conditional branch, in order: mispredicted, predicted
# direction, HitBank, confidence bits high | low<<1 | alt<<2), <name>_per_pc.csv and
# <name>.json (totals).
set -e
TRACES="${1:?trace dir}"; FEATURES="${2:?features dir}"; OUT="${3:?output dir}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$OUT"
for f in "$FEATURES"/*.npz; do
  n=$(basename "$f" .npz)
  [ -s "$OUT/$n.log.bin" ] && continue
  "$ROOT/build/tage_cbp2025" "$TRACES/$n.gz" --log "$OUT/$n.log.bin" \
      --per-pc "$OUT/${n}_per_pc.csv" > "$OUT/$n.json"
done
