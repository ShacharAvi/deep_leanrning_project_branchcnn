#!/bin/sh
# Build the four C++ tools into build/. Needs a C++17 compiler, zlib, and zstd on PATH
# (the SBBT reader decompresses .sbbt.zst through `zstd`).
set -e
cd "$(dirname "$0")"
CXX="${CXX:-c++}"
mkdir -p build
MB=third_party/mbplib
MB_SRC="$MB/src/core/branch.cpp $MB/src/core/predictor.cpp $MB/src/sim/sbbt_reader.cpp $MB/src/sim/sbbt_writer.cpp $MB/src/sim/simulator.cpp"

# Track A: CBP2025 trace -> SBBT branch stream + fetch-safe raw features
$CXX -std=c++17 -O3 -I $MB/include tools/cbp2025_convert.cpp -lz -o build/cbp2025_convert
# Track A: TAGE-SC-L 64 KB (spec_tage_scl) + baselines, per-PC stats and per-execution state log
$CXX -std=c++17 -O3 -DTAGE_SC_L_SIZE=64 -I $MB/include -I third_party/spec_tage_scl/include \
    tools/run_predictors.cpp $MB_SRC -o build/run_predictors
# Track A: dataflow-producer features (Section 5.5)
$CXX -std=c++17 -O3 -I $MB/include tools/cbp2025_producers.cpp -lz -o build/cbp2025_producers
# Track B: CBP2016 TAGE-SC-L 64 KB replay, the generalist's baseline and TAGE inputs
$CXX -std=c++17 -O3 -I third_party/cbp2016_tage_sc_l branchcnn/tools/tage_cbp2025.cpp -lz \
    -o build/tage_cbp2025
echo "built: $(ls build | tr '\n' ' ')"
