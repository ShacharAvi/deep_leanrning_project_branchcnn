# Deep Learning for Hard-to-Predict Branches

Course project, Deep Learning 2026 — Shachar Aviran, Itamar Shugarman, Michael Bruchim.

TAGE-SC-L, the strongest conventional branch predictor, is right on more than 99% of
conditional branches. We ask whether deep learning can lower its error rate further, on
program runs never seen in training. Three components:

| component | what it is | where |
|---|---|---|
| **Specialists** | BranchNet-big, one CNN per hard-to-predict branch, re-implemented in PyTorch, with TAGE-SC-L's internal state and fetch-safe trace features as extra inputs | `src/models/branch_models.py` |
| **Generalist** | BranchCNN, one shared CNN for every conditional branch | `branchcnn/src/model.py` |
| **Arbiter** | a per-execution logistic regression that decides which predictor to follow, kept only if it passes an out-of-fold significance test | `src/evaluation/learned_arbiter.py`, `src/evaluation/arbiter.py` |

Main results: MPKI reduction vs TAGE-SC-L 64 KB, on unseen traces, positive = fewer
mispredictions. Every number is in `results/`.

| system | result | file |
|---|---|---|
| Specialists + learned arbiter, 5-fold CV, web_g0 / infra_g0 | +2.37 ± 0.10% / +1.14 ± 1.92% (beats BranchNet's per-branch rule on 10/10 folds) | `results/cbp2025/kfold_table.json` |
| Generalist BranchCNN, fold 1 / fold 2 | +11.12% / +8.01% | `branchcnn/results/repro_fold*/result.json` |
| Generalist + specialists via the arbiter, fold 1 / fold 2 | +11.76% / +8.52% | `results/cbp2025/combined/fold*_tage_raw.json` |
| Same, both trained on 16 traces | **+11.92%** (generalist alone +9.96%) | `results/cbp2025/combined/big16_tage_raw.json` |

## Repository layout

```
build.sh                 builds the four C++ tools into build/
src/                     Track A (specialists) library
  traces/                  SBBT branch-stream parser (format codec + streaming reader)
  dataset/                 per-branch HDF5 datasets, fetch-safe TAGE/raw feature loading
  models/                  BranchNet-big (+ features), MLP/CNN baselines, cost model
  training/                training loop: selection on validation, test evaluated once
  evaluation/              exact hybrid MPKI, BranchNet rule, threshold + learned arbiters
  predictors/              Python reference baselines (cross-validate the C++ driver)
scripts/                 Track A experiments
  cbp2025_pipeline.py      per-trace prep (convert, TAGE-SC-L, features) + time-split check
  cbp2025_crosstrace.py    main experiment: split by trace, select, train, evaluate arbiters
  run_web_g0.sh            pilot, data-volume curve, 43-trace pooling, dataflow producers
  run_cbp2025_kfold.sh     5-fold cross-validation, web_g0 and infra_g0
  run_head_to_head.sh      specialists on BranchCNN's exact splits (fold 1, fold 2, 16 traces)
  run_cbp2025_all.sh       time-split pipeline check over all categories
tools/
  cbp2025_convert.cpp      CBP2025 trace -> SBBT branch stream + fetch-safe raw features
  run_predictors.cpp       TAGE-SC-L 64 KB + baselines; per-PC stats; per-execution state log
  cbp2025_producers.cpp    dataflow-producer features (report Section 5.5)
  cbp2025_groups.py        groups traces into programs (web_g0, infra_g0, ...)
  combine_tracks.py        generalist + specialists through the arbiter (report Section 5.4)
  cbp2025_feature_screen.py, cbp2025_gap_analysis.py, cbp2025_shared_arbiter.py,
  cbp2025_figures.py, report_figures.py, cbp2025_summary.py, classify_h2p.py
branchcnn/               Track B (generalist)
  src/                     trace decoding, causal feature extraction, dataset, model, training
  scripts/                 build_features.py, build_tage_logs.sh, train_cnn.py, dump_predictions.py
  tools/tage_cbp2025.cpp   CBP2016 TAGE-SC-L 64 KB replay (the generalist's baseline + inputs)
  results/<run>/           trained BranchCNN weights and metrics for the three report runs
configs/splits/          BranchCNN's web splits, used by both tracks
results/                 every result in the report (see "Results" below)
models/                  trained specialists of the best configuration (web_g0, 43 traces)
third_party/             MBPlib, spec_tage_scl (TAGE-SC-L), CBP2016 TAGE-SC-L (see bottom)
tests/, branchcnn/tests/ automated tests
```

**The two TAGE-SC-L copies are deliberate.** Each track uses a standard 64 KB TAGE-SC-L.
The specialists use the Scarab-derived `spec_tage_scl`, and the generalist uses the CBP2016
reference shipped with the CBP2025 framework. Every result is reported against the track's
own copy. On the same test traces the two agree within 0.5% MPKI (4.832 vs 4.836 and 3.802
vs 3.782).

## Setup

Requirements:
- Python 3.11;
- a C++17 compiler;
- zlib;
- `zstd` on `PATH`, used by the SBBT reader.

Training used Apple-silicon GPUs (MPS); CUDA and CPU also work.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./build.sh                                  # -> build/{cbp2025_convert,run_predictors,cbp2025_producers,tage_cbp2025}
.venv/bin/python -m pytest                  # Track A tests
(cd branchcnn && ../.venv/bin/python -m pytest tests)   # Track B tests
```

Five Track A tests skip unless the CBP-5 trace `SHORT_MOBILE-73.sbbt.zst` is placed under
`data/`. They cross-check the SBBT parser and the C++ driver on a real trace. Everything
else is self-contained.

## Data

The data is the **CBP2025 (Championship Branch Prediction 2025) instruction traces**:
- 386 traces: media 14, fp 61, infra 102, web 209;
- about 75 GB compressed.

They are not included. Download them from the CBP2025 trace folder linked in the official
CBP2025 framework README:
<https://drive.google.com/drive/folders/10CL13RGDW3zn-Dx7L0ineRvl7EpRsZDW> (for example
`pip install gdown && gdown --folder <link>`, then `tar -xf <category>.tar.xz`).

Expected layout, relative to the repository root:

```
new_data/media/media_<n>_trace.gz
new_data/extracted/{fp,infra,web}/<category>_<n>_trace.gz
```

Everything derived from the traces is written under `data/`, `experiments/` and
`branchcnn/data/`, all git-ignored. The program groups we used are shipped in
`results/cbp2025/groups.json`. The report's experiments use web_g0 (47 web traces) and
infra_g0 (34 infra traces).

## Reproducing the report

Every stage is resumable, and every command runs from the repository root unless stated.
Approximate times are for an M-series Mac.

**1. Per-trace preparation.** This converts each trace, runs TAGE-SC-L and builds the
fetch-safe features. It takes a few hours for web and infra, all on CPU.

```bash
.venv/bin/python scripts/cbp2025_pipeline.py --category web   --traces 'new_data/extracted/web/*_trace.gz'   --stages prep
.venv/bin/python scripts/cbp2025_pipeline.py --category infra --traces 'new_data/extracted/infra/*_trace.gz' --stages prep
.venv/bin/python tools/cbp2025_groups.py     # -> results/cbp2025/groups.json (the shipped file is this output)
```

**2. Specialists (Track A).**

| report | command | output (`results/cbp2025/`) |
|---|---|---|
| 5.2, 5.5, 6.2: pilot, data volume, pooling, producers, headroom | `scripts/run_web_g0.sh` | `crosstrace/web_g0*.json`, `screen/web_g0.json` |
| 5.3: 5-fold cross-validation | `scripts/run_cbp2025_kfold.sh`, then `tools/cbp2025_figures.py` | `crosstrace/{web,infra}_g0_f*.json`, `kfold_table.json`, `figures/` |
| fold 0 of infra_g0 | `.venv/bin/python scripts/cbp2025_crosstrace.py --group infra_g0 --variants tokens,tage_raw,tage_raw_shuffled,tokens_labelshuffle --max-epochs 8 --patience 2 --max-train-samples 2000000` | `crosstrace/infra_g0.json` |
| 5.2, 5.4: specialists on BranchCNN's splits | `scripts/run_head_to_head.sh` | `crosstrace/bcnn_web_{f1,f2,big}.json` |

Variants:
- `tokens` is plain BranchNet-big.
- `tage_raw` adds TAGE-SC-L's state and the raw trace features.
- `*_shuffled` permutes the feature rows across executions.
- `*_labelshuffle` trains on permuted labels.

The last two are the leakage controls. With shuffled labels, a clean pipeline must
score exactly 0.00%, because the arbiter's significance test rejects every model.

**3. Generalist (Track B).** Run these from `branchcnn/`. The 16-trace run needs about 16 GB
of RAM and about 1.3 h per epoch.

```bash
cd branchcnn
W=../new_data/extracted/web
for n in 2 3 4 5 8 27 31 32 33 34 47 53 75 80 81 82 90 105 106 124 128; do
  ../.venv/bin/python scripts/build_features.py --trace $W/web_${n}_trace.gz --out data/web/features
done
sh scripts/build_tage_logs.sh $W data/web/features data/web/tage

F1="W3,W5,W8,W27/W31/W32,W33"
F2="W3,W5,W8,W27,W32/W33/W31"
BIG="W3,W5,W8,W27,W106,W82,W53,W81,W80,W105,W124,W34,W47,W4,W75,W90/W31,W128,W2/W32,W33"
for job in "repro_fold1 $F1" "repro_fold2 $F2" "big16 $BIG"; do
  set -- $job
  ../.venv/bin/python scripts/train_cnn.py --device mps --features data/web/features \
      --tage-dir data/web/tage --split "$2" --run-name $1 \
      --tage --tage-mode both --tage-init 2.0 --max-epochs 6
  ../.venv/bin/python scripts/dump_predictions.py --run results/$1 \
      --features data/web/features --tage-dir data/web/tage --split "$2"
done
cd ..
```

Two checks:
- The first lines of each training log must report the expected number of training
  traces (4, 5 or 16). The loader silently skips traces whose feature files are missing.
- Without `--tage`, the same command gives the "without TAGE-SC-L state" result of
  Section 5.1.

**4. Combination (report Section 5.4).** This needs steps 2 (head-to-head) and 3.

```bash
.venv/bin/python tools/combine_tracks.py --fold 1 2                     # -> combined/fold{1,2}_tage_raw.json
.venv/bin/python tools/combine_tracks.py --fold 1 2 --variant tokens    # history-only specialists
.venv/bin/python tools/combine_tracks.py --fold 0 --variant tage_raw --run big16 --xt-group bcnn_web_big --tag big16
.venv/bin/python tools/combine_tracks.py --fold 0 --variant tokens   --run big16 --xt-group bcnn_web_big --tag big16
```

System A is the arbiter between TAGE-SC-L and BranchCNN. B is the arbiter between
BranchCNN and the specialists. C is both arbiters in sequence. Each output file holds all
three, their oracle ceilings and the per-branch counts.

**5. Figures.** These run directly on the shipped results:

```bash
.venv/bin/python tools/cbp2025_figures.py    # -> results/cbp2025/figures/, kfold_table.json
.venv/bin/python tools/report_figures.py     # -> results/figures/ (report Figures 2-4)
```

## Results

`results/` holds the result files behind every number in the report. Re-running an
experiment overwrites the corresponding file in place.

| path | content |
|---|---|
| `results/cbp2025/groups.json` | program groups and their member traces |
| `results/cbp2025/crosstrace/web_g0_learned.json` | web_g0 pilot: every feature variant and control, all decision rules |
| `results/cbp2025/crosstrace/web_g0.json` | data-volume curve (`#t1` … `#t16`) |
| `results/cbp2025/crosstrace/web_g0_pool*.json` | 43 training traces; producers (`_prod`, `_arbprod*`); shared arbiters |
| `results/cbp2025/crosstrace/{web,infra}_g0_f*.json`, `kfold_table.json` | 5-fold cross-validation |
| `results/cbp2025/crosstrace/bcnn_web_*.json` | specialists on BranchCNN's splits |
| `results/cbp2025/combined/` | generalist + specialists combination (fold 1, fold 2, 16 traces) |
| `results/cbp2025/screen/web_g0.json` | dataflow-producer screening and the deliberate-leak control |
| `results/cbp2025/media/` | time-split pipeline check |
| `results/branchcnn_published/` | the BranchCNN study's own runs: per-run JSONs, tables, breakdowns, figures |
| `results/figures/`, `results/cbp2025/figures/` | the figures in the report |

## Using the trained models

`models/specialists_web_g0_pool/<pc>/` holds one specialist (`tage_raw`, 43 training traces)
per hard branch:
- the checkpoint;
- its metrics;
- the per-execution validation and test logits (`val.npz` / `test.npz`, with TAGE-SC-L's
  correctness and confidence).

The arbiter is fitted on the validation file and scored on the test file. Loading one:

```python
import torch
from src.models import build_model

ckpt = torch.load("models/specialists_web_g0_pool/0xaaaaab3b3908/checkpoint.pt",
                  map_location="cpu", weights_only=False)
cfg, state = ckpt["config"], ckpt["state_dict"]
model = build_model(cfg["model"], vocabulary_size=1 << (cfg["pc_hash_bits"] + 1),
                    history_length=cfg["history_length"],
                    num_features=state["feature_mask"].numel(),
                    feature_mask=state["feature_mask"].bool())
model.load_state_dict(state)
model.eval()
# model(tokens, features) -> one logit per sample; tokens: (batch, 582) branch-history ids,
# features: (batch, 69) TAGE-SC-L state + raw features, built by src.dataset.BranchDataset
```

`branchcnn/results/<run>/model.pt` are the three BranchCNN runs of the report.
`dump_predictions.py` regenerates their per-execution predictions from the features.

## Third-party code

| path | origin | use |
|---|---|---|
| `third_party/mbplib/` | MBPlib, <https://github.com/useredsa/MBPlib> (headers and `src/{core,sim}` only) | SBBT reader, simulator harness, baseline predictors |
| `third_party/spec_tage_scl/` | `spec_tage_scl`, TAGE-SC-L from the Scarab simulator with an MBPlib interface (`tagescl/` headers only) | Track A's TAGE-SC-L 64 KB |
| `third_party/cbp2016_tage_sc_l/` | `cbp2016_tage_sc_l.h` from the CBP2025 framework (A. Seznec's CBP2016 winner) | Track B's TAGE-SC-L 64 KB |

Three small portability changes are applied to the vendored headers; behaviour is unchanged:
- `constexpr` → `const` for `pow()`-initialised history sizes in `tage.hpp`;
- a `constexpr` `sqrt`/`pow` in MBPlib's `hashed_perceptron.hpp`, pinned by a `static_assert`
  in `tools/run_predictors.cpp`;
- value-initialised predictor tables.

BranchNet's original code (<https://github.com/siavashzk/BranchNet>) was used only as a
reference for its configuration and is not included. It needs Linux, Intel Pin and the SPEC
benchmarks. Our BranchNet-big is a re-implementation; the differences are listed in
Section 4 of the report.
