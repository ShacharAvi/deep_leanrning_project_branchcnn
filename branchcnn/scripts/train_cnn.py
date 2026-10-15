"""Train the branch direction CNN on extracted media features.

    # smoke run
    python scripts/train_cnn.py --max-batches 50 --max-epochs 2

    # history-only baseline, then the full feature set
    python scripts/train_cnn.py --no-scalars --run-name history_only
    python scripts/train_cnn.py --run-name all_features

    # leakage control: must land on the bias floor
    python scripts/train_cnn.py --shuffle-labels --run-name shuffled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    DEFAULT_SPLIT,
    NUM_SCALARS,
    NUM_SEQ_NUMERIC,
    BranchDataset,
    DatasetConfig,
    SplitSpec,
    check_split,
)
from src.model import ModelConfig  # noqa: E402
from src.train import TrainConfig, baselines, train  # noqa: E402

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
RESULT_DIR = PROJECT_ROOT / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURE_DIR)
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--out", type=Path, default=RESULT_DIR)

    parser.add_argument("--history-length", type=int, default=64)
    parser.add_argument("--pc-bits", type=int, default=14)
    parser.add_argument(
        "--max-train-per-trace",
        type=int,
        default=None,
        help="cap training samples per trace; validation and test are never capped",
    )

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--token-embed", type=int, default=32)
    parser.add_argument("--pc-embed", type=int, default=32)
    parser.add_argument(
        "--conv-channels", default="64,64,128", help="comma-separated"
    )
    parser.add_argument("--head-dims", default="256,64", help="comma-separated")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--no-sequence", action="store_true", help="drop the CNN input")
    parser.add_argument("--no-scalars", action="store_true", help="history-only run")
    parser.add_argument("--no-pc", action="store_true", help="drop the PC embedding")
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "families per role as 'train/validation/test', comma-separated "
            "within a role, e.g. 'A/B/C'. Defaults to A,B,C/D,E/F,G, which "
            "needs all 14 traces extracted."
        ),
    )
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="leakage control; must land on the bias floor",
    )
    parser.add_argument("--tage", action="store_true", help="add TAGE-SC-L's decision as an input")
    parser.add_argument("--tage-dir", type=Path, default=PROJECT_ROOT / "data" / "tage")
    parser.add_argument(
        "--tage-mode", choices=("both", "feature", "residual"), default="both",
        help="feature: TAGE vector into the FC head; residual: learnable weight on TAGE's vote; both",
    )
    parser.add_argument("--tage-init", type=float, default=1.0, help="initial TAGE weight")
    parser.add_argument(
        "--shuffle-tage", action="store_true",
        help="control: permute TAGE rows across instances; must remove any gain",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="report reference predictors and exit without training",
    )
    return parser


def _ints(text: str) -> tuple:
    return tuple(int(part) for part in text.split(",") if part.strip())


def _split(text: Optional[str]) -> SplitSpec:
    """Parse 'train/validation/test' family groups into a SplitSpec."""
    if text is None:
        return DEFAULT_SPLIT
    roles = [part.strip() for part in text.split("/")]
    if len(roles) != 3:
        raise SystemExit(
            f"--split needs three groups separated by '/', got {text!r}"
        )
    spec = SplitSpec(
        *(
            tuple(f.strip().upper() for f in role.split(",") if f.strip())
            for role in roles
        )
    )
    check_split(spec)
    return spec


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    dataset_config = DatasetConfig(
        history_length=args.history_length,
        pc_bits=args.pc_bits,
        max_train_samples_per_trace=args.max_train_per_trace,
        seed=args.seed,
        use_tage=args.tage,
        tage_dir=args.tage_dir if args.tage else None,
        shuffle_tage=args.shuffle_tage,
    )
    dataset = BranchDataset(args.features, dataset_config, _split(args.split))
    print(f"loading features from {args.features}")
    dataset.load()

    description = dataset.describe()
    for role, info in description["splits"].items():
        print(
            f"  {role:11s} families {','.join(info['families']):6s} "
            f"{len(info['traces']):2d} traces  {info['samples']:>10,} samples  "
            f"taken {info['taken_rate']:.4f}"
        )
    if dataset.splits.get("train") is None or not dataset.splits["train"].traces:
        print("No training data found. Has scripts/build_features.py finished?")
        return 2

    for role in ("validation", "test"):
        if dataset.splits.get(role) and dataset.splits[role].traces:
            stats = baselines(dataset, role)
            print(
                f"  {role:11s} baselines: majority {stats['majority_accuracy']:.4f}  "
                f"always-taken {stats['always_taken_accuracy']:.4f}  "
                f"2-bit {stats['two_bit_accuracy']:.4f}"
                + (f"  TAGE-SC-L {stats['tage_accuracy']:.4f} (MPKI {stats['tage_mpki']:.3f})" if "tage_accuracy" in stats else "")
            )

    if args.baselines_only:
        return 0

    model_config = ModelConfig(
        vocabulary_size=dataset.vocabulary_size,
        num_pc_ids=dataset.num_pc_ids,
        num_scalars=NUM_SCALARS,
        num_seq_numeric=NUM_SEQ_NUMERIC,
        history_length=args.history_length,
        token_embed_dim=args.token_embed,
        pc_embed_dim=args.pc_embed,
        conv_channels=_ints(args.conv_channels),
        head_dims=_ints(args.head_dims),
        dropout=args.dropout,
        use_sequence=not args.no_sequence,
        use_scalars=not args.no_scalars,
        use_pc=not args.no_pc,
        use_tage=args.tage,
        tage_mode=args.tage_mode,
        tage_init=args.tage_init,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_batches=args.max_batches,
        shuffle_labels=args.shuffle_labels,
        seed=args.seed,
        threads=args.threads,
        device=args.device,
    )

    output_dir = args.out / args.run_name
    result = train(dataset, model_config, train_config, output_dir)

    test = result["test"]
    reference = result["baselines"]["test"]
    print()
    print(f"test accuracy      {test['accuracy']:.4f}  over {test['samples']:,} samples")
    if reference:
        print(f"majority baseline  {reference['majority_accuracy']:.4f}")
        print(f"2-bit baseline     {reference['two_bit_accuracy']:.4f}")
        gain = test["accuracy"] - reference["two_bit_accuracy"]
        print(f"delta vs 2-bit     {gain:+.4f}")
        if "tage_accuracy" in reference:
            print(f"TAGE-SC-L          {reference['tage_accuracy']:.4f}  (MPKI {reference['tage_mpki']:.3f})")
            print(f"delta vs TAGE      {test['accuracy'] - reference['tage_accuracy']:+.4f}  "
                  f"(model MPKI {result['test_mpki_on_scored_branches']:.3f})")
    if result.get("tage_weights"):
        w = result["tage_weights"]
        print(f"learned TAGE weight: global {w['global_scale']:.3f}, per-state range "
              f"[{min(w['per_state_weight']):.3f}, {max(w['per_state_weight']):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
