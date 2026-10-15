"""Extract causal CNN features from CBP2025 traces.

One ``.npz`` per trace under ``data/features/``. Traces are streamed, so memory
stays flat regardless of trace length.

    python scripts/build_features.py --workload media
    python scripts/build_features.py --trace media_0_trace --limit 2000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import (  # noqa: E402
    ExtractionConfig,
    extract_trace,
    save_features,
)
from src.trace_registry import discover_traces, resolve_trace  # noqa: E402

FEATURE_DIR = PROJECT_ROOT / "data" / "features"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", help="extract every trace of this workload")
    parser.add_argument("--trace", action="append", default=[], help="repeatable")
    parser.add_argument("--list", action="store_true", help="list traces and exit")
    parser.add_argument("--limit", type=int, help="stop after N instructions")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--window", type=int, default=1 << 16)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=FEATURE_DIR, help="output directory"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-extract traces already done"
    )
    parser.add_argument("--progress-every", type=int, default=2_000_000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    available = discover_traces()
    if args.list:
        for ref in available:
            print(f"{ref.workload_id}/{ref.trace_id}  {ref.size_bytes / 1e6:.0f} MB")
        return 0

    selected = []
    if args.workload:
        selected.extend(r for r in available if r.workload_id == args.workload)
    for name in args.trace:
        selected.append(resolve_trace(name))
    if not selected:
        print("No traces selected. Use --workload or --trace, or --list.")
        return 2

    seen = set()
    unique = []
    for ref in selected:
        key = (ref.workload_id, ref.trace_id)
        if key not in seen:
            seen.add(key)
            unique.append(ref)

    config = ExtractionConfig(
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        window_capacity=args.window,
        sample_stride=args.sample_stride,
        warmup_instructions=args.warmup,
        instruction_limit=args.limit,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    print(f"Extracting {len(unique)} trace(s) -> {args.out}")
    print(
        f"config: depth<={config.max_depth} nodes<={config.max_nodes} "
        f"window={config.window_capacity} stride={config.sample_stride}"
    )

    for position, ref in enumerate(unique, start=1):
        target = args.out / f"{ref.trace_id}.npz"
        if target.exists() and not args.overwrite:
            print(f"[{position}/{len(unique)}] {ref.trace_id}: exists, skipping")
            continue

        print(f"[{position}/{len(unique)}] {ref.trace_id}: starting", flush=True)
        started = time.time()
        features = extract_trace(
            ref.path,
            ref.workload_id,
            ref.trace_id,
            config,
            progress_every=args.progress_every,
        )
        save_features(features, target)
        elapsed = time.time() - started

        entry = {
            "workload_id": ref.workload_id,
            "trace_id": ref.trace_id,
            "instructions": features.instructions_seen,
            "branch_slots": features.num_branches,
            "samples": features.num_samples,
            "taken_rate": round(features.taken_rate, 6),
            "distinct_pcs": int(len(set(features.s_pc.tolist()))),
            "seconds": round(elapsed, 1),
            "megabytes": round(target.stat().st_size / 1e6, 1),
        }
        manifest.append(entry)
        print(
            f"[{position}/{len(unique)}] {ref.trace_id}: "
            f"{entry['instructions']:,} instr, {entry['branch_slots']:,} branch slots, "
            f"{entry['samples']:,} samples, taken {entry['taken_rate']:.3f}, "
            f"{entry['distinct_pcs']} PCs, {elapsed:.0f}s, {entry['megabytes']} MB",
            flush=True,
        )

    if manifest:
        path = args.out / "manifest.json"
        existing = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        by_trace = {e["trace_id"]: e for e in existing}
        by_trace.update({e["trace_id"]: e for e in manifest})
        path.write_text(
            json.dumps(sorted(by_trace.values(), key=lambda e: e["trace_id"]), indent=2),
            encoding="utf-8",
        )
        print(f"Manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
