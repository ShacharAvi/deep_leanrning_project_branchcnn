#!/usr/bin/env python3
"""Group CBP2025 traces by program: the basis of the cross-trace split.

    python tools/cbp2025_groups.py            # -> results/cbp2025/groups.json

BranchNet and Tarsa test on unseen INPUTS of the same binary. SBBT/CBP2025
traces do not say which binary they come from, so we infer it the way that
matters for per-branch models: two traces belong together if each one's hard
branches are hot in the other.

    link(a, b)  <=>  |H2P(a) & hot(b)| >= min_shared  and  |H2P(b) & hot(a)| >= min_shared

Groups are formed by COMPLETE linkage -- clusters merge only if every cross
pair is linked -- so a chain a~b~c cannot glue unrelated programs together
(single linkage chains unrelated traces together).

Grouping uses only program identity (which PCs exist and are hard), never a
test trace's accuracy for a selected branch, so it does not leak; branch
SELECTION happens later, on the training traces of each split only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TAGE = "tage_sc_l_64KB"


def trace_sets(path: Path, top_k: int, min_exec: int, min_miss: int,
               hot_exec: int) -> tuple[set[str], set[str]]:
    """(H2P PCs, hot PCs) of one trace from its whole-trace per-PC CSV."""
    hard, hot = [], set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ex = int(row["executions"])
            miss = int(row[f"{TAGE}_mispredictions"])
            if ex >= hot_exec:
                hot.add(row["pc"])
            if ex >= min_exec and miss >= min_miss:
                hard.append((miss, row["pc"]))
    hard.sort(reverse=True)
    return {pc for _, pc in hard[:top_k]}, hot


def complete_linkage(names: list[str], linked) -> list[list[str]]:
    clusters = [[n] for n in names]
    merged = True
    while merged:
        merged = False
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if all(linked(a, b) for a in clusters[i] for b in clusters[j]):
                    size = len(clusters[i]) + len(clusters[j])
                    if best is None or size > best[0]:
                        best = (size, i, j)
        if best is not None:
            _, i, j = best
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]
            merged = True
    return sorted(clusters, key=len, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+",
                    default=["media", "fp", "infra", "web"])
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--min-exec", type=int, default=3000)
    ap.add_argument("--min-miss", type=int, default=100)
    ap.add_argument("--hot-exec", type=int, default=1000)
    ap.add_argument("--min-shared", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=REPO / "results" / "cbp2025" / "groups.json")
    args = ap.parse_args()

    out = {"rule": vars(args) | {"out": str(args.out)}, "groups": []}
    for cat in args.categories:
        work = REPO / "data" / "cbp2025" / cat
        sets = {}
        for f in sorted(work.glob("*.full_per_pc.csv")):
            sets[f.name.split(".")[0]] = trace_sets(
                f, args.top_k, args.min_exec, args.min_miss, args.hot_exec)
        names = sorted(t for t in sets if sets[t][0])

        def linked(a, b):
            return (len(sets[a][0] & sets[b][1]) >= args.min_shared and
                    len(sets[b][0] & sets[a][1]) >= args.min_shared)

        clusters = complete_linkage(names, linked)
        for k, members in enumerate(clusters):
            shared = set.intersection(*(sets[m][1] for m in members))
            out["groups"].append({
                "name": f"{cat}_g{k}", "category": cat,
                "members": sorted(members, key=lambda t: (len(t), t)),
                "size": len(members),
                "hard_pcs_hot_in_all": len(
                    set.union(*(sets[m][0] for m in members)) & shared),
            })
        sizes = [len(c) for c in clusters]
        print(f"{cat}: {len(names)} traces -> groups {sizes[:10]}"
              f"{' ...' if len(sizes) > 10 else ''} ({sizes.count(1)} singletons)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    big = [g for g in out["groups"] if g["size"] >= 5]
    print(f"\n{len(big)} groups with >=5 traces:")
    for g in big:
        print(f"  {g['name']:12s} {g['size']:3d} traces, "
              f"{g['hard_pcs_hot_in_all']} hard PCs hot in every member")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
