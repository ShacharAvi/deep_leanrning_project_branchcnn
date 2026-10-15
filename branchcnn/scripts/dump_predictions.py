#!/usr/bin/env python3
"""Save a trained BranchCNN's per-execution predictions for validation and test.

    python scripts/dump_predictions.py --run results/<run> --features data/web/features \
        --tage-dir data/web/tage --split "W3,W5,W8,W27/W31/W32,W33"

Writes <run>/predictions_<role>.npz with, per scored sample in trace order:
  trace  trace id               pos    position among ALL branches (== our branch_index)
  logit  BranchCNN logit        label  actual direction
  tage_pred / tage_hit / tage_conf     CBP2016 TAGE-SC-L's prediction and state
and <run>/predictions_meta.json with the instructions per role (MPKI denominator).
Used to combine BranchCNN with the per-branch BranchNet specialists exactly,
joined on (trace, pos). No training.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import BranchDataset, DatasetConfig, SplitSpec, check_split  # noqa: E402
from src.model import BranchCNN, ModelConfig  # noqa: E402
from src.train import _forward, _to_torch  # noqa: E402


def parse_split(text):
    roles = [tuple(f.strip().upper() for f in r.split(",") if f.strip())
             for r in text.split("/")]
    spec = SplitSpec(*roles)
    check_split(spec)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--tage-dir", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    ck = torch.load(a.run / "model.pt", map_location="cpu")
    cfg = ModelConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                         for k, v in ck["model_config"].items()})
    dc = DatasetConfig(history_length=cfg.history_length, pc_bits=14,
                       use_tage=True, tage_dir=a.tage_dir)
    ds = BranchDataset(a.features, dc, parse_split(a.split))
    ds.load()
    model = BranchCNN(cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(a.device).eval()

    meta = {}
    for role in ("validation", "test"):
        traces = ds.splits[role].traces
        trace = np.concatenate([np.full(t.rows.size, t.features.trace_id.replace("_trace", ""))
                                for t in traces])
        pos = np.concatenate([t.features.s_pos[t.rows] for t in traces])
        label = np.concatenate([t.features.s_out[t.rows] for t in traces]).astype(np.int8)
        tage_pred = np.concatenate([t.tage.pred[t.rows] for t in traces]).astype(np.int8)
        tage_hit = np.concatenate([t.tage.hit[t.rows] for t in traces]).astype(np.int16)
        tage_conf = np.concatenate([t.tage.conf[t.rows] for t in traces]).astype(np.int8)
        logits = []
        with torch.no_grad():
            for batch in ds.iter_batches(role, batch_size=2048, shuffle=False):
                logit, _ = _forward(model, _to_torch(batch, a.device))
                logits.append(logit.float().cpu().numpy())
        logit = np.concatenate(logits)
        assert logit.shape == label.shape, (logit.shape, label.shape)
        np.savez_compressed(a.run / f"predictions_{role}.npz", trace=trace, pos=pos,
                            logit=logit, label=label, tage_pred=tage_pred,
                            tage_hit=tage_hit, tage_conf=tage_conf)
        instr = int(sum(t.features.instructions_seen for t in traces))
        m_wrong = int(((logit > 0).astype(np.int8) != label).sum())
        t_wrong = int((tage_pred != label).sum())
        meta[role] = {"samples": int(label.size), "instructions": instr,
                      "branchcnn_misses": m_wrong, "tage_misses": t_wrong,
                      "branchcnn_mpki": 1000 * m_wrong / instr,
                      "tage_mpki": 1000 * t_wrong / instr}
        print(role, meta[role], flush=True)
    (a.run / "predictions_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
