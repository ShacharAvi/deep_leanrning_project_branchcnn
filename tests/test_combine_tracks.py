"""The generalist/specialist combination joins two pipelines per execution.

A misaligned join would attach a specialist's prediction to the wrong dynamic
branch and still produce a plausible MPKI, so it is pinned here.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.combine_tracks import blocks, conf_level, join  # noqa: E402


def test_join_maps_each_specialist_execution_to_its_own_sample():
    cnn = {"trace": np.array(["a"] * 5 + ["b"] * 4),
           "pos": np.array([3, 7, 9, 12, 20, 2, 5, 8, 11])}
    spec = {"trace": np.array(["b", "a", "a", "b", "c"]),
            "pos": np.array([8, 9, 13, 2, 4])}      # 13 and trace c are absent
    idx, keep = join(cnn, spec)
    assert keep.tolist() == [True, True, False, True, False]
    assert idx[keep].tolist() == [7, 2, 5]           # b@8, a@9, b@2
    assert np.array_equal(cnn["pos"][idx[keep]], spec["pos"][keep])
    assert np.array_equal(cnn["trace"][idx[keep]], spec["trace"][keep])


def test_join_handles_unsorted_sample_positions():
    cnn = {"trace": np.array(["a"] * 4), "pos": np.array([30, 10, 40, 20])}
    spec = {"trace": np.array(["a", "a"]), "pos": np.array([20, 40])}
    idx, keep = join(cnn, spec)
    assert keep.all() and cnn["pos"][idx].tolist() == [20, 40]


def test_blocks_are_contiguous_in_time_and_per_trace():
    trace = np.array(["a"] * 10 + ["b"] * 10)
    pos = np.concatenate([np.arange(10), np.arange(10)])
    g = blocks(trace, pos, n_blocks=5)
    assert len(set(g)) == 10                          # 5 blocks x 2 traces
    for t in ("a", "b"):
        seq = [x for x, tt in zip(g, trace) if tt == t]
        # contiguous: block ids never go back
        ids = [int(x.split("#")[1]) for x in seq]
        assert ids == sorted(ids)


def test_conf_level_from_tage_confidence_bits():
    conf = np.array([1, 2, 0, 5, 6])        # high | low<<1 | alt<<2
    assert conf_level(conf).tolist() == [0, 2, 1, 0, 2]   # HIGH=0, MEDIUM=1, LOW=2
