"""Cross-validate the C++ driver against the Python reference predictors.

``tools/run_predictors.cpp`` is what produces every number in the project, so it
needs independent corroboration. These tests build a synthetic SBBT trace, run
it through both implementations, and require **exact** agreement on
misprediction counts -- not approximate.

That is a strong check: the Python ``Bimodal``/``GShare`` reproduce MBPlib's
indexing (including ``XorFold`` and the history alignment shift) and
``Perceptron`` reproduces the C++ ``ClassicPerceptron``, so any divergence in
hashing, counter width, update rule, or history policy shows up as a mismatch.

Skipped when ``build/run_predictors`` has not been built.
"""

import json
import subprocess

import numpy as np
import pytest

from src.predictors import (
    AlwaysNotTaken,
    AlwaysTaken,
    Bimodal,
    Branch,
    GShare,
    Perceptron,
    simulate,
)
from src.traces import OpCode, build_trace_bytes

from .conftest import REPO_ROOT

DRIVER = REPO_ROOT / "build" / "run_predictors"

pytestmark = pytest.mark.skipif(
    not DRIVER.exists(),
    reason="build/run_predictors not built (see README.md for the build command)",
)

# Must mirror tools/run_predictors.cpp BuildPredictors().
PYTHON_EQUIVALENTS = {
    "always_taken": lambda: AlwaysTaken(),
    "always_not_taken": lambda: AlwaysNotTaken(),
    "bimodal_64KB": lambda: Bimodal(log_table_size=18),
    "gshare_64KB": lambda: GShare(history_length=25, log_table_size=18),
    "perceptron_64KB": lambda: Perceptron(
        history_length=62, log_num_perceptrons=10, weight_bits=8
    ),
}


def make_trace(tmp_path, branches, name="xval.sbbt"):
    """Write a synthetic SBBT trace from a list of Branch objects."""
    raw = build_trace_bytes(
        ip=[b.pc for b in branches],
        target=[b.pc + 0x20 for b in branches],
        opcode=[
            (OpCode.JUMP | OpCode.CND) if b.conditional else OpCode.JUMP
            for b in branches
        ],
        taken=[int(b.taken) for b in branches],
        ninstr=[4] * len(branches),
    )
    path = tmp_path / name
    path.write_bytes(raw)
    return path


def run_driver(path, predictors=None):
    """Run the C++ driver, returning {predictor_name: mispredictions}."""
    cmd = [str(DRIVER), str(path)]
    if predictors:
        cmd += ["--predictors", ",".join(predictors)]
    out = subprocess.run(cmd, capture_output=True, check=True)
    report = json.loads(out.stdout)
    return {p["name"]: p["mispredictions"] for p in report["predictors"]}


def pseudorandom_branches(count, num_pcs=16, seed=0xC0FFEE):
    """A deterministic mixed workload: several PCs, some biased, some periodic.

    Deliberately not uniform noise. Real traces are overwhelmingly predictable
    (TAGE-SC-L mispredicts ~6% on our smoke trace), so a workload that is half
    coin-flips would exercise mostly the predictors' failure mode. The mix here
    is a quarter unpredictable, the rest learnable.
    """
    rng = np.random.default_rng(seed)
    pcs = [0x400000 + 0x40 * i for i in range(num_pcs)]
    bias = rng.uniform(0.05, 0.95, size=num_pcs)
    branches = []
    for i in range(count):
        slot = int(rng.integers(num_pcs))
        kind = slot % 4
        if kind == 0:
            taken = (i // 2) % 2 == 0                 # periodic
        elif kind == 1:
            taken = bool(rng.random() < bias[slot])   # biased
        elif kind == 2:
            taken = (i % 5) != 0                      # strongly biased, periodic
        else:
            taken = bool(rng.random() < 0.5)          # unpredictable
        conditional = slot % 8 != 7                   # some unconditional branches
        branches.append(
            Branch(pc=pcs[slot], taken=taken if conditional else True,
                   conditional=conditional)
        )
    return branches


# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(PYTHON_EQUIVALENTS))
def test_cpp_matches_python_on_mixed_workload(tmp_path, name):
    branches = pseudorandom_branches(20_000)
    path = make_trace(tmp_path, branches)

    cpp = run_driver(path, [name])[name]
    python = simulate(PYTHON_EQUIVALENTS[name](), branches).mispredictions

    assert cpp == python, f"{name}: C++ {cpp} vs Python {python}"


@pytest.mark.parametrize("pattern", ["1", "0", "10", "1100", "11010010"])
def test_cpp_matches_python_on_repeating_patterns(tmp_path, pattern):
    bits = [c == "1" for c in pattern] * 500
    branches = [Branch(pc=0x401000, taken=b) for b in bits]
    path = make_trace(tmp_path, branches)

    cpp = run_driver(path)
    for name, build in PYTHON_EQUIVALENTS.items():
        python = simulate(build(), branches).mispredictions
        assert cpp[name] == python, (
            f"{name} on {pattern!r}: C++ {cpp[name]} vs Python {python}"
        )


def test_cpp_matches_python_when_unconditional_branches_are_present(tmp_path):
    """Pins the history policy: unconditional branches are tracked, not predicted.

    This is the exact behaviour that differed between the canonical MBPlib loop
    and an earlier hand-rolled driver, so it gets its own test.
    """
    branches = []
    for i in range(4000):
        branches.append(Branch(pc=0x402000, taken=i % 3 == 0, conditional=True))
        branches.append(Branch(pc=0x402100, taken=True, conditional=False))
    path = make_trace(tmp_path, branches)

    cpp = run_driver(path, ["gshare_64KB"])["gshare_64KB"]
    python = simulate(GShare(25, 18), branches).mispredictions
    assert cpp == python


def test_driver_counts_branches_consistently(tmp_path):
    branches = pseudorandom_branches(5_000)
    path = make_trace(tmp_path, branches)
    out = subprocess.run([str(DRIVER), str(path)], capture_output=True, check=True)
    meta = json.loads(out.stdout)["metadata"]

    expected_conditional = sum(1 for b in branches if b.conditional)
    assert meta["branches_read"] == len(branches)
    assert meta["conditional_branches_counted"] == expected_conditional
    assert meta["distinct_conditional_pcs"] == len(
        {b.pc for b in branches if b.conditional}
    )


@pytest.mark.real_trace
def test_predictor_ordering_on_a_real_trace():
    """On a real workload the predictors must order by sophistication.

    This is a property of real traces, not of arbitrary data -- on pure noise
    no predictor beats a constant one. Values are from a verified run of
    SHORT_MOBILE-73; they pin the whole baseline stack against regressions.
    """
    from .conftest import SMOKE_TRACE

    cpp = run_driver(SMOKE_TRACE)

    assert cpp == {
        "always_taken": 3_396_315,
        "always_not_taken": 3_643_326,
        "bimodal_64KB": 886_239,
        "gshare_64KB": 562_651,
        "perceptron_64KB": 557_768,
        "hashed_perceptron_64KB": 461_385,
        "tage_sc_l_64KB": 411_926,
    }

    # Strictly improving, in the order the spec introduces them.
    ladder = [
        "always_not_taken",
        "always_taken",
        "bimodal_64KB",
        "gshare_64KB",
        "perceptron_64KB",
        "hashed_perceptron_64KB",
        "tage_sc_l_64KB",
    ]
    counts = [cpp[name] for name in ladder]
    assert counts == sorted(counts, reverse=True), dict(zip(ladder, counts))
