"""Shared pytest fixtures and configuration."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

#: Small real trace used for end-to-end checks (0.12 GB decompressed,
#: 7.4 M branches, 244 distinct conditional PCs).
SMOKE_TRACE = DATA_DIR / "SHORT_MOBILE-73.sbbt.zst"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_trace: needs a trace under data/; skipped automatically if absent",
    )


def pytest_collection_modifyitems(config, items):
    """Skip real_trace tests when the data is not present."""
    if SMOKE_TRACE.exists():
        return
    skip = pytest.mark.skip(reason=f"{SMOKE_TRACE.name} not found under data/")
    for item in items:
        if "real_trace" in item.keywords:
            item.add_marker(skip)
