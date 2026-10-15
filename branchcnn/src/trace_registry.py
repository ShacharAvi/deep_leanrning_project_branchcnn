"""Discover CBP2025 traces on disk, beyond the two bundled sample traces.

Trace files are named ``<workload>/<trace_id>.gz`` inside the official
tarballs (e.g. ``media/media_0.gz``), so the workload comes from the parent
directory and the trace id from the filename stem.

Search roots, in order:

1. ``$CBP2025_TRACE_DIR`` if set.
2. ``C:\\cbp2025_traces`` on Windows / ``~/cbp2025_traces`` elsewhere.
3. ``data/raw`` inside the project.
4. ``external/cbp2025/sample_traces`` (the two bundled samples).

Large traces should live **outside** OneDrive. See ``scripts/ingest_traces.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRACE_DIR = PROJECT_ROOT / "external" / "cbp2025" / "sample_traces"
PROJECT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_EXTERNAL_DIR = (
    Path("C:/cbp2025_traces") if os.name == "nt" else Path.home() / "cbp2025_traces"
)


@dataclass(frozen=True)
class TraceRef:
    """A trace file plus the identifiers derived from its path."""

    path: Path
    workload_id: str
    trace_id: str

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def __str__(self) -> str:
        return f"{self.workload_id}/{self.trace_id}"


GZIP_MAGIC = b"\x1f\x8b"


def _trace_ref(path: Path) -> TraceRef:
    trace_id = path.name[: -len(".gz")] if path.name.endswith(".gz") else path.stem
    workload_id = path.parent.name or "unknown"
    # Bundled samples live in sample_traces/int/sample_int_trace.gz, so the
    # parent directory already carries the workload name.
    return TraceRef(path=path, workload_id=workload_id, trace_id=trace_id)


def is_gzip(path: Path) -> bool:
    """Identify traces by content, since member names may lack a .gz suffix."""
    try:
        with path.open("rb") as handle:
            return handle.read(2) == GZIP_MAGIC
    except OSError:
        return False


def search_roots(extra: Optional[Iterable[Path]] = None) -> list[Path]:
    roots: list[Path] = []
    if extra:
        roots.extend(Path(item) for item in extra)
    env_dir = os.environ.get("CBP2025_TRACE_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    roots.extend([DEFAULT_EXTERNAL_DIR, PROJECT_RAW_DIR, SAMPLE_TRACE_DIR])
    return [root for root in roots if root.is_dir()]


SKIP_SUFFIXES = {".parquet", ".csv", ".md", ".py", ".txt", ".xz", ".tar", ".pdf", ".json"}


def discover_traces(extra_roots: Optional[Iterable[Path]] = None) -> list[TraceRef]:
    """Find every gzip trace under the search roots, de-duplicated."""
    found: dict[tuple[str, str], TraceRef] = {}
    for root in search_roots(extra_roots):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if path.suffix != ".gz" and not is_gzip(path):
                continue
            ref = _trace_ref(path)
            found.setdefault((ref.workload_id, ref.trace_id), ref)
    return sorted(found.values(), key=lambda ref: (ref.workload_id, ref.trace_id))


def resolve_trace(selector: str, extra_roots: Optional[Iterable[Path]] = None) -> TraceRef:
    """Resolve a direct path, a ``workload/trace_id``, or a bare trace id."""
    direct = Path(selector)
    if direct.is_file():
        return _trace_ref(direct)

    candidates = discover_traces(extra_roots)
    wanted = selector.replace("\\", "/").strip("/")

    for ref in candidates:
        if wanted in (f"{ref.workload_id}/{ref.trace_id}", ref.trace_id):
            return ref

    available = ", ".join(str(ref) for ref in candidates[:20]) or "none"
    raise FileNotFoundError(
        f"Could not resolve trace {selector!r}.\n"
        f"Searched: {', '.join(str(r) for r in search_roots(extra_roots)) or 'no existing roots'}\n"
        f"Available: {available}"
    )
