"""Project path resolution.

Every filesystem path in RoadScope is derived from this module, never from the
current working directory.

Why this module exists
----------------------
The previous prototype opened ``models/pothole_yolov8.pt`` and
``data/sample_defects.json`` with CWD-relative paths. Launched from any directory
other than the repository root, the app *silently* fell back to the weaker
OpenCV detector **and** loaded an empty defect ledger -- which then produced a
confident "PCI 100 / Good" report for a road with no data at all.

Resolution order:

1. ``ROADSCOPE_ROOT`` environment variable, if set.
2. The nearest ancestor of this file containing a ``pyproject.toml`` (editable
   install / repo checkout).
3. The current working directory, as a last resort.

The result is cached. :func:`reset_cache` exists for tests.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent

#: Markers that identify the repository root.
_ROOT_MARKERS = ("pyproject.toml", ".git")

#: Candidate weight filenames, in search order.
MODEL_CANDIDATES: tuple[str, ...] = (
    "best.pt",
    "roadscope-yolov8m-seg.pt",
    "pothole_yolov8.pt",
    "yolov8m-seg.pt",
)


class PathResolutionError(RuntimeError):
    """Raised when the project root cannot be determined."""


def _looks_like_root(candidate: Path) -> bool:
    return any((candidate / marker).exists() for marker in _ROOT_MARKERS)


def _discover_root() -> Path:
    # 1. Explicit override.
    override = os.environ.get("ROADSCOPE_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise PathResolutionError(f"ROADSCOPE_ROOT is not a directory: {root}")
        return root

    # 2. Walk up from the installed package location.
    for candidate in PACKAGE_DIR.parents:
        if _looks_like_root(candidate):
            return candidate

    # 3. Fall back to CWD if it looks like the project.
    cwd = Path.cwd().resolve()
    if _looks_like_root(cwd):
        return cwd

    raise PathResolutionError(
        "Could not locate the RoadScope project root. Expected a pyproject.toml "
        f"or .git in an ancestor of {PACKAGE_DIR}, or ROADSCOPE_ROOT to be set. "
        "Refusing to guess, because guessing previously caused silent misconfiguration."
    )


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Absolute path to the repository root."""
    return _discover_root()


def reset_cache() -> None:
    """Clear the cached root. For tests that manipulate ``ROADSCOPE_ROOT``."""
    project_root.cache_clear()


def models_dir() -> Path:
    """Directory holding model weights."""
    return project_root() / "models"


def fixtures_dir() -> Path:
    """Directory holding test fixtures."""
    return project_root() / "tests" / "fixtures"


def data_dir() -> Path:
    """Runtime data directory (SQLite stores, thumbnails)."""
    return project_root() / "var"


def resolve_model(candidates: tuple[str, ...] = MODEL_CANDIDATES) -> Path | None:
    """Return the first weights file that exists under :func:`models_dir`.

    Returns ``None`` when nothing is found, which callers must treat as
    "detector unavailable" rather than silently degrading to a different engine.
    """
    models = models_dir()
    for name in candidates:
        path = models / name
        if path.is_file():
            return path
    return None


def available_models() -> list[Path]:
    """Every weights file present, for diagnostics."""
    models = models_dir()
    if not models.is_dir():
        return []
    return sorted(p for p in models.glob("*.pt") if p.is_file())


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
