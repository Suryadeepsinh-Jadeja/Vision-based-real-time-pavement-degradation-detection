"""Regression tests for path resolution.

Guards the defect where the legacy prototype used CWD-relative paths. Launched
from the wrong directory it silently downgraded to the OpenCV detector *and*
loaded an empty defect ledger, reporting "PCI 100 / Good" for an unsurveyed road.

Verified against the old code: chdir to a temp dir and both
``models/pothole_yolov8.pt`` and ``data/sample_defects.json`` were unreachable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from roadscope import paths


@pytest.fixture(autouse=True)
def _clear_cache():
    paths.reset_cache()
    yield
    paths.reset_cache()


def test_project_root_contains_pyproject():
    root = paths.project_root()
    assert root.is_dir()
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()


def test_paths_independent_of_cwd(tmp_path):
    """The gate test: resolve everything from a foreign working directory."""
    original = Path.cwd()
    try:
        os.chdir(tmp_path)
        paths.reset_cache()

        assert paths.models_dir().is_absolute()
        assert paths.models_dir() == paths.project_root() / "models"
        assert paths.fixtures_dir() == paths.project_root() / "tests" / "fixtures"
        # CWD is not the project root, so a CWD-relative lookup must not be used.
        assert paths.models_dir() != tmp_path / "models"
    finally:
        os.chdir(original)
        paths.reset_cache()


def test_fixtures_exist():
    fixtures = paths.fixtures_dir()
    assert (fixtures / "sample_gps_track.csv").is_file()
    assert (fixtures / "sample_defects.json").is_file()


def test_root_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADSCOPE_ROOT", str(tmp_path))
    paths.reset_cache()
    assert paths.project_root() == tmp_path.resolve()


def test_root_override_rejects_missing_dir(tmp_path):
    bogus = tmp_path / "definitely-not-here"
    import os as _os

    prev = _os.environ.get("ROADSCOPE_ROOT")
    _os.environ["ROADSCOPE_ROOT"] = str(bogus)
    paths.reset_cache()
    try:
        with pytest.raises(paths.PathResolutionError):
            paths.project_root()
    finally:
        if prev is None:
            _os.environ.pop("ROADSCOPE_ROOT", None)
        else:
            _os.environ["ROADSCOPE_ROOT"] = prev
        paths.reset_cache()


def test_resolve_model_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADSCOPE_ROOT", str(tmp_path))
    paths.reset_cache()
    assert paths.resolve_model() is None
    assert paths.available_models() == []


def test_resolve_model_finds_first_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADSCOPE_ROOT", str(tmp_path))
    paths.reset_cache()
    models = paths.ensure_dir(tmp_path / "models")
    (models / "second.pt").write_bytes(b"x")
    (models / "first.pt").write_bytes(b"x")
    assert paths.resolve_model(("first.pt", "second.pt")).name == "first.pt"


def test_no_unresolvable_relative_paths_in_source():
    """No module may build a path from a bare relative string."""
    import roadscope

    package_root = Path(roadscope.__file__).parent
    offenders: list[str] = []
    for py in package_root.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # A relative literal used as a filesystem path.
            for literal in ('"models/', '"data/', "'models/", "'data/"):
                if literal in stripped and "resolve" not in stripped:
                    offenders.append(f"{py.name}:{lineno}: {stripped}")
    assert not offenders, "CWD-relative paths reintroduced:\n" + "\n".join(offenders)
