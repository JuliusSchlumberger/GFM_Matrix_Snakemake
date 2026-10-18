"""Mark and filter the standalone tests/ scripts for pytest.

`tests/sfincs_*` need hydromt_sfincs 2.x (hydromt-sfincs-dev env) and are not
collected without it. `needs_data` tests self-skip when the data root is absent; the
marker is only for selection (`-m "not needs_data"`).
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent


def _hydromt_sfincs_major() -> int:
    try:
        return int(importlib.metadata.version("hydromt_sfincs").split(".")[0])
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return 0


# sfincs_tiles/ uses the hydromt_sfincs 2.x API; some gfm installs carry an
# unrelated 1.x copy, which imports fine but lacks the modules these tests need.
_HAS_HYDROMT_SFINCS = _hydromt_sfincs_major() >= 2

_NEEDS_DATA = {
    "sfincs_antimeridian_validation/validate_antimeridian_handling.py",
    "sfincs_hydrograph_truncation_validation/validate_truncation_window.py",
    "sfincs_synthetic_end_to_end/test_sfincs_smoke.py",
}


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_TESTS_DIR).as_posix()


def _is_sfincs_test(path: Path) -> bool:
    return _rel(path).startswith("sfincs_")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip collecting sfincs_* tests when hydromt_sfincs is not importable."""
    try:
        if not _HAS_HYDROMT_SFINCS and _is_sfincs_test(collection_path):
            return True
    except ValueError:  # outside tests/
        pass
    return None


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Alias for `tmp_path`: scripts' own `main()` passes a temp dir positionally as `tmp_dir`."""
    return tmp_path


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Attach needs_sfincs / needs_data markers by file location."""
    for item in items:
        path = Path(str(item.fspath))
        try:
            rel = _rel(path)
        except ValueError:
            continue
        if rel.startswith("sfincs_"):
            item.add_marker(pytest.mark.needs_sfincs)
        if rel in _NEEDS_DATA:
            item.add_marker(pytest.mark.needs_data)
