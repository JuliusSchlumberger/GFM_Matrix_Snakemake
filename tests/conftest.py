"""pytest wiring for the standalone regression scripts under tests/.

Every script here still runs on its own (`python <script>.py`); this file only
tells pytest which ones need something the current environment may not have:

- `tests/sfincs_*` import sfincs_tiles/ modules that need hydromt_sfincs
  2.x, which lives in the separate hydromt-sfincs-dev env
  (environment_sfincs.yml). They are marked `needs_sfincs` and not collected
  at all without hydromt_sfincs 2.x, since importing them would fail at
  collection time.
- Tests that read the real P:\\ share are marked `needs_pdrive`. They already
  print SKIP and return when the data is absent, so the marker is for
  selection (`-m "not needs_pdrive"`), not for skipping.
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

_NEEDS_PDRIVE = {
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
    """Attach needs_sfincs / needs_pdrive markers by file location."""
    for item in items:
        path = Path(str(item.fspath))
        try:
            rel = _rel(path)
        except ValueError:
            continue
        if rel.startswith("sfincs_"):
            item.add_marker(pytest.mark.needs_sfincs)
        if rel in _NEEDS_PDRIVE:
            item.add_marker(pytest.mark.needs_pdrive)
