"""Validate that run_sfincs_tiles.py (the local build/run orchestrator) does
NOT pass any resolution/subgrid-overriding flags to build_sfincs_tile.py -
it must rely entirely on build_sfincs_tile.py's own CLI defaults
(MAIN_RES_M_DEFAULT/SUBGRID_NR_PIXELS_DEFAULT etc.), so a resolution change
made in ONE place (build_sfincs_tile.py) takes effect everywhere, rather
than silently being overridden by a stale flag baked into the orchestrator.

This is exactly the kind of drift that had to be manually grep-audited by
hand this session (2026-09) before trusting a full local rebuild of 255
production tiles - confirming ALL of run_sfincs_tiles.py,
run_sfincs_tiles.ps1, generate_sfincs_hpc_jobs.py, and
generate_sfincs_array_job.py were free of stale hardcoded resolution flags
after the 90m/15m -> 120m/30m default change. This test locks in the
Python orchestrator's own side of that (`run_sfincs_tiles.ps1` is bash/
PowerShell, not importable - see the note in the test body for how to keep
it honest too).

Usage:
    python validate_cli_defaults.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SFINCS_TILES_DIR = REPO_ROOT / "sfincs_tiles"
sys.path.insert(0, str(SFINCS_TILES_DIR))

import run_sfincs_tiles  # noqa: E402

# Any of these appearing in run_sfincs_tiles.py's own generated subprocess
# args would mean it's overriding build_sfincs_tile.py's own defaults -
# exactly the drift this test exists to catch.
_RESOLUTION_OVERRIDE_FLAGS = [
    "--resolution-m",
    "--subgrid-nr-pixels",
    "--subgrid-nr-levels",
    "--subgrid-nrmax",
]


def test_build_step_has_no_resolution_override_flags() -> None:
    print("=== run_sfincs_tiles.py: build_sfincs_tile.py step passes no resolution-override flags ===")
    steps = run_sfincs_tiles._steps(sfincs_exe="dummy.exe", timeout_s=1800.0, build_only=True)
    build_steps = [s for s in steps if s["script"] == "build_sfincs_tile.py"]
    assert len(build_steps) == 1, f"expected exactly one build_sfincs_tile.py step, found {len(build_steps)}: {steps}"
    extra_args = build_steps[0].get("extra_args", [])

    offending = [f for f in _RESOLUTION_OVERRIDE_FLAGS if f in extra_args]
    print(f"  build_sfincs_tile.py step's own extra_args: {extra_args}")
    assert not offending, (
        f"run_sfincs_tiles.py's build step passes {offending} - this OVERRIDES build_sfincs_tile.py's "
        f"own current defaults, so a future default change there would silently NOT apply when run "
        f"through this orchestrator. Remove the override (or update this test if it's deliberate)."
    )
    print("PASS: no resolution-override flags found - build step relies entirely on build_sfincs_tile.py's own defaults")
    print()


def test_all_steps_use_the_same_interpreter_launch_pattern() -> None:
    """Every step must be launched via sys.executable (not a hardcoded
    `python`/`python.exe`), so the whole pipeline - including
    build_boundary_forcing.py's own real geopandas union_all() call, which
    needs a geopandas version only present in hydromt-sfincs-dev - inherits
    whichever interpreter launched THIS script. A hardcoded 'python' in any
    step would silently break this guarantee for just that one step."""
    print("=== run_sfincs_tiles.py: every step launches via sys.executable, not a hardcoded interpreter ===")
    import inspect
    source = inspect.getsource(run_sfincs_tiles.run_tile)
    assert "sys.executable" in source, (
        "run_tile() no longer uses sys.executable to launch each step - a hardcoded interpreter "
        "would silently break under whichever conda env the outer script was launched from."
    )
    assert '"python"' not in source and "'python'" not in source, (
        "run_tile() appears to reference a hardcoded 'python' string, not just sys.executable."
    )
    print("PASS: run_tile() launches every step via sys.executable")
    print()


def main() -> None:
    test_build_step_has_no_resolution_override_flags()
    test_all_steps_use_the_same_interpreter_launch_pattern()
    print("All CLI-default drift checks passed.")


if __name__ == "__main__":
    main()
