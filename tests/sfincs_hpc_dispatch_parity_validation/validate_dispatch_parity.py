"""Validate that generate_sfincs_hpc_jobs.py (fixed N-way batch split) and
generate_sfincs_array_job.py (one SLURM job array) stay in sync - both
dispatch the SAME already-built sfincs.inp the SAME way, and this session
found REAL drift between them twice: the user had to explicitly correct
"that's my default pipeline too" after generate_sfincs_array_job.py alone
got the Docker Hub image fix and live `tee` output streaming, and
`--skip-tile-ids` was added to the array script first and only added to the
batch script afterward, on request.

Neither script exposes its own per-tile dispatch logic as an importable
function (both build a flat list of bash lines inline), so this test reads
each script's own SOURCE TEXT directly and checks for the same set of
critical substrings in both - this is deliberately a source-text check, not
a behavioural one, because the actual risk this guards against IS source-
text drift (one script getting a fix the other doesn't).

Usage:
    python validate_dispatch_parity.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SFINCS_TILES_DIR = REPO_ROOT / "sfincs_tiles"
sys.path.insert(0, str(SFINCS_TILES_DIR))

import generate_sfincs_array_job as array_job  # noqa: E402
import generate_sfincs_hpc_jobs as batch_job  # noqa: E402

# Constants that must be IDENTICAL between the two scripts - real drift
# risk: e.g. one script's own PARTITION_DEFAULT/CPUS_PER_TASK_DEFAULT gets
# tuned (real 2026-09 change: 1vcpu/1-core -> 4vcpu/4-core after live
# evidence of 34M-cell UTM grids) without the other following.
_SHARED_CONSTANTS = [
    "SFINCS_IMAGE_DEFAULT",
    "PARTITION_DEFAULT",
    "TIME_DEFAULT",
    "CPUS_PER_TASK_DEFAULT",
    "MEM_DEFAULT",
]

# Substrings that must appear in BOTH scripts' own source text - the actual
# apptainer invocation, live-output streaming, and copy-back behaviour that
# has drifted between the two scripts before.
_SHARED_SOURCE_SUBSTRINGS = [
    '/mnt/data "$SFINCS_IMAGE" sfincs',  # the apptainer exec target/bind-mount pattern
    "| tee",  # live stdout streaming into the SLURM %j/%A_%a.out log (was array-job-only once)
    "PIPESTATUS[0]",  # correct exit-code capture through the tee pipe (not $?, which would be tee's own)
    "sfincs_map.nc",  # the real output file copied back on success
    "sfincs_hpc_run.log",  # the per-tile run log copied back regardless of outcome
    "--skip-tile-ids",  # CLI flag for permanently-excluding known-failing tiles (was array-job-only once)
]


def test_shared_constants_match() -> None:
    print("=== generate_sfincs_hpc_jobs.py vs generate_sfincs_array_job.py: shared constants ===")
    mismatches = []
    for name in _SHARED_CONSTANTS:
        batch_val = getattr(batch_job, name)
        array_val = getattr(array_job, name)
        status = "OK" if batch_val == array_val else "MISMATCH"
        print(f"  [{status}] {name}: batch={batch_val!r}  array={array_val!r}")
        if batch_val != array_val:
            mismatches.append(name)
    assert not mismatches, f"constant(s) drifted between the two dispatch scripts: {mismatches}"
    print("PASS: all shared constants match")
    print()


def test_shared_source_substrings_present_in_both() -> None:
    print("=== generate_sfincs_hpc_jobs.py vs generate_sfincs_array_job.py: shared command patterns ===")
    batch_src = Path(batch_job.__file__).read_text(encoding="utf-8")
    array_src = Path(array_job.__file__).read_text(encoding="utf-8")

    missing_from_batch = [s for s in _SHARED_SOURCE_SUBSTRINGS if s not in batch_src]
    missing_from_array = [s for s in _SHARED_SOURCE_SUBSTRINGS if s not in array_src]

    for s in _SHARED_SOURCE_SUBSTRINGS:
        in_batch = s not in missing_from_batch
        in_array = s not in missing_from_array
        status = "OK" if (in_batch and in_array) else "MISSING"
        print(f"  [{status}] {s!r}: batch={in_batch}  array={in_array}")

    assert not missing_from_batch, f"generate_sfincs_hpc_jobs.py is missing: {missing_from_batch}"
    assert not missing_from_array, f"generate_sfincs_array_job.py is missing: {missing_from_array}"
    print("PASS: both scripts contain the same critical dispatch patterns")
    print()


def test_skip_tile_ids_filtering_logic_matches() -> None:
    """Both scripts' own `--skip-tile-ids` filtering (`[t for t in all if t
    not in skip]` + the "no tiles left" guard) should be textually
    identical - not just present in both (checked above), but doing the
    SAME thing."""
    print("=== generate_sfincs_hpc_jobs.py vs generate_sfincs_array_job.py: skip-tile-ids filter logic ===")
    batch_src = Path(batch_job.__file__).read_text(encoding="utf-8")
    array_src = Path(array_job.__file__).read_text(encoding="utf-8")

    filter_pattern = re.compile(r"tile_ids = \[t for t in all_tile_ids if t not in skip\]")
    guard_pattern = re.compile(r"no tile IDs left in .* after excluding")

    for name, src in [("batch", batch_src), ("array", array_src)]:
        assert filter_pattern.search(src), f"{name} script's skip-filter list-comprehension not found or changed"
        assert guard_pattern.search(src), f"{name} script's 'no tiles left after excluding' guard not found or changed"
    print("PASS: both scripts filter skip-tile-ids identically, with the same empty-result guard")
    print()


def main() -> None:
    test_shared_constants_match()
    test_shared_source_substrings_present_in_both()
    test_skip_tile_ids_filtering_logic_matches()
    print("All HPC dispatch-script parity checks passed.")


if __name__ == "__main__":
    main()
