"""Process pools: sized to the available cores, one BLAS thread per worker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from process_pool import available_cpus, process_pool  # noqa: E402


def _blas_threads(_: int) -> list[int]:
    import numpy  # noqa: F401
    from threadpoolctl import threadpool_info

    return [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]


def test_available_cpus_is_positive() -> None:
    assert available_cpus() >= 1


def test_available_cpus_is_the_slurm_allocation(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert available_cpus() == 3


def test_pool_workers_run_one_blas_thread() -> None:
    with process_pool(2) as pool:
        per_task = list(pool.map(_blas_threads, range(4)))
    assert all(threads and set(threads) == {1} for threads in per_task)
