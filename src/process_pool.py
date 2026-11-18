"""Process pools sized to the cores this process may use (on the HPC: the job's allocation)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

_blas_limits = None


def available_cpus() -> int:
    """Return the number of cores this process may run on (under SLURM: the job's cores)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # no affinity API outside Linux
        return os.cpu_count() or 1


def _single_blas_thread() -> None:
    # threadpoolctl (a scikit-learn dependency) resizes BLAS pools already started in a forked
    # worker, where OMP_NUM_THREADS no longer has an effect. numpy is imported first so a
    # spawned worker (Windows) has its BLAS loaded before the limit is set.
    import numpy  # noqa: F401
    from threadpoolctl import threadpool_limits

    global _blas_limits
    _blas_limits = threadpool_limits(1)


def process_pool(workers: int | None = None) -> ProcessPoolExecutor:
    """Return a pool of `workers` processes (default: `available_cpus()`), one BLAS thread each.

    One BLAS thread per worker keeps N workers on N cores rather than N workers times N threads.
    """
    return ProcessPoolExecutor(workers or available_cpus(), initializer=_single_blas_thread)
