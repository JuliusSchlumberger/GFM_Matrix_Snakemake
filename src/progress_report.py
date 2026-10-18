"""Text progress bar and ``--watch`` loop shared by the top-level check_*_progress.py monitors.

The monitors only count files on the shared output tree (read-only) and print one
report, or reprint it on an interval while a cluster batch runs.
"""

import time
from typing import Callable


def progress_bar(done: int, expected: int, bar_len: int = 30) -> str:
    """Return a ``#``/``-`` bar of ``bar_len`` characters for ``done`` out of ``expected``.

    Clamped at a full bar: an output directory can hold more files than the
    config expects (e.g. results of scenarios no longer configured).

    Parameters
    ----------
    done, expected : int
        Finished and expected item counts; ``expected == 0`` gives an empty bar.
    bar_len : int
        Bar width in characters.

    Returns
    -------
    str
        The bar, exactly ``bar_len`` characters long.
    """
    filled = int(bar_len * done / expected) if expected else 0
    filled = min(filled, bar_len)
    return "#" * filled + "-" * (bar_len - filled)


def run_report(report_fn: Callable[[], None], watch_s: float | None) -> None:
    """Call ``report_fn`` once, or every ``watch_s`` seconds until Ctrl+C.

    Parameters
    ----------
    report_fn : callable
        Counts and prints one report.
    watch_s : float or None
        Interval in seconds; None or 0 prints a single report. In watch mode each
        report is preceded by a timestamp header.
    """
    if watch_s:
        try:
            while True:
                print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                report_fn()
                time.sleep(watch_s)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        report_fn()
