"""Retry wrapper for transient FileNotFoundError/OSError on the shared data root (`paths.root`)."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

_T = TypeVar("_T")


# Twin of src/config_utils.py's retry_transient_io: hydromt-sfincs-dev can't import src/
# (src/config_utils needs hydromt 0.9). Pinned by tests/metric_parity_validation/validate_sfincs_src_parity.py.
def retry_transient_io(fn: Callable[..., _T], *args, retries: int = 4, delay_s: float = 5.0, **kwargs) -> _T:
    """Call `fn`, retrying on FileNotFoundError/OSError; `retries` is the total number of attempts.

    The last error is re-raised after `retries` failed attempts (so `retries - 1` retries).
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except (FileNotFoundError, OSError) as e:
            last_err = e
            if attempt < retries:
                print(
                    f"[retry {attempt}/{retries - 1}] {getattr(fn, '__name__', fn)} failed: {e} "
                    f"- retrying in {delay_s:.0f}s",
                    flush=True,
                )
                time.sleep(delay_s)
    raise last_err
