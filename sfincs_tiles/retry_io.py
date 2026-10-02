"""Retry wrapper for transient FileNotFoundError/OSError on the P:\\ SMB
share.

Duplicated from config_utils.retry_transient_io rather than imported:
config_utils's `from hydromt.log import setuplog` import fails under the
hydromt-sfincs-dev environment sfincs_tiles/ scripts run in.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

_T = TypeVar("_T")


def retry_transient_io(fn: Callable[..., _T], *args, retries: int = 4, delay_s: float = 5.0, **kwargs) -> _T:
    """Calls `fn(*args, **kwargs)`, retrying on FileNotFoundError/OSError up
    to `retries` times before raising."""
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
