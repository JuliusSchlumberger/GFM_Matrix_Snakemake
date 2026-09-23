"""Retry wrapper for a transient P:\\ SMB blip - a file that both exists and
is reachable moments later can still raise FileNotFoundError/OSError right
now (confirmed live 2026-09-23 on Hydrax, mid-batch, for a file that was
also confirmed present and readable the whole time from this Windows
machine's own view of the same share).

Duplicated from config_utils.retry_transient_io, not imported: every script
under sfincs_tiles/ that runs under hydromt-sfincs-dev needs this, and
config_utils's own `from hydromt.log import setuplog` import fails in that
env (newer hydromt - see build_elevation.py's own note on why mdt.py's
_load_mdt/_nearest_valid_grid are copied rather than cross-imported from
preparation/prepare_boundary_conditions.py for the same reason).
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

_T = TypeVar("_T")


def retry_transient_io(fn: Callable[..., _T], *args, retries: int = 4, delay_s: float = 5.0, **kwargs) -> _T:
    """Call `fn(*args, **kwargs)`, retrying on FileNotFoundError/OSError
    before giving up. A genuinely missing/corrupted file fails the same way
    on every attempt and still raises after retries are exhausted - this
    only adds a few seconds of delay in that case, in exchange for shrugging
    off the far more common transient blip."""
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
