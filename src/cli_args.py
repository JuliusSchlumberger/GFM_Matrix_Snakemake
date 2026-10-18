"""argparse helper for scripts whose positional arguments became options.

Some calibration/preparation scripts used to take their output directory, figure
directory or tile ids positionally, while the rest of the repo uses options
(`--out-dir`, `--fig-dir`, `--tile-ids`, ...). Job files already written to P:\\
and the user's own command lines still use the positional form, so both must keep
working: `add_option_with_positional` registers the option and the old positional
on the same `dest`, and refuses a command line that gives both.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable


class _SetOnce(argparse.Action):
    # Shared by the option and the positional: the second one to fire is an error,
    # so `--out-dir A B` cannot silently pick one of the two. `convert` replaces
    # argparse's `type` on the positional, whose SUPPRESS default argparse would
    # otherwise run through `type` (int("==SUPPRESS==") fails).
    def __init__(self, *args, convert: Callable[[str], Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert = convert

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{self.dest}: give it either positionally or as an option, not both")
        if self._convert is not None:
            try:
                values = [self._convert(v) for v in values] if isinstance(values, list) else self._convert(values)
            except ValueError as e:
                parser.error(f"{self.dest}: {e}")
        setattr(namespace, self.dest, values)


def add_option_with_positional(
    parser: argparse.ArgumentParser,
    flag: str,
    dest: str,
    *,
    nargs: str | None = None,
    type: Callable[[str], Any] | None = None,  # noqa: A002 - mirrors add_argument
    help: str | None = None,  # noqa: A002 - mirrors add_argument
) -> None:
    """Add option `flag` and the legacy positional `dest`, both writing `args.<dest>`.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    flag : str
        The option string, e.g. ``"--out-dir"``.
    dest : str
        Attribute name, also the positional's name (the old CLI).
    nargs : str, optional
        ``None`` for one value, ``"+"`` for a list (the positional then takes ``"*"``).
    type : callable, optional
        Value conversion, as argparse's ``type``.
    help : str, optional
        Help text of the option.

    Notes
    -----
    Neither form is required by argparse; call `require` after ``parse_args`` when
    the value is mandatory.
    """
    action = lambda *a, **kw: _SetOnce(*a, convert=type, **kw)  # noqa: E731
    parser.add_argument(flag, dest=dest, nargs=nargs, default=None, action=action, help=help)
    # SUPPRESS: an absent positional must not overwrite the option's value.
    parser.add_argument(
        dest, nargs="*" if nargs == "+" else "?", default=argparse.SUPPRESS, action=action,
        help=f"legacy positional form of {flag}",
    )


def require(parser: argparse.ArgumentParser, args: argparse.Namespace, dest: str, flag: str) -> None:
    """Exit with a usage error if neither `flag` nor the positional `dest` was given."""
    if getattr(args, dest, None) in (None, []):
        parser.error(f"{flag} is required")
