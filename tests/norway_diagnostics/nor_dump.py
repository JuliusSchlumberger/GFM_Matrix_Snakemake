"""Reader for the Norwegian StormfloHavniva PostGIS plain-SQL dump.

The dump is ``pg_dump`` plain-text output (PostgreSQL 15) where each table's
data sits in a ``COPY ... FROM stdin;`` block: tab-separated text with
PostgreSQL's COPY TEXT escaping, terminated by a ``\\.`` line.  The PostGIS
geometry column ``omrade`` is written as *hex EWKB* (e.g. ``0103000020E9640000``
= WKB Polygon, little-endian, with the SRID flag set and SRID 0x64E9 = 25833).

Nothing here needs a database: shapely can read the hex EWKB directly.

Requires ``dump_index.json`` from ``01_index_dump.py`` (byte offsets of each
COPY block) so we can seek instead of rescanning 28 GB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

CHUNK = 32 * 1024 * 1024

_UNESCAPE = {
    "\\": "\\",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "v": "\v",
}


def load_index(path: str | Path | None = None) -> dict:
    p = Path(path) if path else Path(__file__).with_name("dump_index.json")
    return json.loads(p.read_text())


def block_for(index: dict, table: str) -> dict:
    for b in index["blocks"]:
        if b["header"].split()[1].split(".")[-1] == table:
            return b
    raise KeyError(f"{table!r} not in dump index")


def columns(block: dict) -> list[str]:
    return [c.strip() for c in block["header"].split("(", 1)[1].rsplit(")", 1)[0].split(",")]


def unescape(field: str) -> str | None:
    """Apply PostgreSQL COPY TEXT unescaping. ``\\N`` is SQL NULL."""
    if field == "\\N":
        return None
    if "\\" not in field:
        return field
    out, i = [], 0
    while i < len(field):
        ch = field[i]
        if ch == "\\" and i + 1 < len(field):
            nxt = field[i + 1]
            if nxt in _UNESCAPE:
                out.append(_UNESCAPE[nxt])
                i += 2
                continue
            if nxt == "x":                       # \xHH
                out.append(chr(int(field[i + 2 : i + 4], 16)))
                i += 4
                continue
            if nxt.isdigit():                    # \OOO octal
                out.append(chr(int(field[i + 1 : i + 4], 8)))
                i += 4
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def iter_raw_lines(dump: Path, block: dict) -> Iterator[bytes]:
    """Yield the raw data lines of one COPY block (terminator excluded)."""
    remaining = block["end_offset"] - block["data_offset"]
    tail = b""
    with dump.open("rb") as fh:
        fh.seek(block["data_offset"])
        while remaining > 0:
            chunk = fh.read(min(CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            parts = (tail + chunk).split(b"\n")
            tail = parts.pop()
            yield from parts
    if tail:
        yield tail


def iter_rows(dump: Path, block: dict, limit: int | None = None) -> Iterator[dict]:
    """Yield each row of a COPY block as a dict of unescaped strings."""
    cols = columns(block)
    for i, line in enumerate(iter_raw_lines(dump, block)):
        if limit is not None and i >= limit:
            return
        fields = line.decode("utf-8").split("\t")
        yield dict(zip(cols, (unescape(f) for f in fields)))
