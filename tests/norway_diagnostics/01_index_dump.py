"""Index the Norwegian StormfloHavniva PostGIS plain-SQL dump.

Streams the ~28 GB pg_dump text file once and records, for every
``COPY <table> (...) FROM stdin;`` block:

* byte offset of the COPY header line
* the full column list from the header
* byte offset of the first data row
* number of data rows (lines until the ``\\.`` terminator)
* byte offset of the terminator

Writes the index to ``dump_index.json`` next to this script so that later
scripts can seek straight to a table without rescanning 28 GB.

Usage:
    python 01_index_dump.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DUMP = Path(
    r"P:\11212688-004-global-floodmaps\modelling\inputs\validation\NOR"
    r"\Samfunnssikkerhet_0000_Norge_25833_StormfloHavniva_PostGIS.sql"
)
OUT = Path(__file__).with_name("dump_index.json")
CHUNK = 64 * 1024 * 1024


def main() -> None:
    size = DUMP.stat().st_size
    print(f"dump: {DUMP}")
    print(f"size: {size:,} bytes ({size / 1024**3:.2f} GiB)")

    blocks: list[dict] = []
    current: dict | None = None

    offset = 0          # byte offset of the start of `tail`
    tail = b""
    t0 = time.time()

    with DUMP.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            data = tail + chunk
            parts = data.split(b"\n")
            tail = parts.pop()          # trailing partial line
            pos = offset
            for line in parts:
                if current is None:
                    if line.startswith(b"COPY "):
                        current = {
                            "header": line.decode("utf-8", "replace"),
                            "header_offset": pos,
                            "data_offset": pos + len(line) + 1,
                            "rows": 0,
                        }
                elif line == b"\\.":
                    current["end_offset"] = pos
                    blocks.append(current)
                    current = None
                else:
                    current["rows"] += 1
                pos += len(line) + 1
            offset = pos

            done = fh.tell()
            el = time.time() - t0
            print(
                f"  {done / 1024**3:7.2f} / {size / 1024**3:.2f} GiB"
                f"  {done / 1024**2 / max(el, 1e-9):6.1f} MB/s"
                f"  blocks={len(blocks)}",
                flush=True,
            )

    if current is not None:                     # unterminated (shouldn't happen)
        current["end_offset"] = None
        blocks.append(current)

    OUT.write_text(json.dumps({"dump": str(DUMP), "size": size, "blocks": blocks}, indent=2))
    print(f"\nwrote {OUT}  ({len(blocks)} COPY blocks)\n")
    for b in blocks:
        table = b["header"].split()[1]
        print(f"{b['rows']:>10,} rows   {table}")


if __name__ == "__main__":
    main()
