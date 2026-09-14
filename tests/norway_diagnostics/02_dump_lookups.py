"""Print the small lookup/codelist tables and the tail of the Norwegian dump.

Uses ``dump_index.json`` produced by ``01_index_dump.py`` to seek straight to
each small COPY block instead of rescanning the 28 GB file.
"""

from __future__ import annotations

import json
from pathlib import Path

IDX = json.loads(Path(__file__).with_name("dump_index.json").read_text())
DUMP = Path(IDX["dump"])
SMALL = {"dekningstatus", "malemetode", "sikkerhetsklasseflom"}


def read_block(fh, block) -> list[str]:
    fh.seek(block["data_offset"])
    raw = fh.read(block["end_offset"] - block["data_offset"])
    return raw.decode("utf-8").split("\n")


def main() -> None:
    with DUMP.open("rb") as fh:
        for b in IDX["blocks"]:
            table = b["header"].split()[1].split(".")[-1]
            if table not in SMALL:
                continue
            print("=" * 78)
            print(b["header"])
            print("=" * 78)
            for line in read_block(fh, b):
                if line:
                    print("   ", line.replace("\t", " | "))
            print()

        # tail of the file: constraints, indexes, comments
        size = IDX["size"]
        fh.seek(max(0, size - 12000))
        print("=" * 78)
        print("TAIL OF FILE (last 12 KB)")
        print("=" * 78)
        print(fh.read().decode("utf-8", "replace"))


if __name__ == "__main__":
    main()
