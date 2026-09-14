"""Summarise the COPY block index: rows, byte size, offsets, column lists."""

import json
from pathlib import Path

d = json.loads(Path(__file__).with_name("dump_index.json").read_text())

print(f"{'table':<34}{'rows':>10}{'bytes':>16}{'GiB':>8}   data_offset")
print("-" * 90)
for b in d["blocks"]:
    t = b["header"].split()[1].split(".")[-1]
    n = b["end_offset"] - b["data_offset"]
    print(f"{t:<34}{b['rows']:>10,}{n:>16,}{n / 1024**3:>8.2f}   {b['data_offset']:,}")

print("\nColumn lists:")
for b in d["blocks"]:
    t = b["header"].split()[1].split(".")[-1]
    cols = b["header"].split("(", 1)[1].rsplit(")", 1)[0]
    print(f"  {t:<34} {cols}")
