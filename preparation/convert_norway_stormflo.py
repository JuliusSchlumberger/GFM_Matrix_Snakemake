"""Convert a Norwegian StormfloHavniva hazard table from the PostGIS SQL dump to a GeoPackage.

Norway's coastal flood-hazard benchmark (Kartverket/DSB
"Samfunnssikkerhet - Stormflo og havniva") ships as a 26.1 GiB plain-text
PostgreSQL ``pg_dump``, not as a shapefile like Spain's and France's benchmarks.
No PostgreSQL/PostGIS install is needed to read it: every table's data sits in a
``COPY <table> (...) FROM stdin;`` block of tab-separated text using
PostgreSQL's COPY TEXT escaping, and the PostGIS geometry column ``omrade`` is
written as *hex EWKB*, which shapely reads directly
(``shapely.from_wkb(bytes.fromhex(...))``).

This script streams one such COPY block straight into a GeoPackage in batches,
so peak memory stays flat regardless of table size (the target table,
``stormflo200ar_klimaarna``, is a 2.4 GiB block of 115,440 polygons). The result
is a plain OGR vector file, i.e. an ordinary ``data_type: GeoDataFrame`` /
``driver: vector`` hydromt catalog entry, exactly like Spain's and France's
benchmark shapefiles - nothing downstream in the validation pipeline needs to
know the data originally came out of a database dump.

Like analysis/extract_country_population.py, this is NOT part of the regular
Snakemake DAG - it is a one-time/on-demand ingest step, run manually when the
source dump is (re)delivered.

Seeking instead of rescanning
-----------------------------
Locating a table by scanning 26 GiB of text takes minutes, so the byte offsets
of every COPY block are cached in a small JSON index next to the output. The
index is built automatically on first run (one full sequential pass) and reused
by every later run and every later table, keyed on the dump's byte size so a
re-delivered dump invalidates it.

Raw polygons include the sea - and that is fine
-----------------------------------------------
Each polygon in these tables is "everything below water level X", which covers
BOTH the flooded land AND the permanently-wet sea/fjord it is contiguous with:
at five AOIs spanning the coast, 97.9-99.8% of raw polygon area is sea and only
0.2-2.1% is genuine land inundation.

No vector-level sea subtraction is done here, on purpose.
validation/validate_country.py already excludes permanent water from the
evaluation domain for EVERY country, via validation.read_permanent_water_mask()
/ permanent_water_mask() (validation.permanent_water_source/permanent_water_codes
in config.yml - DeltaDTM's own land/ocean/lake/river mask, codes {1,2,3}, since
2026-09; was Copernicus Global Land Cover, codes {80,200} - see
docs/flood_extent_validation_caveats.md §1.3), applied identically to the
benchmark-wet area, the model-wet area and the domain. tests/norway_diagnostics/
09_landuse_mask_check.py measured the (then-default, Copernicus) mask against the
real data on the model's own ~30 m grid: it removes 98.1% of the sea inside these
polygons (92.7-99.1% per AOI, incl. the narrow Naeroyfjord and Lofoten's island
maze), so a national dissolve+difference against ``middelhoyvann_klimaarna`` is
redundant work regardless of which permanent-water source is configured.

Caveat that same check turned up, which the vector step would NOT have fixed:
the 100 m land-cover product also swallows ~51% of the genuine land-inundation
strip (29-53% per AOI), because a strip tens of metres wide above mean high
water is finer than one Copernicus pixel. This shrinks Norway's evaluation
domain rather than biasing it - the mask hits model and benchmark alike - and it
is pre-existing pipeline behaviour, not Norway-specific (the same mask removes
17% of Spain's accepted benchmark area, and 98% of Spain's most coastal
feature). Subtracting the sea in vector space would not recover any of it: the
mask is applied to the rasterised grid regardless of the input geometry.

Usage note: convert to EPSG:4326 (``--to-4326``) only if a consumer needs it;
the validation pipeline reprojects benchmark vectors itself.

Output schema (for ``stormflo200ar_klimaarna``)
-----------------------------------------------
``objid`` (int, per-feature ID), ``lokalid`` (str, national stable ID),
``oppdateringsdato``/``datauttaksdato`` (str dates), ``sikkerhetsklasseflom``
(str, TEK17 safety class F1/F2/F3), ``vannstandovernn2000`` (int, water level in
cm above the NN2000 vertical datum), ``geometry`` (Polygon, EPSG:25833).
Constant-valued bookkeeping columns (``objtype``, ``navnerom``, ``malemetode``,
``opphav``) are dropped; pass ``--keep-all-columns`` to retain them.

Usage:
    python snakemake_workflow/preparation/convert_norway_stormflo.py \\
        [--table stormflo200ar_klimaarna] \\
        [--dump P:/.../Samfunnssikkerhet_0000_Norge_25833_StormfloHavniva_PostGIS.sql] \\
        [--out  P:/.../NOR/stormflo200ar_klimaarna.gpkg] \\
        [--limit N] [--to-4326] [--make-valid] [--keep-all-columns] [--list-tables]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import pandas as pd
import shapely

# ── Defaults (production data root) ─────────────────────────────────────────

DEFAULT_DUMP = Path(
    r"P:\11212688-004-global-floodmaps\modelling\inputs\validation\NOR"
    r"\Samfunnssikkerhet_0000_Norge_25833_StormfloHavniva_PostGIS.sql"
)
DEFAULT_TABLE = "stormflo200ar_klimaarna"

# The dump declares SRID 25833 (ETRS89 / UTM zone 33N) on every geometry; the
# hex EWKB carries it too (0x64E9 = 25833), but shapely.from_wkb() discards the
# SRID flag's payload, so it is re-attached explicitly on the GeoDataFrame.
SOURCE_CRS = "EPSG:25833"

# Columns that are constant for the whole table and only bloat the output.
# objid/lokalid are kept as identifiers.
DROP_COLUMNS = ("navnerom", "malemetode", "opphav", "objtype")

# Columns worth storing as integers rather than text.
NUMERIC_COLUMNS = ("objid", "klimaar", "vannstandovernn2000")

GEOMETRY_COLUMN = "omrade"

READ_CHUNK = 32 * 1024 * 1024
BATCH_ROWS = 5000

# PostgreSQL COPY TEXT backslash escapes.
_UNESCAPE = {"\\": "\\", "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v"}


# ── Dump parsing ────────────────────────────────────────────────────────────
# Ported verbatim from the verified prototype in
# tests/norway_diagnostics/nor_dump.py + 01_index_dump.py - deliberately copied
# rather than rewritten, so the parsing that was proven against the real dump is
# the parsing that runs here.


def build_index(dump: Path) -> dict:
    """One sequential pass over the dump recording every COPY block's offsets."""
    size = dump.stat().st_size
    print(f"  Indexing {dump.name} ({size / 1024**3:.2f} GiB) - one-time, please wait...")

    blocks: list[dict] = []
    current: dict | None = None
    offset = 0
    tail = b""
    t0 = time.time()
    next_report = 1024**3

    with dump.open("rb") as fh:
        while True:
            chunk = fh.read(READ_CHUNK)
            if not chunk:
                break
            parts = (tail + chunk).split(b"\n")
            tail = parts.pop()                      # trailing partial line
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
            if fh.tell() >= next_report:
                el = time.time() - t0
                print(f"    {fh.tell() / 1024**3:6.2f} / {size / 1024**3:.2f} GiB "
                      f"({fh.tell() / 1024**2 / max(el, 1e-9):.0f} MB/s, "
                      f"{len(blocks)} blocks)", flush=True)
                next_report += 4 * 1024**3

    if current is not None:                         # unterminated (shouldn't happen)
        current["end_offset"] = None
        blocks.append(current)

    return {"dump": str(dump), "size": size, "blocks": blocks}


def load_or_build_index(dump: Path, index_path: Path) -> dict:
    """Reuse the cached index if it matches this dump's byte size, else rebuild."""
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            index = None
        if index and index.get("size") == dump.stat().st_size:
            print(f"  Reusing dump index: {index_path}")
            return index
        print(f"  Cached index does not match this dump's size - rebuilding: {index_path}")

    index = build_index(dump)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2))
    print(f"  Wrote dump index: {index_path} ({len(index['blocks'])} COPY blocks)")
    return index


def table_name(block: dict) -> str:
    """``COPY schema.table (cols) FROM stdin;`` -> ``table``."""
    return block["header"].split()[1].split(".")[-1]


def block_for(index: dict, table: str) -> dict:
    for block in index["blocks"]:
        if table_name(block) == table:
            return block
    known = ", ".join(sorted(table_name(b) for b in index["blocks"]))
    raise KeyError(f"table {table!r} not in dump index. Available: {known}")


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
            if nxt == "x":                          # \xHH
                out.append(chr(int(field[i + 2:i + 4], 16)))
                i += 4
                continue
            if nxt.isdigit():                       # \OOO octal
                out.append(chr(int(field[i + 1:i + 4], 8)))
                i += 4
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def iter_raw_lines(dump: Path, block: dict) -> Iterator[bytes]:
    """Yield the raw data lines of one COPY block (``\\.`` terminator excluded)."""
    remaining = block["end_offset"] - block["data_offset"]
    tail = b""
    with dump.open("rb") as fh:
        fh.seek(block["data_offset"])
        while remaining > 0:
            chunk = fh.read(min(READ_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            parts = (tail + chunk).split(b"\n")
            tail = parts.pop()
            yield from parts
    if tail:
        yield tail


# ── Conversion ──────────────────────────────────────────────────────────────

def convert(
    dump: Path, index: dict, table: str, out: Path,
    limit: int | None = None, to_4326: bool = False,
    make_valid: bool = False, keep_all_columns: bool = False,
    batch_rows: int = BATCH_ROWS,
) -> dict:
    """Stream `table` out of `dump` into `out` (GeoPackage), batch by batch."""
    block = block_for(index, table)
    cols = columns(block)
    if GEOMETRY_COLUMN not in cols:
        raise KeyError(f"{table!r} has no {GEOMETRY_COLUMN!r} column (columns: {cols})")
    geom_idx = cols.index(GEOMETRY_COLUMN)
    drop = () if keep_all_columns else DROP_COLUMNS
    keep = [(j, c) for j, c in enumerate(cols) if c != GEOMETRY_COLUMN and c not in drop]

    expected = block.get("rows")
    print(f"  Table   : {table}  ({expected:,} rows in index)" if expected
          else f"  Table   : {table}")
    print(f"  Columns : {', '.join(c for _, c in keep)} + geometry")
    print(f"  Output  : {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    stats = {"rows": 0, "invalid": 0, "repaired": 0, "empty": 0,
             "dropped_values": {c: set() for c in drop}}
    batch: list[list[str]] = []
    first = True
    t0 = time.time()

    def flush() -> None:
        nonlocal first
        if not batch:
            return
        geoms = shapely.from_wkb([bytes.fromhex(row[geom_idx]) for row in batch])
        invalid = ~shapely.is_valid(geoms)
        stats["invalid"] += int(invalid.sum())
        stats["empty"] += int(shapely.is_empty(geoms).sum())
        if make_valid and invalid.any():
            geoms[invalid] = shapely.make_valid(geoms[invalid])
            stats["repaired"] += int(invalid.sum())

        data = {c: [unescape(row[j]) for row in batch] for j, c in keep}
        gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry=geoms, crs=SOURCE_CRS)
        for c in NUMERIC_COLUMNS:
            if c in gdf:
                gdf[c] = pd.to_numeric(gdf[c], downcast="integer")
        if to_4326:
            gdf = gdf.to_crs(4326)

        # Record the distinct values of the columns we drop, so the claim that
        # they are constant is checked against the real data, not assumed.
        for c in drop:
            if c in cols:
                j = cols.index(c)
                seen = stats["dropped_values"][c]
                if len(seen) <= 5:
                    seen.update(unescape(row[j]) for row in batch)

        gdf.to_file(out, layer=table, driver="GPKG", mode="w" if first else "a")
        first = False
        stats["rows"] += len(batch)
        batch.clear()
        print(f"    {stats['rows']:>9,} rows  {time.time() - t0:6.0f}s", flush=True)

    for line in iter_raw_lines(dump, block):
        batch.append(line.decode("utf-8").split("\t"))
        if len(batch) >= batch_rows:
            flush()
            if limit is not None and stats["rows"] >= limit:
                break
    else:
        flush()

    stats["seconds"] = time.time() - t0
    stats["size_mb"] = out.stat().st_size / 1024**2 if out.exists() else 0.0
    return stats


def verify(out: Path, table: str, expected_rows: int | None) -> None:
    """Re-open the written file the way the pipeline will and report what is in it."""
    print("\n  Verifying written GeoPackage (geopandas.read_file)...")
    gdf = gpd.read_file(out, layer=table)
    print(f"    rows        : {len(gdf):,}" + (
        f"  (expected {expected_rows:,} - {'OK' if len(gdf) == expected_rows else 'MISMATCH'})"
        if expected_rows else ""))
    print(f"    crs         : {gdf.crs}")
    print(f"    columns     : {dict(gdf.dtypes.astype(str))}")
    print(f"    geom types  : {gdf.geom_type.value_counts().to_dict()}")
    print(f"    invalid     : {int((~gdf.is_valid).sum()):,}")
    print(f"    empty/null  : {int(gdf.is_empty.sum()):,} / {int(gdf.geometry.isna().sum()):,}")
    print(f"    bounds      : {[round(v, 1) for v in gdf.total_bounds]}")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        b = gdf.to_crs(4326).total_bounds
        print(f"    bounds 4326 : {[round(v, 4) for v in b]}")
    print(f"    total area  : {gdf.to_crs(25833).area.sum() / 1e6:,.1f} km2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP, help="plain-text pg_dump .sql file")
    ap.add_argument("--table", default=DEFAULT_TABLE, help=f"table to convert (default {DEFAULT_TABLE})")
    ap.add_argument("--out", type=Path, default=None, help="output .gpkg (default <dump dir>/<table>.gpkg)")
    ap.add_argument("--index", type=Path, default=None, help="dump index JSON (default <dump>.index.json)")
    ap.add_argument("--limit", type=int, default=None, help="stop after ~N rows (smoke test)")
    ap.add_argument("--to-4326", action="store_true", help="reproject to EPSG:4326 on write")
    ap.add_argument("--make-valid", action="store_true", help="repair invalid geometries")
    ap.add_argument("--keep-all-columns", action="store_true", help="keep the constant bookkeeping columns")
    ap.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    ap.add_argument("--list-tables", action="store_true", help="list the dump's tables and exit")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    dump: Path = args.dump
    if not dump.exists():
        raise SystemExit(f"dump not found: {dump}")

    index_path = args.index or dump.with_suffix(dump.suffix + ".index.json")
    index = load_or_build_index(dump, index_path)

    if args.list_tables:
        print(f"\n{'rows':>10}  table")
        for b in index["blocks"]:
            print(f"{b['rows']:>10,}  {table_name(b)}")
        return

    out: Path = args.out or dump.parent / f"{args.table}.gpkg"
    stats = convert(
        dump, index, args.table, out,
        limit=args.limit, to_4326=args.to_4326, make_valid=args.make_valid,
        keep_all_columns=args.keep_all_columns, batch_rows=args.batch_rows,
    )

    print(f"\n  {args.table}: {stats['rows']:,} rows -> {out} "
          f"({stats['size_mb']:,.0f} MB, {stats['seconds']:.0f}s)")
    print(f"  invalid geometries: {stats['invalid']:,}"
          + (f" (repaired {stats['repaired']:,})" if stats["repaired"] else "")
          + f"   empty: {stats['empty']:,}")
    for c, vals in stats["dropped_values"].items():
        shown = sorted(v for v in vals if v is not None)[:5]
        print(f"  dropped column {c!r}: {len(vals)} distinct value(s) seen, e.g. {shown}")

    if not args.no_verify:
        expected = block_for(index, args.table).get("rows") if args.limit is None else None
        verify(out, args.table, expected)


if __name__ == "__main__":
    main()
