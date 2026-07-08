"""
sync_deltadtm.py — Sync DeltaDTM tiles from a local source and verify/download
any tiles listed in the manifest CSV that are missing or corrupted.

Two independent steps, both run by default:

  1. sync        — copy tiles from SOURCE to TARGET if not already present.
  2. verify      — read the manifest CSV in TARGET, check every listed tile
                   exists and is a readable GeoTIFF; download any that are
                   absent or corrupt.

Usage:
    python sync_deltadtm.py [--config PATH] [--dry-run] [--no-sync] [--no-verify]
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request

import yaml

_CHUNK = 1 << 20  # 1 MiB download chunk size
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "config.yml"


def _expand(s: str, root: str) -> str:
    """Substitute the `{root}` placeholder used throughout config.yml paths."""
    return str(s).replace("{root}", root)


# ── Step 1: local sync ────────────────────────────────────────────────────────


def sync(source: Path, target: Path, dry_run: bool = False) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    all_files = [p for p in source.rglob("*") if p.is_file()]
    missing = [p for p in all_files if not (target / p.relative_to(source)).exists()]
    already_ok = len(all_files) - len(missing)

    print(f"Source : {source}")
    print(f"Target : {target}")
    print(f"Total files in source : {len(all_files)}")
    print(f"Already in target     : {already_ok}")
    print(f"To copy               : {len(missing)}")
    if dry_run:
        print("(dry-run — nothing will be written)\n")

    for i, src_file in enumerate(missing, 1):
        dst_file = target / src_file.relative_to(source)
        rel = src_file.relative_to(source)
        print(f"[{i}/{len(missing)}] {rel}")
        if not dry_run:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

    if not dry_run:
        print(f"\nDone. {len(missing)} file(s) copied.")
    else:
        print(f"\nDry-run complete. {len(missing)} file(s) would be copied.")


# ── Step 2: manifest verify + download ───────────────────────────────────────


def _find_manifest(target: Path) -> Path:
    """Return the first CSV file found in *target*, or raise."""
    csvs = sorted(target.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No manifest CSV found in {target}")
    if len(csvs) > 1:
        print(f"  Warning: multiple CSVs found; using {csvs[0].name}")
    return csvs[0]


def _parse_manifest(csv_path: Path) -> list[tuple[str, str]]:
    """
    Read the manifest CSV and return [(url, filename), ...].
    The first column of every non-header row is treated as the URL; the
    tile filename is extracted as the last path component of the URL.
    Rows where the first column is empty or does not end with '.tif' are skipped.
    """
    entries: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            url = row[0].strip()
            if not url or not url.lower().endswith(".tif"):
                continue
            fname = Path(urlparse(url).path).name
            if fname:
                entries.append((url, fname))
    return entries


def _is_valid_tif(path: Path) -> bool:
    """Return True if *path* is a readable GeoTIFF (header + dimensions OK)."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            _ = src.width, src.height, src.crs
        return True
    except Exception:
        return False


def _download(url: str, dst: Path) -> None:
    """
    Stream-download *url* to *dst*, writing via a sibling temp file so the
    destination is never left in a partial state.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    try:
        req = Request(url, headers={"User-Agent": "sync_deltadtm/1.0"})
        with urlopen(req) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := resp.read(_CHUNK):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"\r    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)",
                        end="",
                        flush=True,
                    )
        print()  # newline after progress
        tmp.replace(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def verify_and_download(target: Path, dry_run: bool = False) -> None:
    """
    Check every tile listed in the manifest CSV against the target directory.
    Tiles that are absent or fail the GeoTIFF validity check are downloaded
    (unless dry_run is True).
    """
    manifest = _find_manifest(target)
    print(f"\nManifest : {manifest.name}")
    entries = _parse_manifest(manifest)
    print(f"Tiles in manifest : {len(entries)}")

    missing: list[tuple[str, str]] = []
    corrupted: list[tuple[str, str]] = []

    for url, fname in entries:
        dst = target / fname
        if not dst.exists():
            missing.append((url, fname))
        elif not _is_valid_tif(dst):
            corrupted.append((url, fname))

    present_ok = len(entries) - len(missing) - len(corrupted)
    print(f"  Present and valid : {present_ok}")
    print(f"  Missing           : {len(missing)}")
    print(f"  Corrupted         : {len(corrupted)}")

    to_fetch = missing + corrupted
    if not to_fetch:
        print("All manifest tiles are present and valid.")
        return

    if dry_run:
        print(f"\n(dry-run) Would download {len(to_fetch)} tile(s):")
        for _, fname in to_fetch:
            tag = "MISSING" if any(f == fname for _, f in missing) else "CORRUPT"
            print(f"  [{tag}] {fname}")
        return

    print(f"\nDownloading {len(to_fetch)} tile(s)…")
    failed: list[tuple[str, Exception]] = []
    for i, (url, fname) in enumerate(to_fetch, 1):
        dst = target / fname
        tag = "MISSING" if any(f == fname for _, f in missing) else "CORRUPT"
        print(f"[{i}/{len(to_fetch)}] [{tag}] {fname}")
        try:
            _download(url, dst)
            if not _is_valid_tif(dst):
                dst.unlink(missing_ok=True)
                raise ValueError("Downloaded file failed GeoTIFF validation")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append((fname, exc))

    n_ok = len(to_fetch) - len(failed)
    print(f"\nDownload complete: {n_ok} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed tiles:")
        for fname, exc in failed:
            print(f"  {fname}: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help=f"Path to config.yml (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing anything.",
    )
    parser.add_argument(
        "--no-sync", action="store_true", help="Skip the local SOURCE->TARGET copy step."
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the manifest verify + download step.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    root = config["paths"]["root"]
    sync_cfg = config["sync_deltadtm"]
    source = Path(_expand(sync_cfg["source"], root))
    target = Path(_expand(sync_cfg["target"], root))

    if not args.no_sync:
        print("═" * 60)
        print("Step 1 — local sync")
        print("═" * 60)
        sync(source, target, dry_run=args.dry_run)

    if not args.no_verify:
        print("\n" + "═" * 60)
        print("Step 2 — manifest verify + download")
        print("═" * 60)
        verify_and_download(target, dry_run=args.dry_run)
