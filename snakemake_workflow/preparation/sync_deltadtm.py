"""
Download DeltaDTM v1.1 tiles from 4TU.ResearchData and sort them into the
DEM/mask tile directories the rest of the pipeline expects — the parent
directories of the `deltadtm` / `deltadtm_mask` sources in
data_catalog_gfm.yml (e.g. src/tile_chunking.py and extract_dem.py both
read individual tile files from those same directories, next to the
`deltadtm.vrt` / `deltadtm_mask.vrt` mosaics).

Tiles only - this script does NOT build either VRT mosaic itself (that used
to happen here; see build_deltadtm_vrt.py, a separate preparation step run
after this one - `python run_preparation.py sync_deltadtm build_deltadtm_vrt`,
or just run_preparation.py with no args, since both are enabled by default).
Splitting it out lets the VRT step be re-run on its own, cheaply (seconds,
no network) any time the mosaic needs rebuilding, without re-downloading
tiles - and it's what actually fixed a real bug: this script used to
download 4TU's own pre-built DEM VRT and patch its <SourceFilename> entries
to this machine's LOCAL ABSOLUTE tile paths, which only resolves on the
exact machine that patched it - it broke every DeltaDTM read on the Linux
HPC side (confirmed directly: 27 of 31 HPC preprocessing jobs failed on
2026-08-08 from exactly this). build_deltadtm_vrt.py builds both VRTs with
portable RELATIVE source paths instead, the same way this script's own
mask-VRT step always did (see that script's docstring for the full story).

HOW TO GET THE URLS:
Go to https://data.4tu.nl/datasets/1da2e70f-6c4d-4b03-86bd-b53e789cc629
For each file below, right-click its download button/link -> "Copy Link Address"
and paste it into the dict below. They look like:
https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/<file-uuid>

Only fill in the continents you actually need -- leave others as None
and they'll be skipped.

Not a standalone entry point - exposes `run(config)`, called from
run_preparation.py (`python run_preparation.py sync_deltadtm`).
"""

import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# 1) FILL IN YOUR COPIED LINKS HERE
# ---------------------------------------------------------------------------

DEM_ZIP_URLS = {
    "Africa": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/22ffa027-184b-4f67-9979-c182f3dfb1ab",
    "Antarctica": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/ca957a40-34fa-41eb-b101-e45d1ccbd890",
    "Asia": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/672eba4c-1334-44c6-8119-8879ded25912",
    "Europe": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/cb0b8ee3-b018-4828-a74e-2fb05020b1b6",
    "North_America": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/037664c6-1494-4889-9689-a56570728320",
    "Oceania": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/de972de1-26bd-4303-afdf-21a90a232cff",
    "South_America": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/db980f00-63cd-4a07-a4df-55ab06510594",
    "Seven_seas_(open_ocean)": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/fe986ba6-3db9-40e2-8a49-0fcdb341244a",
}

MASK_ZIP_URL = "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/bfe0fbc1-fdf3-40d0-a62c-adc58bfd9478"  # mask_tiles.zip

# ---------------------------------------------------------------------------
# 2) CORE FUNCTIONS -- shouldn't need to touch below this line
# ---------------------------------------------------------------------------


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream a large file to disk with progress reporting, resuming if partially present."""
    retry_transient_io(dest_path.parent.mkdir, parents=True, exist_ok=True)

    # Skip if already fully downloaded (best-effort check via Content-Length)
    head = requests.head(url, allow_redirects=True, timeout=30)
    expected_size = int(head.headers.get("Content-Length", 0))
    if dest_path.exists() and expected_size and dest_path.stat().st_size == expected_size:
        print(f"  Already downloaded: {dest_path.name} ({expected_size / 1e9:.2f} GB)")
        return

    print(f"  Downloading {dest_path.name} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)", end="")
        print()  # newline after progress


def extract_tifs(zip_path: Path, out_dir: Path) -> int:
    """Extract every .tif file inside a zip (ignoring subfolder structure) into out_dir."""
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".tif"):
                continue
            filename = Path(member).name  # flatten any subfolder structure
            target = out_dir / filename
            if target.exists():
                continue  # skip if already extracted (e.g. re-running the script)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def extraction_marker_path(out_dir: Path, name: str) -> Path:
    """Path to the "already extracted" marker for zip `name` in `out_dir`.

    Needed because `download_file`'s own already-downloaded check compares
    against the local .zip, which no longer exists once
    delete_zip_after_extract has removed it - without a separate marker,
    every re-run would re-download a continent's multi-GB zip purely to
    re-extract .tif files that are already all there (extract_tifs already
    skips per-file, but that doesn't avoid the download itself).
    """
    return out_dir / f"{name}.extracted"


def _try_delete(path: Path, attempts: int = 5, delay_s: float = 2.0) -> bool:
    """Best-effort delete, retrying briefly on PermissionError.

    Windows can transiently hold a lock on a just-closed file (e.g. Defender/
    endpoint AV real-time-scanning a freshly downloaded/extracted zip) for a
    second or two after the `zipfile.ZipFile` context manager has released
    it. Returns False (rather than raising) once attempts are exhausted, so
    a lock that outlasts the retries costs disk space, not the whole
    preparation run - extraction has already succeeded and the `.extracted`
    marker is already written by the time this is called.
    """
    for attempt in range(1, attempts + 1):
        try:
            path.unlink()
            return True
        except PermissionError:
            if attempt == attempts:
                return False
            time.sleep(delay_s)
    return False


def process_zip(
    name: str, url: str, zip_download_dir: Path, out_dir: Path, delete_zip_after_extract: bool,
) -> None:
    if not url:
        print(f"Skipping {name} (no URL provided)")
        return

    print(f"\n=== {name} ===")
    marker = extraction_marker_path(out_dir, name)
    if marker.exists():
        print(f"  Already extracted ({marker.name} found) - skipping download "
              f"(delete {marker} to re-download/re-extract)")
        return

    zip_path = zip_download_dir / f"{name}.zip"
    download_file(url, zip_path)

    print(f"  Extracting .tif files to {out_dir} ...")
    n = extract_tifs(zip_path, out_dir)
    print(f"  Extracted {n} file(s).")
    marker.write_text(f"{n} file(s) extracted to {out_dir}")

    if delete_zip_after_extract:
        if _try_delete(zip_path):
            print(f"  Deleted {zip_path.name} to save space.")
        else:
            print(f"  WARNING: {zip_path} is still locked by another process (e.g. antivirus) "
                  "- left on disk, safe to delete manually later.")


def run(config: dict) -> None:
    sync_cfg = config["sync_deltadtm"]

    # DEM/mask tiles land next to the `deltadtm`/`deltadtm_mask` VRT mosaics
    # in data_catalog_gfm.yml — the same directories src/tile_chunking.py
    # already reads individual tile files from, so nothing downstream needs
    # to know this script ran. The VRT mosaics themselves are built
    # separately, by build_deltadtm_vrt.py (run_preparation.py's next step)
    # - see this module's own docstring for why that's a separate step now.
    catalog = get_data_catalog(
        _REPO_ROOT / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"]
    )
    dem_out_dir = Path(catalog.get_source("deltadtm").path).parent
    mask_out_dir = Path(catalog.get_source("deltadtm_mask").path).parent

    zip_download_dir = Path(sync_cfg["zip_download_dir"])
    delete_zips = bool(sync_cfg.get("delete_zips_after_extract", False))

    retry_transient_io(dem_out_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io(mask_out_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io(zip_download_dir.mkdir, parents=True, exist_ok=True)

    print(f"DEM tiles will be extracted to:  {dem_out_dir}")
    print(f"Mask tiles will be extracted to: {mask_out_dir}")

    print("\nDownloading DEM tiles (per continent) ...")
    for continent, url in DEM_ZIP_URLS.items():
        process_zip(continent, url, zip_download_dir, dem_out_dir, delete_zips)

    print("\nDownloading mask tiles ...")
    process_zip("mask_tiles", MASK_ZIP_URL, zip_download_dir, mask_out_dir, delete_zips)

    print("\nDone.")
    print(f"DEM tiles:  {dem_out_dir.resolve()}")
    print(f"Mask tiles: {mask_out_dir.resolve()}")
    print("Run build_deltadtm_vrt (or run_preparation.py with no args) next to build the VRT mosaics.")
