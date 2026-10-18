"""Download DeltaDTM v1.1 tiles from 4TU.ResearchData into the catalog's tile dirs.

Entry point: run(config), via `python run_preparation.py sync_deltadtm`.
Tiles only; VRT mosaics are built by build_deltadtm_vrt.py.
"""

import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Copied from https://data.4tu.nl/datasets/1da2e70f-6c4d-4b03-86bd-b53e789cc629
# (right-click each download link -> Copy Link Address); None skips a continent.
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


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream a file to disk with progress; skip if already complete (Content-Length)."""
    retry_transient_io(dest_path.parent.mkdir, parents=True, exist_ok=True)

    # Best-effort completeness check via Content-Length.
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
        print()


def extract_tifs(zip_path: Path, out_dir: Path) -> int:
    """Extract new .tif files from a zip into out_dir, flattened; return the count."""
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".tif"):
                continue
            filename = Path(member).name
            target = out_dir / filename
            if target.exists():
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def extraction_marker_path(out_dir: Path, name: str) -> Path:
    """Return the "already extracted" marker path for zip name in out_dir.

    Needed because download_file's size check fails once the zip is deleted,
    which would re-download multi-GB zips on every run.
    """
    return out_dir / f"{name}.extracted"


def _try_delete(path: Path, attempts: int = 5, delay_s: float = 2.0) -> bool:
    """Delete path, retrying on PermissionError; return False instead of raising.

    Windows AV can briefly lock a just-closed zip; a lingering lock costs only disk.
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
    """Download one zip unless its extraction marker exists, then extract its .tifs."""
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
    """Download DeltaDTM DEM/mask tile zips and extract them into the catalog's tile dirs."""
    sync_cfg = config["sync_deltadtm"]

    # Tiles go next to the deltadtm/deltadtm_mask VRTs, where downstream reads them.
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
