"""Download and unzip hourly GTSM reanalysis water levels from the Copernicus CDS.

Raw hourly storm_surge_residual and tidal_elevation at GTSM stations -
not the COAST-RP/COAST-HG return-period products already under inputs/COAST-RP_dataset
and inputs/COAST_HG (see docs/data.md); this is the full time series those are
derived from. tidal_elevation is the tide-only GTSM run (no MSL change, relative to
1986-2005 MSL); total water level = tide + residual, so it is not downloaded.

tidal_elevation needs a DIFFERENT request than storm_surge_residual - confirmed
against the CDS API's own /constraints endpoint 2026-11-03, not guessed:
experiment="historical" (not "reanalysis" - the dataset's own description: tide
is "not based on any future climate scenario, since the tide-generating forces
... are not related to any climatological nor meteorological conditions"),
temporal_aggregation="10_min" (not "hourly" - no hourly aggregation exists for
it), version="v1" (not "v3"). Under experiment="historical" the CDS constraint
also caps tidal_elevation at year<=2014 - tide for 2015-2050 would need
experiment="future", which brings in a 5-GCM ensemble ("model") dimension; not
implemented here, since picking a GCM (or averaging several) isn't this
script's call to make silently.

Needs CDS API credentials (cdsapi package + a ~/.cdsapirc file - see
https://cds.climate.copernicus.eu/how-to-api); resumable: a year already
extracted (its .extracted marker exists) is skipped on a re-run. A year CDS
rejects (e.g. the most recent year, not yet published - a 400 from
client.retrieve) or a corrupt archive is logged and skipped, not fatal: every
other (variable, year) still gets attempted, and the run only raises at the
end, listing everything that needs a re-run.

Entry point: run(config), via `python run_preparation.py sync_gtsm` - opt-in only
(not part of a bare `run_preparation.py`'s default step set: a 75-year, two-variable,
credentialed CDS download is not something to kick off accidentally).
"""

import sys
import zipfile
from pathlib import Path

import cdsapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import retry_transient_io  # noqa: E402

DATASET = "sis-water-level-change-timeseries-cmip6"
MONTHS = [f"{m:02d}" for m in range(1, 13)]

# Per-variable request params + its own valid year range, confirmed against the
# CDS /constraints endpoint (not every variable shares one request shape - see
# module docstring for tidal_elevation's own reasoning).
VARIABLES: dict[str, dict] = {
    "storm_surge_residual": {
        "years": [str(y) for y in range(1950, 2025)],
        "request_extra": {"experiment": "reanalysis", "temporal_aggregation": ["hourly"], "version": ["v3"]},
    },
    # One nodal cycle (18.61 y) ending at the historical cap 2014: enough for MHWS / HAT. 10-min
    # files are ~6x the hourly size, so the full 1950-2014 span would add ~100 GB for nothing.
    # Not total_water_level under experiment="historical": that run is GCM-forced, not ERA5, so
    # its storms do not match the reanalysis surge residual.
    "tidal_elevation": {
        "years": [str(y) for y in range(1996, 2015)],
        "request_extra": {"experiment": "historical", "temporal_aggregation": ["10_min"], "version": ["v1"]},
    },
}

DELETE_ZIP_AFTER_EXTRACT = False


def extract(zip_path: Path, dest: Path) -> bool:
    """Extract zip_path into dest; return False if the archive is corrupt."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()  # CRC check on every member
            if bad is not None:
                print(f"  corrupt member {bad} in {zip_path.name}")
                return False
            zf.extractall(dest)
            print(f"  extracted {len(zf.namelist())} files -> {dest}")
        return True
    except zipfile.BadZipFile:
        print(f"  {zip_path.name} is not a valid zip")
        return False


def run(config: dict) -> None:
    """Download+extract GTSM hourly water levels into inputs/GTSM_storm_tide_hourly."""
    out_dir = Path(config["paths"]["root"]) / "inputs" / "GTSM_storm_tide_hourly"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    client = cdsapi.Client()
    failed: list[str] = []

    for var, spec in VARIABLES.items():
        extract_dir = out_dir / var
        retry_transient_io(extract_dir.mkdir, parents=True, exist_ok=True)

        for year in spec["years"]:
            target = out_dir / f"{var}_{year}.zip"
            done_marker = extract_dir / f".{var}_{year}.extracted"

            if done_marker.exists():
                print(f"skip {target.name} (already extracted)")
                continue

            # Temporary name: an interrupted transfer never looks like a finished zip.
            if not target.exists():
                tmp = target.with_suffix(".zip.part")
                request = {"variable": [var], "year": [year], "month": MONTHS, **spec["request_extra"]}
                print(f"requesting {target.name}")
                try:
                    client.retrieve(DATASET, request).download(str(tmp))
                except Exception as ex:
                    # CDS 400s a year it hasn't published yet (the most recent year
                    # is the usual case) - letting that abort the whole run would
                    # also skip every remaining year and the other variable entirely,
                    # even though they're independently requestable.
                    print(f"  {target.name}: request failed ({ex}) - skipping this year", file=sys.stderr)
                    tmp.unlink(missing_ok=True)
                    failed.append(target.name)
                    continue
                tmp.replace(target)

            print(f"extracting {target.name}")
            if not extract(target, extract_dir):
                # Removed so the next run fetches it again.
                target.unlink(missing_ok=True)
                failed.append(target.name)
                continue

            done_marker.touch()
            if DELETE_ZIP_AFTER_EXTRACT:
                target.unlink()

    if failed:
        raise RuntimeError(f"{len(failed)} archive(s) corrupt, re-run to fetch again: {', '.join(failed)}")
