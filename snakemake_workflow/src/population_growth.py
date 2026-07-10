"""Loading and processing of SSP population growth factors.

Reads the per-country, per-SSP, per-year growth factors from the project
Excel file (`inputs/SSPs/getting_SSP_population_growth_factors.xlsx`,
sheet `growth_factors` by default) and exposes them in a tidy DataFrame
indexed by (SSP, ISO-3, year).

Growth factors are cumulative and relative to the 2020 baseline
(2020 value = 1.0 everywhere).  A value of 1.15 means 15 % population
increase relative to 2020; 0.90 means 10 % decline.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import warnings


# ── Manual name → ISO-3 overrides ────────────────────────────────────────────
# Countries that are absent from or named differently in Natural Earth lowres.
_NAME_TO_ISO_OVERRIDES: dict[str, str] = {
    "Antigua and Barbuda":           "ATG",
    "Aruba":                         "ABW",
    "Bahrain":                       "BHR",
    "Barbados":                      "BRB",
    "Bosnia and Herzegovina":        "BIH",
    "Brunei Darussalam":             "BRN",
    "Cabo Verde":                    "CPV",
    "Central African Republic":      "CAF",
    "Comoros":                       "COM",
    "Curaçao":                       "CUW",
    "Curacao":                       "CUW",   # ASCII variant
    "Democratic Republic of the Congo": "COD",
    "Dominican Republic":            "DOM",
    "Equatorial Guinea":             "GNQ",
    "Eswatini":                      "SWZ",
    "French Guiana":                 "GUF",
    "French Polynesia":              "PYF",
    "Grenada":                       "GRD",
    "Guadeloupe":                    "GLP",
    "Guam":                          "GUM",
    "Hong Kong":                     "HKG",
    "Kiribati":                      "KIR",
    "Macao":                         "MAC",
    "Maldives":                      "MDV",
    "Malta":                         "MLT",
    "Martinique":                    "MTQ",
    "Mauritius":                     "MUS",
    "Mayotte":                       "MYT",
    "Micronesia":                    "FSM",
    "Russian Federation":            "RUS",
    "Réunion":                       "REU",
    "Reunion":                       "REU",   # ASCII variant
    "Saint Lucia":                   "LCA",
    "Saint Vincent and the Grenadines": "VCT",
    "Samoa":                         "WSM",
    "Sao Tome and Principe":         "STP",
    "Seychelles":                    "SYC",
    "Singapore":                     "SGP",
    "Solomon Islands":               "SLB",
    "South Sudan":                   "SSD",
    "Tonga":                         "TON",
    "United States":                 "USA",
    "United States Virgin Islands":  "VIR",
    "Viet Nam":                      "VNM",
    "Vietnam":                       "VNM",
    "Western Sahara":                "ESH",
}


def _build_name_to_iso() -> dict[str, str]:
    """Build a complete country-name → ISO-3 lookup using Natural Earth data.

    Natural Earth lowres covers ~170 countries; the manual overrides above
    fill the remainder of the ~201 countries in the SSP dataset.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ne_map = {
        row["name"].strip(): row["iso_a3"].strip()
        for _, row in world.iterrows()
        if pd.notna(row["iso_a3"]) and str(row["iso_a3"]).strip() not in ("-99", "")
    }
    # Overrides take precedence
    ne_map.update(_NAME_TO_ISO_OVERRIDES)
    return ne_map


def load_ssp_growth_factors(xlsx_path: str | Path, sheet_name: str = "growth_factors") -> pd.DataFrame:
    """Load SSP population growth factors and return a tidy per-(SSP, ISO, year) DataFrame.

    Args:
        xlsx_path: Path to the SSP growth-factors Excel file (catalog key
            ssp_population_growth_factors in data_catalog_gfm.yml).
        sheet_name: Sheet name within the Excel file - defaults to the
            catalog entry's own driver_kwargs.sheet_name value.

    Returns:
        DataFrame with MultiIndex (scenario, ISO) and columns = integer years.
        Values are growth factors relative to 2020 (2020 always = 1.0).
        Countries that could not be matched to an ISO-3 code are dropped with
        a warning.

    Example::

        df = load_ssp_growth_factors("inputs/SSPs/…/growth_factors.xlsx")
        factor_2050 = df.loc[("SSP2", "BGD"), 2050]   # Bangladesh under SSP2
    """
    xlsx_path = Path(xlsx_path)
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Year columns are integer (or may come in as int/float)
    year_cols = [c for c in raw.columns if isinstance(c, (int, float)) and c >= 2000]
    year_cols_int = [int(c) for c in year_cols]

    name_to_iso = _build_name_to_iso()

    records = []
    unmatched = []
    for _, row in raw.iterrows():
        ssp = str(row["scenario"]).strip()
        country_name = str(row["region"]).strip()
        iso = name_to_iso.get(country_name)
        if iso is None:
            unmatched.append(country_name)
            continue
        vals = {yr_int: float(row[yr]) for yr, yr_int in zip(year_cols, year_cols_int)}
        vals["scenario"] = ssp
        vals["ISO"] = iso
        records.append(vals)

    if unmatched:
        unique_unmatched = sorted(set(unmatched))
        print(
            f"population_growth: {len(unique_unmatched)} country name(s) could not be "
            f"mapped to ISO-3 and were skipped:\n  " + "\n  ".join(unique_unmatched)
        )

    df = pd.DataFrame(records)
    df = df.set_index(["scenario", "ISO"])
    df.columns = [int(c) for c in df.columns]  # ensure integer year columns
    df = df.sort_index()
    return df


def interpolate_growth_factor(
    growth_df: pd.DataFrame,
    ssp: str,
    iso: str,
    year: int,
    default: float = 1.0,
) -> float:
    """Return the interpolated growth factor for (SSP, ISO, year).

    Linear interpolation between the 5-year grid points provided by the
    SSP dataset.  Returns `default` if the (SSP, ISO) pair is not in the
    table or the year is outside the available range.
    """
    try:
        row = growth_df.loc[(ssp, iso)]
    except KeyError:
        return default

    years = np.array(sorted(row.index.astype(int)))
    values = row[years].to_numpy(dtype=float)
    if year <= years[0]:
        return float(values[0])
    if year >= years[-1]:
        return float(values[-1])
    return float(np.interp(year, years, values))


def get_geogunit_growth_series(
    growth_df: pd.DataFrame,
    iso_lookup: pd.Series,
    ssp: str,
    year: int,
    default: float = 1.0,
) -> pd.Series:
    """Return a per-geogunit growth factor Series for one (SSP, year).

    Args:
        growth_df:   Result of load_ssp_growth_factors.
        iso_lookup:  Series mapping geogunit_id → ISO-3 (e.g. df["ISO"]).
        ssp:         SSP label, e.g. "SSP2".
        year:        Target year (interpolated from 5-year grid).
        default:     Fallback for geogunits with no ISO or no SSP data.

    Returns:
        Series indexed like iso_lookup with float growth factors.
    """
    years = sorted(growth_df.columns.astype(int))
    y_lo, y_hi = years[0], years[-1]
    year_clamped = max(y_lo, min(y_hi, year))

    # Vectorised interpolation
    if year_clamped in growth_df.columns:
        ssp_vals = growth_df.xs(ssp, level="scenario")[year_clamped] if ssp in growth_df.index.get_level_values("scenario") else pd.Series(dtype=float)
    else:
        # Linear interpolation between bracketing years
        lo = max(y for y in years if y <= year_clamped)
        hi = min(y for y in years if y >= year_clamped)
        if lo == hi:
            ssp_vals = growth_df.xs(ssp, level="scenario")[lo] if ssp in growth_df.index.get_level_values("scenario") else pd.Series(dtype=float)
        else:
            try:
                grp = growth_df.xs(ssp, level="scenario")
                alpha = (year_clamped - lo) / (hi - lo)
                ssp_vals = grp[lo] + alpha * (grp[hi] - grp[lo])
            except KeyError:
                ssp_vals = pd.Series(dtype=float)

    # Map per-geogunit ISO → growth factor
    iso_to_factor = ssp_vals.to_dict()  # {ISO: factor}
    return iso_lookup.map(iso_to_factor).fillna(default)
