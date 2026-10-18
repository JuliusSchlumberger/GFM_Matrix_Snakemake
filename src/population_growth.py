"""SSP population growth factors per (SSP, ISO-3, year) from IIASA-WiC projections.

Factors are cumulative relative to the 2020 baseline (2020 = 1.0; 1.15 = +15 %).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import warnings


# Countries absent from or named differently in Natural Earth lowres.
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
    "Curacao":                       "CUW",
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
    "Reunion":                       "REU",
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ne_map = {
        row["name"].strip(): row["iso_a3"].strip()
        for _, row in world.iterrows()
        if pd.notna(row["iso_a3"]) and str(row["iso_a3"]).strip() not in ("-99", "")
    }
    ne_map.update(_NAME_TO_ISO_OVERRIDES)
    return ne_map


def load_ssp_growth_factors(
    xlsx_path: str | Path,
    scenario_sheet: str = "SSP_scenarios",
    baseline_sheet: str = "historical_data_SSP",
    base_year: int = 2020,
) -> pd.DataFrame:
    """Load SSP growth factors as a DataFrame indexed (scenario, ISO), year columns.

    factor = population(SSP, country, year) / population(country, base_year), from
    the raw projections (2025-2100, million people) in `scenario_sheet` and the
    base-year population in `baseline_sheet`. The workbook's `growth_factors` sheet
    is not used: its lookup copied SSP1 into every SSP. Countries without an ISO-3
    match or base-year population are dropped with a warning.
    """
    xlsx_path = Path(xlsx_path)
    # Row 0 of both sheets is an Excel-table header ("Column1", ...); the real header is row 1.
    proj = pd.read_excel(xlsx_path, sheet_name=scenario_sheet, header=1)
    hist = pd.read_excel(xlsx_path, sheet_name=baseline_sheet, header=1)

    year_cols = sorted(int(c) for c in proj.columns
                       if isinstance(c, (int, float)) and c > base_year and proj[c].notna().any())
    base_pop = hist.drop_duplicates("region").set_index(hist["region"].astype(str).str.strip())[base_year]

    name_to_iso = _build_name_to_iso()

    records = []
    unmatched, no_base = [], []
    for _, row in proj.iterrows():
        country_name = str(row["region"]).strip()
        iso = name_to_iso.get(country_name)
        if iso is None:
            unmatched.append(country_name)
            continue
        pop0 = base_pop.get(country_name, np.nan)
        if not np.isfinite(pop0) or pop0 <= 0:
            no_base.append(country_name)
            continue
        vals = {base_year: 1.0, **{yr: float(row[yr]) / pop0 for yr in year_cols}}
        vals["scenario"] = str(row["scenario"]).strip()
        vals["ISO"] = iso
        records.append(vals)

    for names, what in ((unmatched, "could not be mapped to ISO-3"), (no_base, f"have no {base_year} population")):
        if names:
            unique = sorted(set(names))
            print(f"population_growth: {len(unique)} country name(s) {what} and were skipped:\n  "
                  + "\n  ".join(unique))

    df = pd.DataFrame(records)
    df = df.set_index(["scenario", "ISO"])
    df.columns = [int(c) for c in df.columns]
    df = df[sorted(df.columns)].sort_index()
    return df


def interpolate_growth_factor(
    growth_df: pd.DataFrame,
    ssp: str,
    iso: str,
    year: int,
    default: float = 1.0,
) -> float:
    """Linearly interpolate the growth factor for (SSP, ISO, year) on the 5-year grid.

    Returns `default` if the (SSP, ISO) pair is not in the table or the year is
    outside the available range.
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
