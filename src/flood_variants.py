"""Named flood-model variants compared against SFINCS and the local hazard maps.

Each variant is an override of `simulation.flooding`; `variant_flooding` applies one with its
parameter and duration exponent. Stdlib only, so both conda envs import it.
"""

from __future__ import annotations

import copy

# model (flood_model.FLOOD_MODELS), friction source of the attenuation, correction B on/off.
VARIANTS = {
    "bathtub": {"model": "bathtub", "friction_source": None, "duration": False},
    "bathtub_connected": {"model": "bathtub_connected", "friction_source": None, "duration": False},
    "straight_line": {"model": "straight_line", "friction_source": "uniform", "duration": False},
    "eikonal_uniform": {"model": "eikonal", "friction_source": "uniform", "duration": False},
    "eikonal_landcover": {"model": "eikonal", "friction_source": "landcover", "duration": False},
    "eikonal_uniform_duration": {"model": "eikonal", "friction_source": "uniform", "duration": True},
    "eikonal_landcover_duration": {"model": "eikonal", "friction_source": "landcover", "duration": True},
}

# The config key a variant's calibrated parameter sets, by friction source.
PARAMETER_KEY = {"landcover": "friction_scale_factor", "uniform": "uniform_attenuation_m_per_km"}
PARAMETER_TAG = {"landcover": "fsf", "uniform": "rate"}


def parameter_key(name: str) -> str | None:
    """Return the `simulation.flooding` key of the variant's parameter (None: bathtubs)."""
    source = VARIANTS[name]["friction_source"]
    return None if source is None else PARAMETER_KEY[source]


def variant_flooding(flooding: dict, name: str, parameter: float | None = None,
                     alpha: float | None = None) -> dict:
    """Return a copy of `simulation.flooding` set to variant `name`.

    `parameter` sets friction_scale_factor (land cover, per metre) or
    uniform_attenuation_m_per_km (uniform); None keeps the config's. `alpha` sets the
    duration exponent of the duration variants; None keeps the config's.
    """
    spec = VARIANTS[name]
    out = copy.deepcopy(flooding)
    out["model"] = spec["model"]
    if spec["friction_source"] is not None:
        out["friction_source"] = spec["friction_source"]
        if parameter is not None:
            out[PARAMETER_KEY[spec["friction_source"]]] = float(parameter)
    out["duration_correction"]["enabled"] = spec["duration"]
    if spec["duration"] and alpha is not None:
        out["duration_correction"]["alpha"] = float(alpha)
    return out


def run_key(name: str, parameter: float | None = None, alpha: float | None = None) -> str:
    """Return the label of one variant run, e.g. "eikonal_landcover_duration_fsf0.25_a1"."""
    spec = VARIANTS[name]
    key = name
    if spec["friction_source"] is not None and parameter is not None:
        key += f"_{PARAMETER_TAG[spec['friction_source']]}{parameter:g}"
    if spec["duration"] and alpha is not None:
        key += f"_a{alpha:g}"
    return key
