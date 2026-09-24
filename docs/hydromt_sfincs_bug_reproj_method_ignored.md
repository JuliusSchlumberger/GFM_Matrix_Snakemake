## `merge_multi_dataarrays()` silently discards an explicitly-requested `reproj_method`, always falling back to `"bilinear"`

**Component:** `hydromt_sfincs.workflows.merge.merge_multi_dataarrays()`, used internally by `SfincsModel.subgrid.create(elevation_list=..., roughness_list=...)`.

**Version:** `hydromt_sfincs` 2.0.0-rc4dev, `hydromt` 1.4.1.

**Summary:** When a caller explicitly sets `"reproj_method"` on an entry passed to `elevation_list`/`roughness_list` (e.g. `{"elevation": my_source, "reproj_method": "nearest"}`), that choice is silently overwritten with `"bilinear"` before the actual reprojection happens. There is no warning or error - the function just does something different from what was asked.

**Where:** `hydromt_sfincs/workflows/merge.py`, inside `merge_multi_dataarrays()`.

First dataset in the list (~line 89-100):
```python
if method is None and da_like is not None:
    dx_like = ...
    if dx_1 >= dx_like:
        method = "bilinear"
    else:
        method = "average"
else:
    method = "bilinear"          # <-- also hit when method WAS explicitly given
```
`method` is read a few lines above via `method = da_list[0].get("reproj_method", None)`. The `else` branch is clearly meant to be the fallback for "no `da_like` grid available" - but it's also reached whenever `method is not None` (i.e. the caller *did* specify one), since the `if` requires both conditions. In that case the caller's explicit choice is discarded and replaced with `"bilinear"`.

Every subsequent dataset in the list (~line 148-163) has the identical pattern:
```python
reproj_method = da_list[i].get("reproj_method", None)
...
if reproj_method is None:
    ... auto-detect bilinear/average by resolution ...
else:
    reproj_method = "bilinear"   # <-- discards the caller's explicit value here too
logger.debug(f"Reprojection method of dataset {str(i)} is: {method}")  # also logs the wrong variable
```

**Expected:** an explicitly-supplied `reproj_method` should be used as-is; auto-detection should only apply when the caller left it unset (`None`).

**Actual:** any explicit `reproj_method` is unconditionally overwritten to `"bilinear"`.

**Impact on us:** we build SFINCS subgrid tables from a DEM whose real native resolution (~30 m) is coarser than the subgrid we wanted (15 m in an early version of our pipeline). We wanted nearest-neighbour resampling so every subgrid pixel stays a real source sample rather than an interpolated value - both to avoid fabricating sub-resolution detail that doesn't exist in the source, and to keep our SFINCS DEM values comparable pixel-for-pixel against a separate model we run on the same grid. Because of this bug, requesting `reproj_method="nearest"` had no effect; we observed small, physically implausible depressions (several metres of spurious "flooding" not present when compared against an independent, subgrid-free reference) that we traced to forced bilinear interpolation below the source data's real resolution.

**Current workaround on our side:** we pre-reproject our elevation/roughness rasters onto the exact destination grid ourselves (nearest-neighbour) before calling `subgrid.create()`, so whatever method the function picks internally becomes a no-op (source and destination grids are already identical). This works, but it shouldn't be necessary - the public `reproj_method` parameter should do what it says.

**Suggested fix:** let an explicit value pass straight through instead of being overwritten, e.g.:
```python
if method is None and da_like is not None:
    ... auto-detect ...
elif method is None:
    method = "bilinear"  # only when there's no da_like to auto-detect from
# else: leave the caller's own explicit method alone
```
with the equivalent fix in the `i >= 1` loop for `reproj_method`, and fixing the debug log there to print `reproj_method` instead of the unrelated `method` variable.
