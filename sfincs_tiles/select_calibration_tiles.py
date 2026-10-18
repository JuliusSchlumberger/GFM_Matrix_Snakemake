"""Select the tiles for the final SFINCS calibration (run by generate_calibration_jobs.py).

  pool    hop_distance == 0 tiles minus antimeridian-crossing ones (no size/river/
          ocean/latitude filter, by design), split into --n-shards id files
  screen  production inputs present and >= 1 COAST-RP station
  select  screened + suitable boundary line + SFINCS mask pre-flight ok; stratified
          sample of --n-tiles over 15 deg lon/lat bins

Usage:
    python select_calibration_tiles.py pool   --config <cfg> --base-dir-name calibration_roughness_SFINCS --n-shards 10
    python select_calibration_tiles.py screen --config <cfg> --base-dir-name calibration_roughness_SFINCS \\
        --tile-ids-file candidates/shard_000.txt --out-ids candidates/screened_000.txt
    python select_calibration_tiles.py select --config <cfg> --base-dir-name calibration_roughness_SFINCS --n-tiles 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

TILE_GRID = "processed_inputs/mask/domain_tiles_global.gpkg"
OCEAN_CODE, RIVER_CODE = 1, 3  # config tile_generation.ocean_code / river_code
GRID_DEG = 15.0  # lon/lat bin size for the stratified sample
ANTIMERIDIAN_WIDTH_DEG = 60.0  # wider than any real tile's bbox
CATEGORY_STYLE = {  # funnel stage -> (colour, label)
    "selected": ("#2a78d6", "selected"),
    "eligible": ("#7fb3e8", "eligible, not sampled"),
    "preflight_failed": ("#e8a33d", "line ok, SFINCS mask pre-flight failed"),
    "line_unsuitable": ("#d6452a", "boundary line unsuitable"),
    "screened_out": ("#b3b3ad", "no production inputs / no COAST-RP station"),
    "not_screened": ("#dcdcd8", "not screened (stage-1 job missing?)"),
}


def stratified_sample_exact(df: pd.DataFrame, n_target: int, grid_deg: float, seed: int) -> pd.DataFrame:
    """Sample n_target tiles round-robin over shuffled lon/lat bins, so sparse bins are not starved.

    `n_target` must be <= len(df).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_bin"] = (np.floor(df["lon"] / grid_deg).astype(int).astype(str) + "_"
                  + np.floor(df["lat"] / grid_deg).astype(int).astype(str))
    pools = {b: list(g.index) for b, g in df.groupby("_bin")}
    for b in pools:
        rng.shuffle(pools[b])
    bins = list(pools)
    rng.shuffle(bins)
    picked = []
    while len(picked) < n_target and any(pools.values()):
        for b in bins:
            if len(picked) >= n_target:
                break
            if pools[b]:
                picked.append(pools[b].pop())
    return df.loc[picked].drop(columns="_bin").sort_values("tile_id").reset_index(drop=True)


def _ids(path: Path) -> list[str]:
    return [t.strip() for t in path.read_text().split() if t.strip()]


def cmd_pool(args, root: Path, base: Path) -> None:
    """Write the candidate tile pool and its interleaved shards under base/candidates/."""
    tiles = gpd.read_file(root / TILE_GRID)
    n_all = len(tiles)
    tiles = tiles[tiles["hop_distance"] == 0]
    b = tiles.geometry.bounds
    tiles = tiles[(b["maxx"] - b["minx"]) <= ANTIMERIDIAN_WIDTH_DEG]
    ids = sorted(int(t) for t in tiles["tile_id"])
    out = base / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_tile_ids.txt").write_text("".join(f"{t}\n" for t in ids), newline="\n")
    for k in range(args.n_shards):  # interleaved, so every shard gets a similar mix of tile sizes
        (out / f"shard_{k:03d}.txt").write_text("".join(f"{t}\n" for t in ids[k::args.n_shards]), newline="\n")
    print(f"{n_all} tiles -> {len(ids)} candidates (hop_distance 0, not antimeridian-crossing) "
          f"in {args.n_shards} shard(s) -> {out}")


def screen_tile(tile_dir: Path) -> dict:
    """Check one tile's production inputs; the geometry stats are for the figures only."""
    tid = int(tile_dir.name)
    inputs = tile_dir / "inputs"
    for f in ("mask.tif", "tile_geometry.gpkg", "boundaries_RP100_SLR_0.gpkg"):
        if not (inputs / f).exists():
            return {"tile_id": tid, "ok": False, "reason": f"no model_outputs inputs/{f}"}
    with rasterio.open(inputs / "mask.tif") as src:
        mask = src.read(1)
        b = src.bounds
    n_st = len(gpd.read_file(inputs / "boundaries_RP100_SLR_0.gpkg"))
    rec = {"tile_id": tid, "lon": (b.left + b.right) / 2, "lat": (b.bottom + b.top) / 2, "n_cells": int(mask.size),
           "ocean_frac": float(np.mean(mask == OCEAN_CODE)), "river_frac": float(np.mean(mask == RIVER_CODE)),
           "n_boundary_points": n_st}
    if n_st == 0:
        return {**rec, "ok": False, "reason": "no COAST-RP station"}
    return {**rec, "ok": True, "reason": None}


def cmd_screen(args, root: Path, base: Path) -> None:
    """Screen one shard's tiles and write the ids that passed.

    Results are cached in candidates/screen/{tile}.json (reused unless --overwrite).
    """
    out = base / "candidates" / "screen"
    out.mkdir(parents=True, exist_ok=True)
    passed = []
    ids = _ids(Path(args.tile_ids_file))
    for k, tid in enumerate(ids, 1):
        p = out / f"{tid}.json"
        if p.exists() and not args.overwrite:
            rec = json.loads(p.read_text())
        else:
            try:
                rec = screen_tile(root / "model_outputs" / tid)
            except Exception as e:  # noqa: BLE001 - a broken tile is screened out, not fatal
                rec = {"tile_id": int(tid), "ok": False, "reason": f"{type(e).__name__}: {e}"[:300]}
            p.write_text(json.dumps(rec))
        if rec["ok"]:
            passed.append(tid)
        if k % 50 == 0 or k == len(ids):
            print(f"  [{k}/{len(ids)}] {len(passed)} passed", flush=True)
    Path(args.out_ids).write_text("".join(f"{t}\n" for t in passed), newline="\n")
    print(f"screen: {len(passed)}/{len(ids)} passed -> {args.out_ids}")


def _read_jsons(d: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
    return pd.DataFrame(rows)


def _plot_map(gdf: gpd.GeoDataFrame, all_tiles: gpd.GeoDataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.5))
    all_tiles.plot(ax=ax, color="#ececea", edgecolor="none")
    for cat in reversed(list(CATEGORY_STYLE)):
        sub = gdf[gdf["category"] == cat]
        colour, label = CATEGORY_STYLE[cat]
        if len(sub):
            sub.plot(ax=ax, color=colour, edgecolor="none", label=f"{label} ({len(sub)})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_STYLE[c][0]) for c in CATEGORY_STYLE]
    labels = [f"{CATEGORY_STYLE[c][1]} ({int((gdf['category'] == c).sum())})" for c in CATEGORY_STYLE]
    ax.legend(handles, labels, loc="lower left", fontsize=10, frameon=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Final SFINCS calibration - tile selection ({int((gdf['category'] == 'selected').sum())} selected "
                 f"of {len(gdf)} candidates)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_histograms(funnel: pd.DataFrame, out: Path) -> None:
    groups = [("screened", funnel[funnel["screen_ok"]], "#b3b3ad"),
              ("eligible", funnel[funnel["category"].isin(["eligible", "selected"])], "#7fb3e8"),
              ("selected", funnel[funnel["category"] == "selected"], "#2a78d6")]
    cols = [("lat", "tile centroid latitude (deg)", None), ("ocean_frac", "ocean fraction", None),
            ("n_cells", "native cells (M)", 1e-6), ("n_forcing_stations", "forcing stations on the line", None)]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.6))
    for ax, (col, label, scale) in zip(axes, cols):
        data = [g[col].dropna().to_numpy(dtype=float) * (scale or 1) for _, g, _ in groups if col in g]
        if not data or not any(len(d) for d in data):
            ax.set_visible(False)
            continue
        edges = np.histogram_bin_edges(np.concatenate(data), bins=30)
        for (name, g, colour), d in zip(groups, data):
            ax.hist(d, bins=edges, color=colour, alpha=0.75, label=f"{name} ({len(d)})")
        ax.set_xlabel(label)
    axes[0].set_ylabel("tiles")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def cmd_select(args, root: Path, base: Path) -> None:
    """Classify every candidate by funnel stage, sample the eligible ones and write the selection."""
    cand = [int(t) for t in _ids(base / "candidates" / "candidate_tile_ids.txt")]
    funnel = pd.DataFrame({"tile_id": cand})
    screen = _read_jsons(base / "candidates" / "screen")
    lines = _read_jsons(base / "boundary_lines" / "status")
    pre = _read_jsons(base / "boundary_lines" / "preflight")
    funnel = funnel.merge(screen.rename(columns={"ok": "screen_ok", "reason": "screen_reason"}), on="tile_id", how="left")
    screened = funnel["screen_ok"].notna()
    if len(lines):
        keep = [c for c in ("tile_id", "status", "suitable", "n_used_stations", "n_forcing_stations",
                            "land_inactive_pct", "ocean_active_pct", "ocean_enclosed_pct") if c in lines]
        funnel = funnel.merge(lines[keep].rename(columns={"status": "line_status", "suitable": "line_suitable"}),
                              on="tile_id", how="left")
    if len(pre):
        funnel = funnel.merge(pre.rename(columns={"ok": "preflight_ok", "error": "preflight_error"}), on="tile_id", how="left")
    for c in ("screen_ok", "line_suitable", "preflight_ok"):
        funnel[c] = funnel[c].fillna(False).astype(bool) if c in funnel else False

    funnel["category"] = np.where(screened, "screened_out", "not_screened")
    funnel.loc[funnel["screen_ok"], "category"] = "line_unsuitable"
    funnel.loc[funnel["screen_ok"] & funnel["line_suitable"], "category"] = "preflight_failed"
    eligible_mask = funnel["screen_ok"] & funnel["line_suitable"] & funnel["preflight_ok"]
    funnel.loc[eligible_mask, "category"] = "eligible"
    eligible = funnel[eligible_mask]
    selected = stratified_sample_exact(eligible, min(args.n_tiles, len(eligible)), GRID_DEG, args.seed)
    funnel.loc[funnel["tile_id"].isin(selected["tile_id"]), "category"] = "selected"

    n_unscreened = int((~screened).sum())
    print(f"{len(funnel)} candidates: {int(funnel['screen_ok'].sum())} screened ok, "
          f"{int((funnel['screen_ok'] & funnel['line_suitable']).sum())} with a suitable line, "
          f"{len(eligible)} pass the SFINCS mask pre-flight -> {len(selected)} selected"
          f"{f' ({n_unscreened} never screened)' if n_unscreened else ''}")
    if len(selected) < args.n_tiles:
        print(f"WARNING: only {len(selected)} eligible tile(s), fewer than --n-tiles {args.n_tiles}")

    (base / "candidates").mkdir(parents=True, exist_ok=True)
    funnel.to_csv(base / "candidates" / "candidate_funnel.csv", index=False)
    selected.to_csv(base / "tile_selection_metadata.csv", index=False)
    (base / "tile_ids.txt").write_text("".join(f"{t}\n" for t in selected["tile_id"]), newline="\n")

    all_tiles = gpd.read_file(root / TILE_GRID)
    all_tiles["tile_id"] = all_tiles["tile_id"].astype(int)
    gdf = all_tiles[["tile_id", "geometry"]].merge(funnel[["tile_id", "category"]], on="tile_id")
    gdf.to_file(base / "tile_selection.gpkg", layer="tiles", driver="GPKG")
    fig_dir = base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_map(gdf, all_tiles, fig_dir / "tile_selection_map.png")
    _plot_histograms(funnel, fig_dir / "tile_selection_histograms.png")
    print(f"wrote {base / 'tile_ids.txt'}, tile_selection_metadata.csv, tile_selection.gpkg, "
          f"candidates/candidate_funnel.csv, figures/tile_selection_{{map,histograms}}.png")
    print(funnel["category"].value_counts().to_string())


def main() -> None:
    """Run the requested pool/screen/select subcommand."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["pool", "screen", "select"])
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--n-shards", type=int, default=10, help="pool: number of stage-1 jobs")
    parser.add_argument("--tile-ids-file", help="screen: this shard's candidate ids")
    parser.add_argument("--out-ids", help="screen: where to write the ids that passed")
    parser.add_argument("--overwrite", action="store_true", help="screen: redo tiles already screened")
    parser.add_argument("--n-tiles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = read_root(Path(args.config))
    base = root / args.base_dir_name
    {"pool": cmd_pool, "screen": cmd_screen, "select": cmd_select}[args.command](args, root, base)


if __name__ == "__main__":
    main()
