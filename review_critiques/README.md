# review_critiques

Experiments answering the reviewer critiques on SLR propagation, land-cover friction and
event volume. Inputs are read from `<data_root>` (read-only); outputs go to `--out-dir`.

| Script | Critique | What it does |
|---|---|---|
| `run_two_component_bangkok.py` | 1a | Re-solves the Bangkok / Chao Phraya tiles with two-component SLR propagation |
| `plot_two_component_bangkok.py` | 1a | Flooded area vs SLR, depth maps, water-level profiles (attenuated vs two-component) |
| `plot_protection_rule.py` | 1a | Effective protection RP vs SLR; case EAI and rule-disagreement maps (flood-fraction vs water-level rule) |
| `calibration_experiments.py` | 2, 3 | Scores land-cover, uniform, volume-cap (A) and duration-friction (B) variants against the SFINCS RP100 calibration tiles |
| `plot_calibration_experiments.py` | 2, 3 | Harness check, pooled / per-tile metrics, basin statistics, maps |

Run in the `gfm` env, e.g.

```
python review_critiques/calibration_experiments.py --out-dir <dir> --n-random 60 --workers 2
python review_critiques/plot_calibration_experiments.py --results-dir <dir> --fig-dir <dir>/figures
```

Both runners append to a log in `--out-dir` and skip finished scenarios / tiles on restart.
