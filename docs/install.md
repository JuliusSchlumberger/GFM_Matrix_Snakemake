# Installation

## Environments

Install [Miniforge](https://github.com/conda-forge/miniforge), which provides `conda` and `mamba` (either works below).
The pipeline runs on Linux (the cluster) and Windows; both environments are pinned to exact versions, including Python (3.10 in `gfm`, 3.11 in `hydromt-sfincs-dev`).

| Env | File | Used for |
|---|---|---|
| `gfm` | `environment.yml` | pipeline, preparation, analysis, validation and all plotting (including the `sfincs_tiles/` plot scripts); has pytest and ruff |
| `hydromt-sfincs-dev` | `environment_sfincs.yml` | `sfincs_tiles/` SFINCS model build and run only; has neither pytest nor ruff |

```
conda env create -f environment.yml
conda activate gfm
```

Only the SFINCS calibration needs `hydromt-sfincs-dev`.
It needs an editable install of a pinned hydromt_sfincs checkout, which is not on conda-forge at this version.
Clone it next to this checkout, not inside it: `ruff check .` would lint it, and `.gitignore` does not exclude it.

```
conda env create -f environment_sfincs.yml
git clone https://github.com/Deltares/hydromt_sfincs.git ../hydromt_sfincs
git -C ../hydromt_sfincs checkout cbba4362
conda run -n hydromt-sfincs-dev pip install -e ../hydromt_sfincs --no-deps
```

matplotlib's `savefig()` crashes in `hydromt-sfincs-dev`: never plot in it, run plotting scripts in `gfm`.

After editing an environment file: `conda env update -f environment.yml --prune`.

## SFINCS

The eikonal pipeline does not need SFINCS.
The calibration runs it on the cluster from a container image (`SFINCS_IMAGE`, default `docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release`) with Apptainer (`CONTAINER_CMD`); both are set in `sfincs_tiles/site_env.sh` ([hpc.md](hpc.md#one-time-site-setup)).
Locally, only the optional smoke test `tests/sfincs_synthetic_end_to_end/test_sfincs_smoke.py` uses a SFINCS executable, set as `paths.sfincs_exe` in `config_local.yml`; it skips when that is unset.

## Shell

Commands in these docs are bash (`export`, `$VAR`, `\` line continuations, `VAR=x cmd`); on Windows run them in Git Bash.
In PowerShell, set a variable with `$env:NAME = "value"` instead of `export NAME=value`, and join continued lines.

**Windows:** if `python` resolves to the Microsoft Store stub (the env is not activated), call the env's executables by full path:

```
<env>/python.exe <script>.py
<env>/Scripts/snakemake.exe <target> --cores 4
```
```
C:/Users/me/miniforge3/envs/gfm/python.exe preparation/run_preparation.py
C:/Users/me/miniforge3/envs/gfm/Scripts/snakemake.exe preprocess --cores 4
```

## Local configuration

`paths.root` (`<data_root>`, data and outputs) and `paths.code_root` (`<code_root>`, this checkout) have no default; set them in the gitignored `config_local.yml`:

```
cp snakemake_workflow/config/config_local.yml.example snakemake_workflow/config/config_local.yml
```

It holds only the keys you override, with the same nesting as `config.yml`, and is loaded automatically by Snakemake and every script.
Use forward slashes on Windows too; `<data_root>` must hold the inputs of [data.md](data.md), and outputs are created under it as needed.

```
paths:
  root: "<data_root>"
  code_root: "<code_root>"
validation:
  benchmark_root: "<benchmark_root>"   # optional, default <data_root>/inputs/validation
```
```
paths:
  root: "D:/gfm_data"
  code_root: "C:/Users/me/gfm"
```

Never put machine-specific values in `config.yml`.
On a SLURM cluster, also create the HPC site files ([hpc.md](hpc.md)).

## Tests

```
pytest -m "not needs_data"                      # in gfm
pytest -m "not needs_data" tests/sfincs_*       # in hydromt-sfincs-dev
```

The `gfm` tests need no data and no `config_local.yml`.
`needs_data` marks tests that read real input data under `<data_root>`; they skip themselves when the data root is absent, and the marker lets you deselect them.
`tests/sfincs_*` are collected only when hydromt_sfincs 2.x is installed, so the `gfm` run skips them.
`hydromt-sfincs-dev` has no pytest: install it there (`conda install -n hydromt-sfincs-dev pytest`) or run each test script standalone with `python <script>.py`.

## Lint

```
ruff check .
```

Run it in `gfm`; configuration is in `pyproject.toml`, and the check passes on a clean checkout.
