# Installation

## Environments

Two conda environments, both pinned to exact versions:

| Env | File | Used for |
|---|---|---|
| `gfm` | `environment.yml` | pipeline, preparation, analysis, validation and all plotting (including the `sfincs_tiles/` plot scripts) |
| `hydromt-sfincs-dev` | `environment_sfincs.yml` | `sfincs_tiles/` SFINCS model build and run only |

```
mamba env create -f environment.yml
conda activate gfm
```

`hydromt-sfincs-dev` needs an editable install of a pinned hydromt_sfincs checkout, which is not on conda-forge at this version:

```
mamba env create -f environment_sfincs.yml
git clone https://github.com/Deltares/hydromt_sfincs.git
git -C hydromt_sfincs checkout cbba4362
conda run -n hydromt-sfincs-dev pip install -e ./hydromt_sfincs --no-deps
```

matplotlib's `savefig()` crashes in `hydromt-sfincs-dev`: never plot in it, run plotting scripts in `gfm`.

After editing an environment file: `mamba env update -f environment.yml --prune`.

**Windows:** if `python` resolves to the Microsoft Store stub, call the env's interpreter by full path:

```
<env>/python.exe <script>.py
```
```
C:/Users/me/miniforge3/envs/gfm/python.exe preparation/run_preparation.py
```

## Local configuration

`paths.root` (`<data_root>`, data and outputs) and `paths.code_root` (`<code_root>`, this checkout) have no default; set them in the gitignored `config_local.yml`:

```
cp snakemake_workflow/config/config_local.yml.example snakemake_workflow/config/config_local.yml
```

It is loaded automatically by Snakemake and every script, and holds only the keys you override; never put machine-specific values in `config.yml`.
On a SLURM cluster, also create the HPC site files ([hpc.md](hpc.md)).

## Tests

```
pytest -m "not needs_data"                      # in gfm
pytest -m "not needs_data" tests/sfincs_*       # in hydromt-sfincs-dev
```

`needs_data` marks tests that read real input data under `<data_root>`; they skip themselves when the data root is absent, and the marker lets you deselect them.
`tests/sfincs_*` are collected only when hydromt_sfincs 2.x is installed, so the `gfm` run skips them.
Each test script also runs standalone with `python <script>.py`.

## Lint

```
ruff check .
```

Configuration is in `pyproject.toml`; the check must stay clean.
