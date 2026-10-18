# Contributing

## Setup

Create the `gfm` env (pipeline, analysis, validation, all plotting) and, for `sfincs_tiles/` model build/run only, `hydromt-sfincs-dev`; see [docs/install.md](docs/install.md).

## Checks

Both must pass before a change is merged:

```
pytest -m "not needs_data"                      # in gfm
pytest -m "not needs_data" tests/sfincs_*       # in hydromt-sfincs-dev, when touching sfincs_tiles/
ruff check .
```

## Rules

- Changes are behaviour-preserving unless a result change has been agreed beforehand.
- `src/` is a flat library: scripts reach it with `sys.path.insert(0, ...)`, following the sibling file's pattern; no packaging or relative imports.
- Tests never write to the shared data root (`<data_root>`); write to `tmp_path`.
- Machine values go in the gitignored `config_local.yml`, cluster values in the gitignored site files `config_hpc.yml` and `sfincs_tiles/site_env.sh` ([docs/hpc.md](docs/hpc.md)); never in `config.yml`.

## Documentation

Follow the "Documentation standard" section of [CLAUDE.md](CLAUDE.md). In short:

- Docs and comments describe the code as it is: no change history, dates or "now/previously"; git is the changelog.
- Docstrings: one imperative summary line; add lines only for what the signature does not say (units, datums, sign conventions, assumptions, references).
- Comments explain why, not what.
- No new `.md` files unless agreed.
- `docs/data.md` is generated: edit the data catalogs and run `python docs/make_data_md.py`.
