"""One front door onto the HPC: every job chain, check and helper of the pipelines, by scenario name.

Scenario NAME means snakemake_workflow/config/<NAME>_materialized.yml (`production`: config.yml);
`--config PATH` takes any config file instead. Generators write job scripts, a submit script and its
jobs.json, and print the submit script; `--submit` also runs it. See docs/hpc.md.

Usage (on the cluster login node, with the gfm env's python):
    python hpc/gfm_hpc.py run preprocess|simulate|postprocess|exposure|flood-totals --scenario NAME [--submit]
    python hpc/gfm_hpc.py product [--submit]
    python hpc/gfm_hpc.py studies sweep|obstacle|wet-tiles|figures --source landcover [uniform] [--submit]
    python hpc/gfm_hpc.py sfincs stage1|stage2|post|rerun [...] [--submit]
    python hpc/gfm_hpc.py validate --scenario NAME [--only GROUP ...] [--submit]
    python hpc/gfm_hpc.py progress preprocess|simulate|postprocess|sfincs --scenario NAME [--watch S]
    python hpc/gfm_hpc.py failures --scenario NAME
    python hpc/gfm_hpc.py config NAME [--hpc] [--key dotted.key]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "snakemake_workflow" / "config"
for _sub in ("src", "snakemake_workflow/scripts", "sfincs_tiles", "calibration_studies"):
    sys.path.insert(0, str(REPO / _sub))

STUDIES = {"sweep": "sweep_budget", "obstacle": "obstacle_coupling"}
PROGRESS = {"preprocess": "check_preprocess_progress.py", "simulate": "check_simulation_progress.py",
            "postprocess": "check_postprocess_progress.py", "sfincs": "sfincs_tiles/check_calibration_progress.py"}


# ── scenarios and configs ─────────────────────────────────────────────────────


def scenario_config(name: str, variant: str | None = None, config_dir: Path = CONFIG_DIR) -> Path:
    """Return the config file of scenario `name` (and flood-model `variant`); warn when it is stale.

    A materialized file older than its override, config.yml or config_local.yml misses their
    latest edits: rerun materialize_configs.py.
    """
    if name == "production":
        if variant:
            raise SystemExit("production has no variant configs")
        return config_dir / "config.yml"
    stem = f"{name}__{variant}" if variant else name
    path = config_dir / f"{stem}_materialized.yml"
    if not path.exists():
        raise SystemExit(f"{path} not found: python snakemake_workflow/scripts/materialize_configs.py {name}"
                         + (f" --variants {variant} --selection <variant_selection.json>" if variant else ""))
    sources = [config_dir / f"{name}.yml", config_dir / "config.yml", config_dir / "config_local.yml"]
    newer = [s.name for s in sources if s.exists() and s.stat().st_mtime > path.stat().st_mtime]
    if newer:
        print(f"WARNING: {path.name} is older than {', '.join(newer)}: rerun "
              f"python snakemake_workflow/scripts/materialize_configs.py {name}", file=sys.stderr)
    return path


def _config_from(args) -> Path:
    if getattr(args, "config", None):
        return Path(args.config)
    if getattr(args, "scenario", None):
        return scenario_config(args.scenario, getattr(args, "variant", None))
    raise SystemExit("give --scenario NAME or --config PATH")


def hpc_view(config_path: Path) -> Path:
    """Write the config as the compute nodes see it to <hpc.state_root>/configs/; return the path."""
    import yaml
    from config_utils import atomic_write, load_hpc_config

    config = load_hpc_config(config_path)
    state_root = config["hpc"].get("state_root")
    if not state_root:
        raise SystemExit("hpc.state_root is not set (config_hpc.yml)")
    out = Path(state_root) / "configs" / Path(config_path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, lambda f: yaml.safe_dump(config, f), encoding="utf-8", newline="")
    return out


def _lookup(config: dict, dotted: str):
    value = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise SystemExit(f"{dotted}: no such key")
        value = value[part]
    return value


def _python(script: str, *args) -> None:
    subprocess.run([sys.executable, str(REPO / script), *[str(a) for a in args]], cwd=REPO, check=True)


def _finish(submit: str | None, args) -> None:
    from hpc_common import submit_script

    if submit is None:
        return
    if args.submit:
        submit_script(submit)
    else:
        print(f"Submit on the cluster with: bash {submit}")


def _inline_job(config_path: Path, name: str, profile: str, argv: list[str], profile_sets: list[str]) -> str:
    """Write a one-job chain running `gfm_hpc.py <argv>` on a compute node; return its submit script."""
    from config_utils import load_config, load_hpc_config
    from hpc_common import JobChain, apply_profile_sets

    local, linux = load_config(config_path), load_hpc_config(config_path)
    hpc = apply_profile_sets(linux["hpc"], profile_sets)
    local_dir = Path(local["hpc"]["jobs_dir"])
    (local_dir / "logs").mkdir(parents=True, exist_ok=True)
    code_root = linux["paths"]["code_root"]
    node_config = str(hpc_view(config_path))  # complete in itself, wherever config_path lives
    chain = JobChain(hpc, local_dir, hpc["jobs_dir"])
    chain.add(name, profile, [f'cd "{code_root}"',
                              " ".join(['"$GFM_PY" hpc/gfm_hpc.py', *argv, "--config", node_config])])
    return chain.write_submit(f"submit_{name}.sh")


# ── commands ──────────────────────────────────────────────────────────────────


def cmd_run(args) -> None:
    config = _config_from(args)
    sets = args.profile_set
    if args.stage != "preprocess" and (args.fuse or args.preprocess_only):
        raise SystemExit("--fuse and --preprocess-only belong to run preprocess")
    if args.resume and args.stage not in ("simulate", "postprocess", "exposure"):
        raise SystemExit("--resume belongs to run simulate|postprocess|exposure")
    if args.stage == "preprocess":
        import generate_hpc_preprocess_job as gen
        submit = gen.generate(config, fuse=args.fuse, preprocess_only=args.preprocess_only, profile_sets=sets)
    elif args.stage == "simulate":
        import generate_hpc_simulation_jobs as gen
        submit = gen.generate(config, resume=args.resume, profile_sets=sets)
    elif args.stage == "postprocess":
        import generate_hpc_postprocess_job as gen
        submit = gen.generate(config, resume=args.resume, flood_totals=not args.no_flood_totals, profile_sets=sets)
    elif args.stage == "exposure":
        import generate_hpc_exposure_jobs as gen
        submit = gen.generate(config, resume=args.resume, profile_sets=sets)
    else:
        import generate_hpc_flood_totals_and_plots_job as gen
        submit = gen.generate(config, profile_sets=sets)
    _finish(submit, args)


def cmd_product(args) -> None:
    config = CONFIG_DIR / "config.yml"
    if args.submit:
        _finish(_inline_job(config, "storm_tide_product", "product", ["product"], args.profile_set), args)
        return
    _python("preparation/run_preparation.py", "storm_tide_product", "boundary_conditions",
            "--config", hpc_view(Path(args.config) if args.config else config))


def cmd_studies(args) -> None:
    from config_utils import load_config

    config = _config_from(args)
    study_root = Path(load_config(config)["paths"]["root"]) / args.study_dir_name
    if args.action in STUDIES:
        import generate_calibration_batch_jobs as gen
        for source in args.source:
            args.friction_source = source
            _finish(gen.generate(STUDIES[args.action], config, **gen.study_kwargs(args)), args)
    elif args.action == "wet-tiles":
        for source in args.source:
            _python("calibration_studies/aggregate_wet_tiles.py", study_root / f"sweep_budget_{source}",
                    "--n-tiles-wanted", args.n_tiles_wanted)
    elif args.submit:
        argv = ["studies", "figures", "--source", *args.source, "--study-dir-name", args.study_dir_name]
        _finish(_inline_job(config, f"study_figures_{'_'.join(args.source)}", "study_figures", argv,
                            args.profile_set), args)
    else:
        for source in args.source:
            figs = study_root / "figures" / source
            _python("calibration_studies/plot_sweep_calibration_bands.py", study_root / f"sweep_budget_{source}",
                    "--fig-dir", figs)
            _python("calibration_studies/plot_sweep_time_vs_size.py", study_root / f"sweep_budget_{source}",
                    "--fig-dir", figs)
            _python("calibration_studies/plot_obstacle_coupling_calibration_bands.py",
                    study_root / f"obstacle_coupling_{source}", "--fig-dir", figs)


def cmd_sfincs(args) -> None:
    import generate_calibration_jobs as gen

    args.config = str(_config_from(args)) if (args.config or args.scenario) else str(CONFIG_DIR / "config.yml")
    submit = gen.run(args.command, args)
    if submit and not args.submit:
        print(f"Submit on the cluster with: bash {submit}")


def validation_runs(config: dict, only: list[str] | None) -> list[list[str]]:
    """Return the commands of the config's validation.runs groups (all, or the `only` ones, in order)."""
    runs = (config.get("validation") or {}).get("runs") or {}
    unknown = sorted(set(only or []) - set(runs))
    if unknown:
        raise SystemExit(f"validation.runs has no group(s) {unknown} (known: {', '.join(runs)})")
    return [list(map(str, cmd)) for group, cmds in runs.items() if not only or group in only for cmd in cmds]


def expand_run(cmd: list[str], own_config: Path) -> list[str]:
    """Replace {config} (this scenario) and {config:NAME} with HPC-view config paths."""
    out = []
    for arg in cmd:
        if arg == "{config}":
            arg = str(hpc_view(own_config))
        elif arg.startswith("{config:") and arg.endswith("}"):
            arg = str(hpc_view(scenario_config(arg[len("{config:"):-1])))
        out.append(arg)
    return out


def cmd_validate(args) -> None:
    from config_utils import load_hpc_config

    config = _config_from(args)
    if args.submit:
        argv = ["validate", *(["--only", *args.only] if args.only else [])]
        name = f"validate_{Path(config).stem.replace('_materialized', '')}" + "".join(f"_{g}" for g in args.only or [])
        _finish(_inline_job(config, name, "validation", argv, args.profile_set), args)
        return
    commands = validation_runs(load_hpc_config(config), args.only)
    if not commands:
        raise SystemExit(f"{config} has no validation.runs")
    for cmd in commands:
        full = expand_run(cmd, config)
        print("$ " + " ".join(full), flush=True)
        _python(*full)


def cmd_progress(args, extra: list[str]) -> None:
    config = _config_from(args) if (args.config or args.scenario) else CONFIG_DIR / "config.yml"
    _python(PROGRESS[args.pipeline], "--config", config, *(["--watch", args.watch] if args.watch else []), *extra)


def cmd_failures(args) -> None:
    _python("snakemake_workflow/scripts/collect_hpc_failures.py", "--config", _config_from(args),
            *(["--out", args.out] if args.out else []))


def cmd_config(args) -> None:
    import yaml
    from config_utils import load_config, load_hpc_config

    path = Path(args.config) if args.config else scenario_config(args.name, args.variant)
    if args.key:
        value = _lookup((load_hpc_config if args.hpc else load_config)(path), args.key)
        print(value if not isinstance(value, (dict, list)) else yaml.safe_dump(value, sort_keys=False).rstrip())
    elif args.hpc:
        print(hpc_view(path))
    else:
        print(path)


def cmd_check_targets(args) -> None:
    """Exit 1 unless every path listed in the target files exists (the dispatch job's gate)."""
    from hpc_common import missing_paths

    paths = [line.strip() for f in args.files for line in Path(f).read_text().splitlines() if line.strip()]
    n, first = missing_paths(paths)
    if n:
        print(f"{n} of {len(paths)} preprocessing output(s) missing, e.g.:", *first, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)
    print(f"all {len(paths)} preprocessing output(s) present")


# ── command line ──────────────────────────────────────────────────────────────


def _scenario_args(p: argparse.ArgumentParser, default: str | None = None) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scenario", default=default, help="config/<NAME>_materialized.yml; production = config.yml")
    g.add_argument("--config", default=None, help="a config file instead of a scenario name")
    p.add_argument("--variant", default=None, help="flood-model variant: config/<NAME>__<VARIANT>_materialized.yml")


def build_parser() -> argparse.ArgumentParser:
    import generate_calibration_jobs as sfincs_gen
    import generate_calibration_batch_jobs as study_gen
    from hpc_common import add_profile_set_argument

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="write (and --submit) a pipeline stage's job chain")
    p.add_argument("stage", choices=["preprocess", "simulate", "postprocess", "exposure", "flood-totals"])
    _scenario_args(p)
    p.add_argument("--resume", action="store_true", help="simulate/postprocess/exposure: only what is missing")
    p.add_argument("--preprocess-only", action="store_true", help="preprocess: no simulation dispatch")
    p.add_argument("--fuse", action="store_true", help="preprocess: every tile hop 0, simulate in the same job")
    p.add_argument("--no-flood-totals", action="store_true", help="postprocess: end after the exposure analysis")
    p.add_argument("--submit", action="store_true")
    add_profile_set_argument(p)

    p = sub.add_parser("product", help="rebuild the storm-tide product + boundary conditions (production)")
    p.add_argument("--config", default=None, help="run inline with this config (as the job does)")
    p.add_argument("--submit", action="store_true", help="as a job (profile product)")
    add_profile_set_argument(p)

    p = sub.add_parser("studies", help="eikonal solver studies (calibration_500_tiles)")
    p.add_argument("action", choices=[*STUDIES, "wet-tiles", "figures"])
    _scenario_args(p, default="calibration_500_tiles")
    study_gen.add_study_arguments(p)
    p.add_argument("--source", nargs="+", choices=["landcover", "uniform"], required=True,
                   help="friction source(s); one chain per source")
    p.add_argument("--n-tiles-wanted", type=int, default=500, help="wet-tiles")
    p.add_argument("--submit", action="store_true")

    p = sub.add_parser("sfincs", help="SFINCS calibration chain")
    ssub = p.add_subparsers(dest="command", required=True)
    for command in sfincs_gen.COMMANDS:
        sp = ssub.add_parser(command)
        _scenario_args(sp)
        sfincs_gen.add_arguments(sp, command)

    p = sub.add_parser("validate", help="run the scenario's validation.runs")
    _scenario_args(p)
    p.add_argument("--only", nargs="+", default=None, help="validation.runs groups")
    p.add_argument("--submit", action="store_true", help="as a job (profile validation)")
    add_profile_set_argument(p)

    p = sub.add_parser("progress", help="the check_*_progress.py monitors (extra options pass through)")
    p.add_argument("pipeline", choices=list(PROGRESS))
    _scenario_args(p)
    p.add_argument("--watch", default=None, metavar="SECONDS")

    p = sub.add_parser("failures", help="collect_hpc_failures.py")
    _scenario_args(p)
    p.add_argument("--out", default=None)

    p = sub.add_parser("config", help="print a scenario's config path, the HPC view (--hpc) or one value")
    p.add_argument("name", nargs="?", default="production")
    p.add_argument("--variant", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--hpc", action="store_true", help="the compute nodes' view (config_hpc.yml layered)")
    p.add_argument("--key", default=None, help="dotted key, e.g. hpc.jobs_dir")

    p = sub.add_parser("check-targets", help="exit 1 unless every path in the target files exists")
    p.add_argument("files", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> None:
    args, extra = build_parser().parse_known_args(argv)
    if extra and args.cmd != "progress":
        raise SystemExit(f"unrecognized arguments: {' '.join(extra)}")
    if args.cmd == "progress":
        cmd_progress(args, extra)
        return
    {"run": cmd_run, "product": cmd_product, "studies": cmd_studies, "sfincs": cmd_sfincs,
     "validate": cmd_validate, "failures": cmd_failures, "config": cmd_config,
     "check-targets": cmd_check_targets}[args.cmd](args)


if __name__ == "__main__":
    main()
