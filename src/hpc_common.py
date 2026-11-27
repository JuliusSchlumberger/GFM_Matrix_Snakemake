"""One writer for every generated SLURM job: resource profiles, header, env, packing, submit chain."""

from __future__ import annotations

import json
import posixpath
from pathlib import Path

PROFILE_FIELDS = ("partition", "account", "time", "mem", "cpus_per_task")
# A job's parallelism is its processes (process_pool.available_cpus); one BLAS/OpenMP thread each
# keeps N processes on N cores. SFINCS sets its own OMP_NUM_THREADS (run_final_tile.sh).
THREAD_ENV = "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1"
SNAKEMAKE = '"$GFM_PY" -m snakemake'


def resolve_profile(hpc_cfg: dict, name: str, overrides: dict | None = None) -> dict:
    """Return the resources of profile `name`: hpc.profiles.default < profiles[name] < `overrides`.

    Empty or null values fall back to the layer below; `account` starts from hpc.account.
    An empty partition or account omits that #SBATCH line.
    """
    profiles = hpc_cfg.get("profiles") or {}
    if name not in profiles:
        raise KeyError(f"hpc.profiles.{name} is not defined: re-materialize the scenario configs or regenerate "
                       "resolved_config.yml from the current config.yml + config_hpc.yml")
    out = {"account": hpc_cfg.get("account") or ""}
    for layer in (profiles.get("default") or {}, profiles[name] or {}, overrides or {}):
        out.update({k: v for k, v in layer.items() if v not in (None, "")})
    missing = [k for k in ("time", "mem", "cpus_per_task") if not out.get(k)]
    if missing:
        raise ValueError(f"hpc.profiles.{name}: {', '.join(missing)} not set (config_hpc.yml)")
    return out


def apply_profile_sets(hpc_cfg: dict, items: list[str] | None) -> dict:
    """Apply `PROFILE.FIELD=VALUE` overrides to hpc.profiles in place; return `hpc_cfg`.

    Written into the resolved_config.yml a chain carries, so jobs it submits later keep them.
    """
    profiles = hpc_cfg.setdefault("profiles", {})
    for item in items or []:
        key, sep, value = item.partition("=")
        name, dot, field = key.partition(".")
        if not sep or not dot:
            raise ValueError(f"--profile-set {item!r}: expected PROFILE.FIELD=VALUE")
        if name not in profiles:
            raise ValueError(f"--profile-set {item!r}: unknown profile {name!r} (known: {', '.join(profiles)})")
        if field not in PROFILE_FIELDS:
            raise ValueError(f"--profile-set {item!r}: field must be one of {', '.join(PROFILE_FIELDS)}")
        profiles[name] = dict(profiles[name] or {})
        profiles[name][field] = int(value) if field == "cpus_per_task" else value
    return hpc_cfg


def env_lines(hpc_cfg: dict, env: str = "gfm") -> list[str]:
    """Return the interpreter exports, PATH and common env lines every job script starts with.

    `env` ("gfm" or "sfincs") puts that env's bin directory first on PATH.
    """
    python = hpc_cfg.get("python") or {}
    if not python.get("gfm"):
        raise ValueError("hpc.python.gfm is not set: copy config_hpc.example.yml to config_hpc.yml and fill it in")
    if not python.get(env):
        raise ValueError(f"hpc.python.{env} is not set (config_hpc.yml)")
    lines = [f'export GFM_PY="{python["gfm"]}"']
    if python.get("sfincs"):
        lines.append(f'export SFINCS_PY="{python["sfincs"]}"')
    lines.append(f'export PATH="{posixpath.dirname(python[env])}:$PATH"')
    return lines + [THREAD_ENV, *(hpc_cfg.get("job_env") or [])]


def job_lines(
    hpc_cfg: dict,
    profile: str,
    name: str,
    body: list[str],
    log_dir: str,
    *,
    resources: dict | None = None,
    errexit: bool = True,
    env: str = "gfm",
    exports: dict | None = None,
    helpers: tuple[str, ...] = (),
) -> tuple[list[str], dict]:
    """Return a job script's lines and its resolved resources.

    Header from profile `profile` (+ `resources`), logs `<log_dir>/<name>_%j.out/.err`.
    `errexit` False writes `set -uo pipefail` for scripts whose later commands must run after an
    earlier one failed. `helpers` names bash functions to define: "retry" (run_snakemake_with_retry),
    "stage_config", "stage_tile_grid".
    """
    res = resolve_profile(hpc_cfg, profile, resources)
    lines = ["#!/bin/bash", f"#SBATCH --job-name={name}"]
    lines += [f"#SBATCH --{key}={res[key]}" for key in ("partition", "account") if res.get(key)]
    lines += [
        f"#SBATCH --time={res['time']}",
        f"#SBATCH --mem={res['mem']}",
        f"#SBATCH --cpus-per-task={res['cpus_per_task']}",
        f"#SBATCH --output={log_dir}/{name}_%j.out",
        f"#SBATCH --error={log_dir}/{name}_%j.err",
        "",
        "set -euo pipefail" if errexit else "set -uo pipefail",
        *env_lines(hpc_cfg, env),
        *[f'export {k}="{v}"' for k, v in (exports or {}).items()],
        "",
    ]
    for helper in helpers:
        lines += _HELPERS[helper]()
    return lines + list(body), res


def even_split(items: list, n: int) -> list[list]:
    """Split `items` into min(n, len) contiguous, near-equal batches, keeping their order."""
    if not items:
        return []
    n_batches = min(n, len(items))
    k, m = divmod(len(items), n_batches)
    return [items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n_batches)]


def lpt_pack(costs: dict, n: int) -> list[list]:
    """Pack the keys of `costs` into `n` buckets, longest processing time first.

    Each key, by descending cost (ties in insertion order), goes to the least-loaded bucket
    (ties to the lowest index). Buckets may stay empty when there are fewer keys than `n`.
    """
    buckets: list[list] = [[] for _ in range(n)]
    loads = [0.0] * n
    for key in sorted(costs, key=lambda k: -costs[k]):
        j = loads.index(min(loads))
        buckets[j].append(key)
        loads[j] += costs[key]
    return buckets


def missing_paths(paths, limit: int = 5) -> tuple[int, list[str]]:
    """Return how many of `paths` do not exist and the first `limit` of them.

    A path os.path.exists() misses is checked again with retries: a shared-mount blip must not
    read as a missing output.
    """
    import os

    from config_utils import path_ready

    n, first = 0, []
    for p in paths:
        if os.path.exists(p) or path_ready(p):
            continue
        n += 1
        if len(first) < limit:
            first.append(p)
    return n, first


class JobChain:
    """Job scripts plus the submit script and `<submit>.jobs.json` that queue them in order.

    Every dependency is afterany: a failed job only fails what reads its output, which a resume
    picks up.
    """

    def __init__(self, hpc_cfg: dict, local_dir: Path, linux_dir: str):
        self.hpc_cfg = hpc_cfg
        self.local_dir = Path(local_dir)
        self.linux_dir = linux_dir
        self.jobs: list[dict] = []

    def add(self, name: str, profile: str, body: list[str], *, after: list[str] = (), filename: str | None = None,
            local_dir: Path | None = None, linux_dir: str | None = None, log_dir: str | None = None,
            **job_kw) -> str:
        """Write job `name` and queue it after the jobs named in `after`; return its Linux path."""
        local_dir = Path(local_dir or self.local_dir)
        linux_dir = linux_dir or self.linux_dir
        filename = filename or f"{name}.sbatch"
        lines, res = job_lines(self.hpc_cfg, profile, name, body, log_dir or f"{linux_dir}/logs", **job_kw)
        (local_dir / filename).write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8", newline="\n")
        return self.add_existing(name, f"{linux_dir}/{filename}", after=after, profile=profile, resources=res)

    def add_existing(self, name: str, linux_script: str, *, after: list[str] = (), profile: str | None = None,
                     resources: dict | None = None) -> str:
        """Queue an already written script (resubmitted unchanged)."""
        known = {j["name"] for j in self.jobs}
        unknown = [a for a in after if a not in known]
        if unknown:
            raise ValueError(f"job {name} waits for unknown job(s) {unknown}")
        self.jobs.append({"name": name, "script": linux_script, "profile": profile, "after": list(after),
                          **{k: (resources or {}).get(k) for k in PROFILE_FIELDS}})
        return linux_script

    def names(self, prefix: str = "") -> list[str]:
        return [j["name"] for j in self.jobs if j["name"].startswith(prefix)]

    def write_submit(self, filename: str, prelude: list[str] | None = None) -> str:
        """Write the submit script and its jobs.json next to it; return the submit script's Linux path."""
        var = {j["name"]: f"J{i}" for i, j in enumerate(self.jobs)}
        lines = ["#!/bin/bash", "set -euo pipefail", "", *(prelude or [])]
        for j in self.jobs:
            dep = f" --dependency=afterany:{':'.join('$' + var[a] for a in j['after'])}" if j["after"] else ""
            v = var[j["name"]]
            lines += [f'{v}=$(sbatch --parsable{dep} "{j["script"]}")', f"{v}=${{{v}%%;*}}",
                      f'echo "submitted {j["name"]} -> job ${v}"']
        path = self.local_dir / filename
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        linux_path = f"{self.linux_dir}/{filename}"
        manifest = {"submit": linux_path, "jobs": self.jobs}
        (self.local_dir / f"{Path(filename).stem}.jobs.json").write_text(json.dumps(manifest, indent=1),
                                                                        encoding="utf-8", newline="\n")
        return linux_path


def read_targets_lines(var: str, targets_file: str) -> list[str]:
    """Return bash lines loading `targets_file` (one target per line) into array `var`.

    Use as `"${VAR[@]}"`. Unlike an unchecked `$(cat file)`, a missing or empty
    file exits the script: an empty target list would make snakemake build its
    first rule instead of nothing.
    """
    return [
        f'mapfile -t {var} < "{targets_file}" || exit 1',
        f'if [ "${{#{var}[@]}}" -eq 0 ] || [ -z "${{{var}[0]}}" ]; then',
        f'    echo "ERROR: no targets in {targets_file}" >&2',
        "    exit 1",
        "fi",
    ]


def tile_grid_override_lines() -> list[str]:
    """Bash lines pointing the staged config's tile_grid.path at the staged tile grid."""
    return [
        "\"$GFM_PY\" -c 'import sys, yaml; cfg_path, new_path = sys.argv[1], sys.argv[2]; "
        "cfg = yaml.safe_load(open(cfg_path, encoding=\"utf-8\")); cfg[\"tile_grid\"][\"path\"] = new_path; "
        "yaml.safe_dump(cfg, open(cfg_path, \"w\", encoding=\"utf-8\"))' "
        # Unchecked, a failed rewrite would silently read the tile grid off the shared data root again.
        '"$LOCAL_CONFIGFILE" "$LOCAL_TILE_GRID" || exit 1',
    ]


def retry_wrapper_lines() -> list[str]:
    """Bash fn retrying snakemake: shared data root reads can hit stale caches (UnicodeDecodeError).

    Each failure dumps checksum/mtime of the candidate config files for forensics.
    """
    return [
        "run_snakemake_with_retry() {",
        "  local attempt=1 max_attempts=10 delay=30",
        '  while [ "$attempt" -le "$max_attempts" ]; do',
        '    if "$@"; then',
        "      return 0",
        "    fi",
        '    echo "  [retry $attempt/$max_attempts] snakemake invocation failed - retrying in ${delay}s..." >&2',
        '    echo "  [forensics] candidate config files at time of failure:" >&2',
        '    sha256sum snakemake_workflow/config/config.yml snakemake_workflow/config/config_local.yml "$LOCAL_CONFIGFILE" >&2 || true',
        '    stat snakemake_workflow/config/config.yml snakemake_workflow/config/config_local.yml "$LOCAL_CONFIGFILE" >&2 || true',
        '    sleep "$delay"',
        "    attempt=$((attempt + 1))",
        "  done",
        '  echo "  snakemake invocation failed after $max_attempts attempts - giving up." >&2',
        "  return 1",
        "}",
    ]


def stage_configfile_lines() -> list[str]:
    """Bash fn staging a config file to node-local disk, validated as UTF-8 YAML.

    A compute node's stale shared-mount cache can corrupt the same file on every retry, so
    only the cheap copy+parse step is retried and later reads hit local disk.
    """
    return [
        "stage_configfile_locally() {",
        '  local src="$1" out_var="$2"',
        '  local stage_dir="${TMPDIR:-/tmp}/gfm_resolved_config"',
        '  mkdir -p "$stage_dir"',
        '  local dst="$stage_dir/resolved_config_${SLURM_JOB_ID:-$$}_$$.yml"',
        "  local attempt=1 max_attempts=8 delay=5",
        '  while [ "$attempt" -le "$max_attempts" ]; do',
        '    cp -f -- "$src" "$dst" 2>/dev/null',
        "    if \"$GFM_PY\" -c \"import sys, yaml; yaml.safe_load(open(sys.argv[1], encoding='utf-8'))\" \"$dst\" 2>/dev/null; then",
        "      printf -v \"$out_var\" '%s' \"$dst\"",
        "      return 0",
        "    fi",
        '    echo "  [config-stage retry $attempt/$max_attempts] $src did not copy/parse cleanly - retrying in ${delay}s..." >&2',
        '    rm -f -- "$dst"',
        '    sleep "$delay"',
        "    attempt=$((attempt + 1))",
        "  done",
        '  echo "  failed to stage a valid local copy of $src after $max_attempts attempts - giving up." >&2',
        "  return 1",
        "}",
    ]


def stage_tile_grid_lines() -> list[str]:
    """Bash fn staging the tile_grid gpkg to node-local disk, validated by byte size.

    GDAL/SQLite's advisory lock on a gpkg read directly from the shared data root can hang
    indefinitely on cluster compute nodes; a plain `cp` never takes that lock. The copy keeps
    the source's mtime: it is a Snakemake input, and a fresh mtime would mark every tile outdated.
    """
    return [
        "stage_tile_grid_locally() {",
        '  local src="$1" out_var="$2"',
        '  local stage_dir="${TMPDIR:-/tmp}/gfm_tile_grid"',
        '  mkdir -p "$stage_dir"',
        '  local dst="$stage_dir/tile_grid_${SLURM_JOB_ID:-$$}_$$.gpkg"',
        "  local attempt=1 max_attempts=8 delay=5",
        '  local src_size; src_size=$(stat -c%s -- "$src" 2>/dev/null)',
        '  while [ "$attempt" -le "$max_attempts" ]; do',
        '    cp -f --preserve=timestamps -- "$src" "$dst" 2>/dev/null',
        '    local dst_size; dst_size=$(stat -c%s -- "$dst" 2>/dev/null)',
        '    if [ -n "$dst_size" ] && [ "$dst_size" = "$src_size" ]; then',
        "      printf -v \"$out_var\" '%s' \"$dst\"",
        "      return 0",
        "    fi",
        '    echo "  [tile-grid-stage retry $attempt/$max_attempts] $src did not copy cleanly - retrying in ${delay}s..." >&2',
        '    rm -f -- "$dst"',
        '    sleep "$delay"',
        "    attempt=$((attempt + 1))",
        "  done",
        '  echo "  failed to stage a valid local copy of $src after $max_attempts attempts - giving up." >&2',
        "  return 1",
        "}",
    ]


_HELPERS = {"retry": retry_wrapper_lines, "stage_config": stage_configfile_lines, "stage_tile_grid": stage_tile_grid_lines}


def submit_script(path: str) -> None:
    """Run a written submit script with bash (on the cluster, where its Linux path exists)."""
    import subprocess

    subprocess.run(["bash", path], check=True)


def add_profile_set_argument(parser) -> None:
    """Add the repeatable `--profile-set PROFILE.FIELD=VALUE` option to an argparse parser."""
    parser.add_argument("--profile-set", action="append", default=[], metavar="PROFILE.FIELD=VALUE",
                        help=f"override a job resource profile (fields: {', '.join(PROFILE_FIELDS)}); repeatable")
