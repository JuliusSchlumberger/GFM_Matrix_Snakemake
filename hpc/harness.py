"""Run a plan of HPC steps under one core budget; every `sbatch` of a step is queued and released here.

A step runs its shell command on the controller host, or as a job (`job:`). While it runs, and in
every job it starts, `hpc/bin/sbatch` stands first on PATH: it queues the submission and answers
with a placeholder job id, so existing generators, submit scripts and self-submitting jobs work
unchanged. The controller submits a queued job once its dependencies have ended and its cores
fit the budget (partition `<N>vcpu` counts N, else cpus-per-task x ntasks), translating
placeholder ids. A step is done when its command and every job it queued have ended. The plan's
`preamble` (shell) runs before every step's command.

Usage (on the cluster login node, in tmux; GFM_REAL_SBATCH if sbatch is not /usr/bin/sbatch):
    python hpc/harness.py run --plan hpc/plans/hydrax_rerun.yml [--allow-destructive] [--only a,b]
    python hpc/harness.py status --plan hpc/plans/hydrax_rerun.yml
    python hpc/harness.py retry --plan ... <step>       # failed or interrupted step -> pending
    python hpc/harness.py mark-done --plan ... <step>   # accept a step without rerunning it
    python hpc/harness.py cancel --plan ...             # scancel every live job of the plan
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HOME = Path(__file__).resolve().parent
FAKE_ID_MIN = 10**11  # SLURM job ids are 32-bit, so anything this large is a placeholder
LIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED", "SUBMITTED"}
NO_VALUE_OPTIONS = {"--parsable", "--hold", "--exclusive", "--wait", "--requeue", "--no-requeue", "--test-only",
                    "--quiet", "-Q", "-H", "-W", "--overcommit", "-O", "--contiguous", "--spread-job",
                    "--use-min-nodes", "--ignore-pbs", "--no-kill", "-k"}
SHORT_VALUE_OPTIONS = {"-p": "--partition", "-c": "--cpus-per-task", "-n": "--ntasks", "-N": "--nodes",
                       "-d": "--dependency", "-J": "--job-name", "-t": "--time", "-o": "--output", "-e": "--error",
                       "-A": "--account", "-D": "--chdir", "-a": "--array"}


# ── sbatch arguments ──────────────────────────────────────────────────────────


def split_sbatch_args(argv: list[str]) -> tuple[dict[str, str | bool], str | None, list[str]]:
    """Split an sbatch command line into (options by long name, script, script args)."""
    opts: dict[str, str | bool] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("-"):
            return opts, a, argv[i + 1:]
        if a.startswith("--") and "=" in a:
            k, v = a.split("=", 1)
            opts[k] = v
        elif a in NO_VALUE_OPTIONS:
            opts[a] = True
        else:
            opts[SHORT_VALUE_OPTIONS.get(a, a)] = argv[i + 1]
            i += 1
        i += 1
    return opts, None, []


def script_options(script: Path) -> dict[str, str | bool]:
    """Return the #SBATCH options of a job script (command-line options override them)."""
    lines = []
    for line in script.read_text().splitlines():
        if line.startswith("#SBATCH"):
            lines += shlex.split(line[len("#SBATCH"):])
        elif line.strip() and not line.startswith("#"):
            break
    return split_sbatch_args(lines + ["script"])[0]


def job_cores(opts: dict, partition_cores: dict[str, int]) -> int:
    """Cores a job holds: its partition's node size when known, else cpus-per-task x ntasks."""
    cpus = int(opts.get("--cpus-per-task", 1)) * int(opts.get("--ntasks", 1))
    partition = str(opts.get("--partition", ""))
    if partition in partition_cores:
        return max(cpus, partition_cores[partition])
    m = re.fullmatch(r"(\d+)vcpu", partition)
    return max(cpus, int(m.group(1))) if m else cpus


def parse_dependency(spec: str) -> list[tuple[str, list[str]]]:
    """Parse a --dependency value into [(type, [job ids])]; singleton and ?-alternatives kept as types."""
    out = []
    for part in re.split(r"[,?]", spec):
        if not part:
            continue
        kind, *ids = part.split(":")
        out.append((kind, [i.split("+")[0] for i in ids]))
    return out


# ── queue (shared by the sbatch stand-in on any node and the controller) ──────


@dataclass
class Task:
    id: str
    step: str
    argv: list[str]
    cwd: str
    env: dict[str, str]
    cores: int
    queued_at: float
    state: str = "QUEUED"  # QUEUED, SUBMITTED, then a SLURM state; DEP_NEVER for an unmet afterok
    real_id: str | None = None
    submitted_at: float | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def path(self, queue_dir: Path) -> Path:
        return queue_dir / f"{self.id}.json"


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def save_task(task: Task, queue_dir: Path) -> None:
    _write_json(task.path(queue_dir), task.__dict__)


def load_tasks(queue_dir: Path) -> list[Task]:
    tasks = []
    for p in sorted(queue_dir.glob("*.json")):
        try:
            tasks.append(Task(**json.loads(p.read_text())))
        except (OSError, ValueError):
            continue  # a stand-in still writing it: picked up next poll
    return tasks


def enqueue(argv: list[str], queue_dir: Path, step: str, partition_cores: dict[str, int],
            env: dict[str, str] | None = None, cwd: str | None = None) -> str:
    """Queue one sbatch call with the caller's environment and directory; return its placeholder id."""
    cwd = cwd or os.getcwd()
    opts, script, _ = split_sbatch_args(argv)
    if script is None:
        raise SystemExit("hpc/bin/sbatch: no job script (stdin and --wrap are not supported)")
    script_path = Path(script) if Path(script).is_absolute() else Path(cwd) / script
    all_opts = {**script_options(script_path), **opts}
    queue_dir.mkdir(parents=True, exist_ok=True)
    while True:
        fake = str(random.randrange(FAKE_ID_MIN, 10 * FAKE_ID_MIN))
        if not (queue_dir / f"{fake}.json").exists():
            break
    task = Task(id=fake, step=step, argv=argv, cwd=cwd, env=dict(env if env is not None else os.environ),
                cores=job_cores(all_opts, partition_cores), queued_at=time.time())
    save_task(task, queue_dir)
    return fake


# ── SLURM ─────────────────────────────────────────────────────────────────────


def real_sbatch() -> str:
    """Return the real sbatch: GFM_REAL_SBATCH, else the first on PATH that is not the stand-in."""
    if os.environ.get("GFM_REAL_SBATCH"):
        return os.environ["GFM_REAL_SBATCH"]
    path = os.pathsep.join(d for d in os.environ.get("PATH", "").split(os.pathsep)
                           if Path(d).resolve() != (HOME / "bin").resolve())
    return shutil.which("sbatch", path=path) or "/usr/bin/sbatch"


def slurm_states(ids: list[str]) -> dict[str, str]:
    """Return {job id: state} from sacct (base job rows only)."""
    if not ids:
        return {}
    res = subprocess.run(["sacct", "-n", "-P", "-X", "-o", "JobID,State", "-j", ",".join(ids)],
                         capture_output=True, text=True)
    out = {}
    for line in res.stdout.splitlines():
        jid, _, state = line.partition("|")
        if jid in ids:
            out[jid] = state.split()[0] if state else "PENDING"
    return out


def dependency_ready(task: Task, by_id: dict[str, Task]) -> tuple[bool, str | None, bool]:
    """Return (ready, dependency string to pass on, never satisfiable)."""
    opts, _, _ = split_sbatch_args(task.argv)
    spec = opts.get("--dependency")
    if not spec:
        return True, None, False
    keep = []
    for kind, ids in parse_dependency(str(spec)):
        real_ids = []
        for jid in ids:
            dep = by_id.get(jid)
            if dep is None:  # a real job outside the harness: SLURM keeps the dependency
                real_ids.append(jid)
                continue
            if dep.state in ("QUEUED",) or dep.state in LIVE:
                return False, None, False
            failed = dep.state != "COMPLETED"
            if (kind == "afterok" and failed) or (kind == "afternotok" and not failed):
                return False, None, True
        if real_ids:
            keep.append(":".join([kind, *real_ids]))
    return True, ",".join(keep) or None, False


def submit(task: Task, dependency: str | None) -> str:
    """Submit a queued task to SLURM with its own environment and directory; return the job id."""
    opts, script, script_args = split_sbatch_args(task.argv)
    cmd = [real_sbatch(), "--parsable"]
    for k, v in opts.items():
        if k in ("--parsable", "--dependency"):
            continue
        cmd += [k] if v is True else [f"{k}={v}"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd += [script, *script_args]
    res = subprocess.run(cmd, cwd=task.cwd, env=task.env, capture_output=True, text=True, check=True)
    return res.stdout.strip().split(";")[0]


# ── plan and controller ───────────────────────────────────────────────────────


def load_plan(path: Path) -> dict:
    plan = yaml.safe_load(Path(path).read_text())
    plan["state_dir"] = Path(os.path.expandvars(plan["state_dir"]))
    ids = [s["id"] for s in plan["steps"]]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate step ids")
    for s in plan["steps"]:
        missing = [a for a in s.get("after", []) if a not in ids]
        if missing:
            raise SystemExit(f"{path}: step {s['id']} waits for unknown step(s) {missing}")
    return plan


def _state_path(plan: dict) -> Path:
    return plan["state_dir"] / "steps.json"


def load_step_states(plan: dict) -> dict[str, dict]:
    p = _state_path(plan)
    return json.loads(p.read_text()) if p.exists() else {}


def save_step_states(plan: dict, states: dict) -> None:
    _write_json(_state_path(plan), states)


def step_env(plan: dict, step_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({k: os.path.expandvars(str(v)) for k, v in (plan.get("env") or {}).items()})
    env.update({"PATH": f"{HOME / 'bin'}:{env.get('PATH', '')}", "GFM_HARNESS_QUEUE": str(plan["state_dir"] / "queue"),
                "GFM_HARNESS_STEP": step_id, "GFM_HARNESS_PYTHON": sys.executable,
                "GFM_HARNESS_PARTITION_CORES": json.dumps(plan.get("partition_cores") or {}),
                "GFM_REAL_SBATCH": real_sbatch()})
    return env


def job_script(plan: dict, step: dict) -> Path:
    """Write the sbatch wrapper of a `job:` step."""
    job = step["job"]
    logs = plan["state_dir"] / "logs"
    lines = ["#!/bin/bash", f"#SBATCH --job-name=h_{step['id']}", f"#SBATCH --output={logs}/{step['id']}_%j.out",
             f"#SBATCH --error={logs}/{step['id']}_%j.err"]
    lines += [f"#SBATCH --{k.replace('_', '-')}={v}" for k, v in job.items()]
    lines += ["set -eo pipefail", plan.get("preamble", ""), step["run"]]
    path = plan["state_dir"] / "steps" / f"{step['id']}.sbatch"
    path.write_text("\n".join(lines) + "\n")
    return path


class Controller:
    def __init__(self, plan: dict, only: set[str] | None, allow_destructive: bool, poll_s: float):
        self.plan, self.only, self.allow_destructive, self.poll_s = plan, only, allow_destructive, poll_s
        self.max_cores = int(plan["max_cores"]) - int(plan.get("reserve_cores", 0))
        self.queue_dir = plan["state_dir"] / "queue"
        for d in ("queue", "logs", "steps"):
            (plan["state_dir"] / d).mkdir(parents=True, exist_ok=True)
        self.states = load_step_states(plan)
        self.procs: dict[str, subprocess.Popen] = {}
        for s in plan["steps"]:
            st = self.states.setdefault(s["id"], {"status": "pending"})
            if st["status"] == "running" and s.get("job") is None:
                st["status"] = "interrupted"  # its launcher died with the controller: retry or mark-done

    def log(self, msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    def start_step(self, step: dict) -> None:
        sid = step["id"]
        env = step_env(self.plan, sid)
        cwd = os.path.expandvars(self.plan.get("cwd", "."))
        if step.get("job"):
            script = job_script(self.plan, step)
            fake = enqueue([str(script)], self.queue_dir, sid, self.plan.get("partition_cores") or {}, env, cwd)
            self.states[sid] = {"status": "running", "job_task": fake, "started": time.time()}
        else:
            log = open(self.plan["state_dir"] / "logs" / f"{sid}.log", "a")
            script = "\n".join(["set -eo pipefail", self.plan.get("preamble", ""), step["run"]])
            self.procs[sid] = subprocess.Popen(["bash", "-c", script], env=env, cwd=cwd,
                                               stdout=log, stderr=subprocess.STDOUT)
            self.states[sid] = {"status": "running", "started": time.time()}
        self.log(f"step {sid} started")

    def poll_tasks(self) -> list[Task]:
        tasks = load_tasks(self.queue_dir)
        live = [t for t in tasks if t.real_id and (t.state in LIVE)]
        states = slurm_states([t.real_id for t in live])
        for t in live:
            new = states.get(t.real_id, t.state)
            if new != t.state:
                t.state = new
                save_task(t, self.queue_dir)
        return tasks

    def release(self, tasks: list[Task]) -> None:
        by_id = {t.id: t for t in tasks}
        used = sum(t.cores for t in tasks if t.state in LIVE)
        order = {s["id"]: k for k, s in enumerate(self.plan["steps"])}
        waiting = sorted((t for t in tasks if t.state == "QUEUED"), key=lambda t: (order.get(t.step, 1e9), t.queued_at))
        patience = float(self.plan.get("backfill_patience_min", 30)) * 60
        for t in waiting:
            if t.cores > self.max_cores:
                t.state, t.error = "TOO_BIG", f"needs {t.cores} cores, budget {self.max_cores}"
                save_task(t, self.queue_dir)
                continue
            ready, dependency, never = dependency_ready(t, by_id)
            if never:
                t.state = "DEP_NEVER"
                save_task(t, self.queue_dir)
                continue
            if not ready:
                continue
            if used + t.cores > self.max_cores:
                if time.time() - t.queued_at > patience:
                    break  # stop backfilling so this one gets the next free cores
                continue
            try:
                t.real_id = submit(t, dependency)
                t.state, t.submitted_at = "SUBMITTED", time.time()
                used += t.cores
                self.log(f"submitted {t.step}: {Path(split_sbatch_args(t.argv)[1]).name} -> {t.real_id} "
                         f"({t.cores} cores, {used}/{self.max_cores} in use)")
            except subprocess.CalledProcessError as exc:
                t.state, t.error = "SUBMIT_FAILED", (exc.stderr or str(exc)).strip()
                self.log(f"submit failed {t.step} {t.argv}: {t.error}")
            save_task(t, self.queue_dir)

    def update_steps(self, tasks: list[Task]) -> None:
        by_step: dict[str, list[Task]] = {}
        for t in tasks:
            by_step.setdefault(t.step, []).append(t)
        for s in self.plan["steps"]:
            sid, st = s["id"], self.states[s["id"]]
            if st["status"] != "running":
                continue
            mine = by_step.get(sid, [])
            if sid in self.procs:
                rc = self.procs[sid].poll()
                if rc is None:
                    continue
                del self.procs[sid]
                st["command_rc"] = rc
            elif "job_task" in st:
                job = next((t for t in mine if t.id == st["job_task"]), None)
                if job is None or job.state == "QUEUED" or job.state in LIVE:
                    continue
                st["command_rc"] = 0 if job.state == "COMPLETED" else 1
            if any(t.state == "QUEUED" or t.state in LIVE for t in mine):
                continue
            failed = [t.id for t in mine if t.state != "COMPLETED"]
            st.update(status="failed" if st["command_rc"] else ("done_with_failed_jobs" if failed else "done"),
                      ended=time.time(), n_jobs=len(mine), n_failed_jobs=len(failed))
            self.log(f"step {sid} {st['status']} ({len(mine)} job(s), {len(failed)} not COMPLETED)")

    def start_ready_steps(self) -> int:
        """Start every pending step whose waits are met; return how many started.

        With --only, a wait on a step outside the selection counts as met unless it is pending.
        """
        started = 0
        for s in self.plan["steps"]:
            sid, st = s["id"], self.states[s["id"]]
            if st["status"] != "pending" or (self.only and sid not in self.only):
                continue
            if s.get("destructive") and not self.allow_destructive:
                continue
            after = [self.states[a]["status"] for a in s.get("after", [])
                     if not self.only or a in self.only or self.states[a]["status"] != "pending"]
            if all(a in ("done", "done_with_failed_jobs") for a in after):
                self.start_step(s)
                started += 1
        return started

    def run(self) -> None:
        while True:
            started = self.start_ready_steps()
            tasks = self.poll_tasks()
            self.release(tasks)
            self.update_steps(load_tasks(self.queue_dir))
            save_step_states(self.plan, self.states)
            if not started and not any(st["status"] == "running" for st in self.states.values()):
                self.log("nothing left to run (steps still pending wait on a failed, interrupted or "
                         "destructive step)")
                print_status(self.plan)
                return
            time.sleep(self.poll_s)


def print_status(plan: dict) -> None:
    states = load_step_states(plan)
    tasks = load_tasks(plan["state_dir"] / "queue")
    live = sum(t.cores for t in tasks if t.state in LIVE)
    print(f"{plan['name']}: {live}/{plan['max_cores']} cores in use, {sum(t.state == 'QUEUED' for t in tasks)} "
          f"job(s) queued")
    for s in plan["steps"]:
        st = states.get(s["id"], {"status": "pending"})
        mine = [t for t in tasks if t.step == s["id"]]
        counts: dict[str, int] = {}
        for t in mine:
            counts[t.state] = counts.get(t.state, 0) + 1
        flag = " [destructive]" if s.get("destructive") else ""
        print(f"  {s['id']:<28} {st['status']:<22} {', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}{flag}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "enqueue":  # the sbatch stand-in
        argv = sys.argv[3:] if sys.argv[2:3] == ["--"] else sys.argv[2:]
        fake = enqueue(argv, Path(os.environ["GFM_HARNESS_QUEUE"]), os.environ.get("GFM_HARNESS_STEP", "?"),
                       json.loads(os.environ.get("GFM_HARNESS_PARTITION_CORES") or "{}"))
        print(fake if "--parsable" in argv else f"Submitted batch job {fake}")
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status", "retry", "mark-done", "cancel"])
    parser.add_argument("step", nargs="?")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--only", default=None, help="run: comma-separated step ids (their own order and waits)")
    parser.add_argument("--allow-destructive", action="store_true", help="run: also run steps marked destructive")
    parser.add_argument("--poll-s", type=float, default=60.0)
    args = parser.parse_args()
    plan = load_plan(Path(args.plan))
    if args.command == "run":
        Controller(plan, set(args.only.split(",")) if args.only else None, args.allow_destructive, args.poll_s).run()
    elif args.command == "status":
        print_status(plan)
    elif args.command in ("retry", "mark-done"):
        states = load_step_states(plan)
        if args.step not in states:
            raise SystemExit(f"unknown or never-started step {args.step!r}")
        states[args.step] = {"status": "pending" if args.command == "retry" else "done"}
        save_step_states(plan, states)
        print(f"step {args.step}: {states[args.step]['status']}")
    else:
        ids = [t.real_id for t in load_tasks(plan["state_dir"] / "queue") if t.real_id and t.state in LIVE]
        if ids:
            subprocess.run(["scancel", *ids], check=False)
        print(f"cancelled {len(ids)} job(s)")


if __name__ == "__main__":
    main()
