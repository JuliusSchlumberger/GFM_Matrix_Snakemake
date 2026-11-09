"""The HPC harness: sbatch argument parsing, core counts, budgeted release and dependency translation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hpc"))
import harness  # noqa: E402
from harness import Controller, enqueue, job_cores, load_tasks, parse_dependency, split_sbatch_args  # noqa: E402


def _script(tmp_path: Path, name: str, partition: str, cpus: int = 1) -> Path:
    p = tmp_path / name
    p.write_text(f"#!/bin/bash\n#SBATCH --partition={partition}\n#SBATCH --cpus-per-task={cpus}\n\necho hi\n")
    return p


def test_split_sbatch_args() -> None:
    opts, script, rest = split_sbatch_args(["--parsable", "--dependency=afterany:1:2", "-p", "4vcpu", "job.sbatch", "x"])
    assert opts == {"--parsable": True, "--dependency": "afterany:1:2", "--partition": "4vcpu"}
    assert (script, rest) == ("job.sbatch", ["x"])


def test_job_cores_counts_the_partition_node() -> None:
    assert job_cores({"--partition": "24vcpu", "--cpus-per-task": "1"}, {}) == 24
    assert job_cores({"--partition": "4vcpu", "--cpus-per-task": "4"}, {}) == 4
    assert job_cores({"--partition": "big", "--cpus-per-task": "8", "--ntasks": "2"}, {}) == 16
    assert job_cores({"--partition": "big"}, {"big": 32}) == 32


def test_parse_dependency() -> None:
    assert parse_dependency("afterok:1:2,afterany:3") == [("afterok", ["1", "2"]), ("afterany", ["3"])]


def _controller(tmp_path: Path, max_cores: int) -> Controller:
    plan = {"name": "t", "max_cores": max_cores, "state_dir": tmp_path / "state", "steps": [{"id": "s", "run": "true"}]}
    return Controller(plan, None, False, 0)


def test_release_respects_the_budget_and_translates_dependencies(tmp_path, monkeypatch) -> None:
    ctl = _controller(tmp_path, 10)
    q = ctl.queue_dir
    small = _script(tmp_path, "small.sbatch", "4vcpu", 4)
    a = enqueue(["--parsable", str(small)], q, "s", {})
    b = enqueue(["--parsable", str(small)], q, "s", {})
    c = enqueue(["--parsable", f"--dependency=afterany:{a}:{b}", str(small)], q, "s", {})
    d = enqueue(["--parsable", f"--dependency=afterok:{a}", str(small)], q, "s", {})
    submitted = []
    monkeypatch.setattr(harness, "submit", lambda t, dep: submitted.append((t.id, dep)) or str(100 + len(submitted)))

    ctl.release(load_tasks(q))
    assert [s[0] for s in submitted] == [a, b]  # 8 of 10 cores; c and d wait on them

    tasks = {t.id: t for t in load_tasks(q)}
    tasks[a].state, tasks[b].state = "FAILED", "COMPLETED"
    for t in (tasks[a], tasks[b]):
        harness.save_task(t, q)
    ctl.release(load_tasks(q))
    assert submitted[-1] == (c, None)  # afterany met by ended jobs: submitted with no dependency left
    assert {t.id: t.state for t in load_tasks(q)}[d] == "DEP_NEVER"  # afterok on a failed job


def test_too_big_job_is_refused(tmp_path, monkeypatch) -> None:
    ctl = _controller(tmp_path, 10)
    big = _script(tmp_path, "big.sbatch", "24vcpu")
    tid = enqueue([str(big)], ctl.queue_dir, "s", {})
    monkeypatch.setattr(harness, "submit", lambda t, dep: "1")
    ctl.release(load_tasks(ctl.queue_dir))
    assert {t.id: t.state for t in load_tasks(ctl.queue_dir)}[tid] == "TOO_BIG"


def test_step_is_done_once_its_command_and_jobs_end(tmp_path, monkeypatch) -> None:
    ctl = _controller(tmp_path, 10)
    ctl.states["s"] = {"status": "running", "job_task": "x"}
    small = _script(tmp_path, "small.sbatch", "1vcpu")
    tid = enqueue([str(small)], ctl.queue_dir, "s", {})
    tasks = load_tasks(ctl.queue_dir)
    tasks[0].id = "x"
    tasks[0].state = "RUNNING"
    ctl.update_steps(tasks)
    assert ctl.states["s"]["status"] == "running"
    tasks[0].state = "COMPLETED"
    ctl.update_steps(tasks)
    assert ctl.states["s"]["status"] == "done" and tid
