"""JSONL records and parallel worker processes shared by the two summary generators."""
import json
import subprocess
import sys

from .paths import REPOSITORY


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write_jsonl(path, rows, mode="w"):
    destination = path.with_suffix(path.suffix + ".tmp") if mode == "w" else path
    with destination.open(mode) as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if mode == "w":
        destination.replace(path)


def run_workers(module, output, count):
    """Run `module worker` for every shard in parallel, each logging to worker-<i>.log."""
    processes = []
    for worker_id in range(count):
        with (output / f"worker-{worker_id}.log").open("w") as log:
            command = [sys.executable, "-m", module, "worker", "--output", str(output), "--worker", str(worker_id)]
            processes.append(subprocess.Popen(command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT))
    if any([process.wait() for process in processes]):
        raise RuntimeError(f"A summary worker failed; see the worker logs in {output}")
