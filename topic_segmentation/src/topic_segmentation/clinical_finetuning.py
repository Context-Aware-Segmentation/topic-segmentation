"""Fine-tune models on MIMIC discharge notes at increasing training sizes.

    python -m topic_segmentation.clinical_finetuning --initialization news_pretrain --gpus 0 1

Read settings from configs/mimic_scaling.json. Train with the TS objective and
evaluate on the test notes. Each GPU runs one job at a time; completed runs are skipped.
"""
import argparse
import json
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path

from topic_segmentation import launch
from topic_segmentation.paths import REPOSITORY

CONFIG = json.loads((REPOSITORY / "configs/mimic_scaling.json").read_text())
OBJECTIVE = "ts_noda"  # folder name of the TS-only objective in the saved runs


def path(value):
    return (REPOSITORY / value).resolve()


def training_command(initialization, level, size, model=None, output_root=None):
    """Return the output directory and training command for one level and sample size."""
    train = f"train_{size:04d}.jsonl"
    splits = path(CONFIG["data"]["splits"])
    if not (splits / level / train).exists():
        splits = path(CONFIG["data"]["large_splits"])
    output = path(output_root or CONFIG["output"]) / initialization / level / f"n{size:04d}" / OBJECTIVE / f"seed{CONFIG['seed']}"
    command = [
        sys.executable, "-m", "topic_segmentation.training",
        "--model_name_or_path", path(model or CONFIG["initializations"][initialization]),
        "--dataset_name", "mimic",
        "--train_file", splits / level / train,
        "--validation_file", splits / level / "valid_1000.jsonl",
        "--test_file", path(CONFIG["data"]["test_inputs"]) / f"mimic_discharge_{level}.jsonl",
        "--dataset_cache_dir", REPOSITORY / "cache/mimic/finetune/reproductions" / level / f"n{size:04d}",
        "--test_data_name", f"mimic_discharge_{level}",
        "--seed", CONFIG["seed"],
        "--max_seq_length", CONFIG["max_seq_length"],
        "--learning_rate", CONFIG["learning_rate"],
        "--num_train_epochs", CONFIG["epochs"],
        "--per_device_train_batch_size", CONFIG["train_batch_size"],
        "--gradient_accumulation_steps", CONFIG["gradient_accumulation_steps"],
        "--per_device_eval_batch_size", CONFIG["eval_batch_size"],
        "--eval_strategy", "epoch",  # small training sets have too few steps for step-based evaluation
        "--save_strategy", "epoch",
        "--logging_strategy", "epoch",
        "--overwrite_output_dir", "True",
        "--do_da_ts", "False",
        "--do_tssp", "False",
        "--cl_loss_weight", "0.0",
        "--cl_temp", "0.1",
        "--tssp_loss_weight", "0.0",
        "--output_dir", output,
    ]
    return output, [str(part) for part in command + launch.TRAINING_DEFAULTS]


def finished(output):
    """Return whether all_results.json contains test scores."""
    results = output / "all_results.json"
    return results.exists() and "threshold_0.5_example_level_f1" in json.loads(results.read_text())


def run(jobs, gpus):
    """Run one job at a time per GPU and return the output paths of failed jobs."""
    pending, failures = queue.Queue(), []
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        while True:
            try:
                output, command = pending.get_nowait()
            except queue.Empty:
                return
            output.mkdir(parents=True, exist_ok=True)
            with open(output / "run.log", "w", encoding="utf-8") as log:
                code = subprocess.run(command, cwd=REPOSITORY, env=launch.environment(gpu),
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"[{'done' if code == 0 else f'failed ({code})'}] gpu {gpu}: {output}", flush=True)
            if code:
                failures.append(output)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--initialization", required=True, choices=list(CONFIG["initializations"]))
    parser.add_argument("--gpus", nargs="+", default=["0"], help="One run at a time on each GPU")
    parser.add_argument("--model", type=Path, help="Override the initialization checkpoint")
    parser.add_argument("--output-dir", type=Path, help="Parent directory of the initialization, level, and sample-size folders")
    parser.add_argument("--dry-run", action="store_true", help="Print the training commands without running them")
    args = parser.parse_args()
    jobs = [training_command(args.initialization, level, size, args.model, args.output_dir)
            for level in CONFIG["levels"] for size in CONFIG["sizes"]]
    if args.dry_run:
        for _, command in jobs:
            print(shlex.join(command))
        return
    missing = sorted({part for _, command in jobs for part in command if part.endswith(".jsonl") and not Path(part).exists()})
    if missing:
        sys.exit("missing inputs:\n" + "\n".join(missing))
    jobs = [(output, command) for output, command in jobs if not finished(output)]
    print(f"{len(jobs)} runs to do on GPUs {', '.join(args.gpus)}", flush=True)
    failures = run(jobs, args.gpus)
    if failures:
        sys.exit("failed runs:\n" + "\n".join(map(str, failures)))


if __name__ == "__main__":
    main()
