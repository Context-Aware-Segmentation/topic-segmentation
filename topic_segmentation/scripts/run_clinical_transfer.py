"""Stage 6: transfer of the news segmenters to MIMIC-IV discharge notes (configs/mimic_inference.json, configs/mimic_scaling.json).

    python scripts/run_clinical_transfer.py zero-shot --models news_ft --levels block   # no clinical training
    python scripts/run_clinical_transfer.py scaling --initialization base              # clinical fine-tuning

zero-shot segments the 1,000 evaluation notes with each configured news model and baseline (configs/mimic_inference.json).
scaling fine-tunes each initialization (base, news_pretrain, news_ft) on growing numbers of annotated notes
(configs/mimic_scaling.json).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
INFERENCE = json.loads((REPOSITORY / "configs/mimic_inference.json").read_text())
SCALING = json.loads((REPOSITORY / "configs/mimic_scaling.json").read_text())


def path(value):
    return (REPOSITORY / value).resolve()


def zero_shot(models, levels, gpu, checkpoints=None, output=None):
    """Inference commands of every model on the discharge evaluation set at each target level."""
    commands = []
    checkpoints = checkpoints or {}
    for name in models:
        model = INFERENCE["models"][name]
        for level in levels:
            command = ["-m", "topic_segmentation.inference", "--model-type", model["model_type"], "--model", path(checkpoints.get(name, model["checkpoint"])),
                       "--input", path(INFERENCE["inputs"]["discharge"][level]),
                       "--output", path(output or INFERENCE["output"]) / f"{name}_discharge_{level}.jsonl", "--device", f"cuda:{gpu}",
                       "--window", INFERENCE["window"], "--stride", INFERENCE["stride"], "--batch-size", INFERENCE["batch_size"],
                       "--protect-window-end" if INFERENCE["protect_window_end"] else "--no-protect-window-end"]
            commands.append(command)
    return commands


def scaling(initializations, gpus, checkpoints=None, output=None):
    """Clinical fine-tuning commands, one per initialization; each schedules every level and training-set size."""
    commands = []
    checkpoints = checkpoints or {}
    for name in initializations:
        command = ["-m", "topic_segmentation.clinical_finetuning", "--initialization", name, "--gpus", *gpus]
        if name in checkpoints:
            command += ["--model", path(checkpoints[name])]
        if output:
            command += ["--output-dir", path(output)]
        commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["zero-shot", "scaling"])
    parser.add_argument("--models", nargs="+", choices=list(INFERENCE["models"]), default=list(INFERENCE["models"]))
    parser.add_argument("--levels", nargs="+", choices=["block", "subheading"], default=["block", "subheading"])
    parser.add_argument("--initialization", nargs="+", choices=list(SCALING["initializations"]),
                        default=list(SCALING["initializations"]), help="Starting checkpoints for clinical fine-tuning")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"], help="GPUs of the scaling runs, one run each; zero-shot uses the first")
    parser.add_argument("--news-pretrain", type=Path, help="MIND-adapted news checkpoint for zero-shot and scaling")
    parser.add_argument("--news-ft", type=Path, help="News-only checkpoint for zero-shot and scaling")
    parser.add_argument("--output-dir", type=Path, help="Clinical run directory, with zero_shot/ and scaling/ folders")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()
    checkpoints = {name: value for name, value in (("news_pretrain", args.news_pretrain), ("news_ft", args.news_ft)) if value}
    root = path(args.output_dir) if args.output_dir else None
    if args.step == "zero-shot":
        commands = zero_shot(args.models, args.levels, args.gpus[0], checkpoints, root / "zero_shot" if root else None)
    else:
        commands = scaling(args.initialization, args.gpus, checkpoints, root / "scaling" if root else None)
    for command in commands:
        command = [sys.executable, *map(str, command)]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
