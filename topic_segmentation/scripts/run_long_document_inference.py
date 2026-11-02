"""Stage 5: sliding-window inference over long news articles, and the off-the-shelf baselines (configs/news_inference.json).

    python scripts/run_long_document_inference.py --experiment strides       # a thesis experiment group
    python scripts/run_long_document_inference.py --model runs/mind_adaptation/mind_25k_full_news_ts \\
        --window 1024 --stride 256 512 768 1024 --no-protect                              # any setting
    python scripts/run_long_document_inference.py --baseline semantic

Window-end protection defers the last unit of a non-final window to a later window
when the window contains more than one unit.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPOSITORY / "configs/news_inference.json").read_text())
BASELINES = {"semantic": "chonkie_semantic", "neural": "chonkie_neural"}  # name -> model type


def segmenter(model, window, stride, protect, batch_size, output, gpu):
    """Inference command of one fine-tuned segmentation model."""
    return ["-m", "topic_segmentation.inference", "--model-type", "modernbert_ts", "--model", model, "--input", CONFIG["input"],
            "--output", output, "--window", window, "--stride", stride, "--batch-size", batch_size,
            "--protect-window-end" if protect else "--no-protect-window-end", "--device", f"cuda:{gpu}"]


def baseline(name, output, gpu):
    """Inference command of one Chonkie baseline."""
    return ["-m", "topic_segmentation.inference", "--model-type", BASELINES[name], "--model", REPOSITORY / CONFIG["baselines"][name],
            "--input", CONFIG["input"], "--output", output, "--window", CONFIG["window"], "--stride", "512", "--batch-size", "32",
            "--no-protect-window-end", "--device", f"cuda:{gpu}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=[*CONFIG["experiments"], "baselines"], help="Every setting of a thesis experiment")
    parser.add_argument("--model", help="Fine-tuned segmentation model")
    parser.add_argument("--baseline", choices=list(BASELINES), help="An off-the-shelf Chonkie segmenter")
    parser.add_argument("--window", type=int, default=CONFIG["window"], help="Tokens per window")
    parser.add_argument("--stride", type=int, nargs="+", default=[512], help="One or more strides")
    parser.add_argument("--protect", action=argparse.BooleanOptionalAction, default=True, help="Window-end protection")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--news-models-dir", type=Path, help="Directory containing reproduced news checkpoints")
    parser.add_argument("--mind-models-dir", type=Path, help="Directory containing reproduced MIND checkpoints")
    parser.add_argument("--output-dir", type=Path, help="Directory of the prediction files")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()
    if not (args.experiment or args.model or args.baseline):
        parser.error("give --experiment, --model, or --baseline")
    root = args.output_dir or Path("runs/long_document_inference/reproductions") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    root = (REPOSITORY / root).resolve()
    commands = []
    if args.experiment == "baselines" or args.baseline:
        for name in BASELINES if args.experiment else [args.baseline]:
            commands.append(baseline(name, root / f"chonkie_{name}.jsonl", args.gpu))
    elif args.experiment:
        for item in CONFIG["experiments"][args.experiment]:
            model = Path(item["model"])
            parents = (REPOSITORY / model).parents
            if args.news_models_dir and REPOSITORY / "runs/news_finetuning" in parents:
                model = REPOSITORY / args.news_models_dir / model.name
            elif args.mind_models_dir and REPOSITORY / "runs/mind_adaptation" in parents:
                model = REPOSITORY / args.mind_models_dir / model.name
            name = f"{Path(item['model']).name}/w{CONFIG['window']}_s{item['stride']}" + ("_protected" if item["protect"] else "")
            commands.append(segmenter(model, CONFIG["window"], item["stride"], item["protect"], item["batch_size"],
                                      root / f"{name}.jsonl", args.gpu))
    else:
        for stride in args.stride:
            name = f"{Path(args.model).name}/w{args.window}_s{stride}" + ("_protected" if args.protect else "")
            commands.append(segmenter(args.model, args.window, stride, args.protect, 32, root / f"{name}.jsonl", args.gpu))
    for command in commands:
        command = [sys.executable, *map(str, command)]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
