"""Rebuild the three article representations, optionally regenerating segmentation and summaries, and train.

    python -m news_recommendation.pipeline replay --output runs/replay [--train]
    python -m news_recommendation.pipeline regenerate --output runs/regenerated [--train]

`replay` rebuilds the impression splits and packs the saved segmentation, summaries, and segment selection;
`regenerate` first segments the full-text catalog and generates both kinds of summary.
Each step runs in its own environment and logs to <output>/<step>.log.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .paths import REPOSITORY

ENVIRONMENTS = Path(os.environ.get("MT_UV_ENV_HOME") or Path.home() / ".local/share/uv/mt")


def steps(output, regenerate, train, segmentation_python, generation_python, summary_gpus, training_gpus):
    """(name, command) of every step."""
    data = output / "data"
    preparation = [sys.executable, "-m", "news_recommendation.preparation"]
    if not regenerate:
        commands = [("prepare", preparation + ["all", "--output", str(data)])]
    else:
        segmented = output / "segmented.jsonl"
        commands = [("segment", [segmentation_python, "-m", "news_recommendation.segmentation",
                                 "--output", str(segmented), "--device", f"cuda:{summary_gpus[0]}"]),
                    ("split", preparation + ["split", "--output", str(data), "--segmented", str(segmented)])]
        commands += [(name, [generation_python, "-m", f"news_recommendation.{module}", "run",
                             "--output", str(output / name), "--dataset", str(data), "--source", str(segmented),
                             "--gpus", *summary_gpus])
                     for name, module in (("segments", "segment_summaries"), ("articles", "article_summaries"))]
        commands.append(("inputs", preparation + [
            "inputs", "--output", str(data), "--segmented", str(segmented), "--new-selection",
            "--section-summaries", str(output / "segments/outputs.jsonl"), "--article-summaries", str(output / "articles")]))
    if train:
        commands.append(("train", [sys.executable, str(REPOSITORY / "scripts/run_training.py"), "--data", str(data),
                                   "--output-dir", str(output / "training"), "--gpus", *training_gpus]))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("replay", "regenerate"))
    parser.add_argument("--output", required=True, help="New directory, relative to recommendation/")
    parser.add_argument("--train", action="store_true", help="Also train and evaluate the three representations")
    parser.add_argument("--summary-gpus", nargs=2, default=["0", "1"], help="Two summary GPUs; segmentation uses the first")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"], help="GPUs for training")
    parser.add_argument("--segmentation-python", default=str(ENVIRONMENTS / "topic-segmentation/bin/python"))
    parser.add_argument("--generation-python", default=str(ENVIRONMENTS / "news-summary-generation/bin/python"))
    args = parser.parse_args()
    output = REPOSITORY / args.output
    output.mkdir(parents=True)
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY / "src"))
    for name, command in steps(output, args.mode == "regenerate", args.train, args.segmentation_python,
                               args.generation_python, args.summary_gpus, args.gpus):
        print(f"{name}: {shlex.join(command)}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=REPOSITORY, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
