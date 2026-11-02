"""Train PLM-NR on each article representation with torchrun, then score its best validation epoch on the test split.

    python scripts/run_training.py                                  # the three representations, one after another
    python scripts/run_training.py title_section_summary_512 --gpus 0 1 2 3

Runs go to <output-dir>/<representation>/ (default: runs/reproductions/<time>/).
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]


def main():
    inputs = json.loads((REPOSITORY / "configs/plm_nr.json").read_text())["inputs"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", default=inputs, metavar="INPUT", help=f"Any of {', '.join(inputs)}")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--data", help="Prepared-data directory, relative to recommendation/")
    parser.add_argument("--output-dir", default=f"runs/reproductions/{datetime.now():%Y%m%d-%H%M%S}")
    args = parser.parse_args()
    for variant in args.inputs:
        run = REPOSITORY / args.output_dir / variant
        run.mkdir(parents=True)
        data = ["--data", args.data] if args.data else []
        training = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                    f"--nproc_per_node={len(args.gpus)}", "-m", "news_recommendation.training",
                    "train", variant, *data, "--output-dir", str(run)]
        evaluation = [sys.executable, "-m", "news_recommendation.training",
                      "evaluate", "--output-dir", str(run)]
        for mode, command, gpus in (("train", training, args.gpus), ("evaluate", evaluation, args.gpus[:1])):
            with (run / f"{mode}.log").open("w") as log:
                subprocess.run(command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), OMP_NUM_THREADS="8"))


if __name__ == "__main__":
    main()
