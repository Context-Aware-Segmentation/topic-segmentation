"""Stage 2: paragraph-boundary pre-training of ModernBERT on C4 (configs/boundary_pretraining.json).

    python scripts/run_boundary_pretraining.py --gpus 0 1
    python scripts/run_boundary_pretraining.py --smoke-test --gpus 0     # first 100,000 examples only
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--smoke-test", action="store_true", help="Train on the first 100,000 examples only")
    parser.add_argument("--output-dir", type=Path, help="Run folder; an existing one resumes from its last checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = parser.parse_args()
    run = args.output_dir or Path("runs/boundary_pretraining/reproductions") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = (REPOSITORY / run).resolve()
    command = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={len(args.gpus)}",
               "--module", "topic_segmentation.boundary_pretraining", "--output-dir", str(run)]
    if args.smoke_test:
        command.append("--smoke-test")
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    run.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REPOSITORY, env=dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(args.gpus)), check=True)


if __name__ == "__main__":
    main()
