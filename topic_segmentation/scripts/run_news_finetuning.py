"""Stage 3: supervised fine-tuning on the news dataset (configs/news_finetuning.json).

    python scripts/run_news_finetuning.py --experiment backbones            # a thesis experiment group
    python scripts/run_news_finetuning.py --run modernbert_c4_1024_cssl     # one thesis run
    python scripts/run_news_finetuning.py --backbone roberta --length 512 --objective ts_da   # any setting

Objectives: ts (topic segmentation), da (with document augmentation), cssl and tssp (auxiliary objectives), full (all).
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from topic_segmentation import launch, metrics

REPOSITORY = Path(__file__).resolve().parents[1]

CONFIG = json.loads((REPOSITORY / "configs/news_finetuning.json").read_text())


def command(run, output, c4_model=None):
    """Training command of one run: backbone, window length, objective, and paragraph or sentence data."""
    model = REPOSITORY / CONFIG["backbones"][run["backbone"]]
    if run["backbone"] == "modernbert_c4" and c4_model:
        model = REPOSITORY / c4_model
    objective = dict(CONFIG["objectives"][run["objective"]], **run.get("overrides", {}))
    cache = REPOSITORY / "cache/news/reproductions" / output.name
    data = REPOSITORY / "data" / run["data"]
    return launch.command(model, run["data"], data, CONFIG["training"], objective, output, cache, max_length=run["length"])


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def record_paragraph_scores(output, length):
    """Score a sentence run at the paragraph ends of the news test set and append the scores to results.jsonl."""
    predictions = read_jsonl(output / f"predict_news_sent_max_seq{length}_ts_score_lt.txt")
    scores = metrics.score_paragraphs(predictions, read_jsonl(REPOSITORY / "data/news/test.jsonl"))
    metrics.record(output.parent / "results.jsonl", output.name, {"test": "news", "scored_at": "paragraph ends"}, scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", nargs="+", choices=list(CONFIG["experiments"]), help="Experiment groups; shared runs train once")
    parser.add_argument("--run", choices=list(CONFIG["runs"]), help="One thesis run")
    parser.add_argument("--backbone", choices=list(CONFIG["backbones"]), default="modernbert_c4")
    parser.add_argument("--length", type=int, choices=[512, 1024], default=1024, help="Tokens per training window")
    parser.add_argument("--objective", choices=list(CONFIG["objectives"]), default="ts")
    parser.add_argument("--data", choices=["news", "news_sent"], default="news", help="Paragraph or sentence units")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--c4-model", type=Path, help="Boundary-pretrained checkpoint for the modernbert_c4 backbone")
    parser.add_argument("--output-dir", type=Path, help="Parent directory of the run outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()
    if args.experiment:
        runs = {name: CONFIG["runs"][name] for experiment in args.experiment for name in CONFIG["experiments"][experiment]}
    elif args.run:
        runs = {args.run: CONFIG["runs"][args.run]}
    else:
        name = f"{args.backbone}_{args.length}_{args.objective}" + ("_sentences" if args.data == "news_sent" else "")
        runs = {name: {"backbone": args.backbone, "length": args.length, "data": args.data, "objective": args.objective}}
    root = args.output_dir or Path("runs/news_finetuning/reproductions") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    root = (REPOSITORY / root).resolve()
    for name, run in runs.items():
        arguments = command(run, root / name, args.c4_model)
        print(" ".join(map(str, arguments)), flush=True)
        if not args.dry_run:
            (root / name).mkdir(parents=True)
            subprocess.run(arguments, cwd=REPOSITORY, env=launch.environment(args.gpu), check=True)
            if run["data"] == "news_sent":
                record_paragraph_scores(root / name, run["length"])


if __name__ == "__main__":
    main()
