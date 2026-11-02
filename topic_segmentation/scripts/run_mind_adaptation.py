"""Stage 4: MIND silver-label adaptation, then fine-tuning on the news dataset (configs/mind_adaptation.json).

    python scripts/run_mind_adaptation.py --experiment objective_matrix          # a thesis experiment group
    python scripts/run_mind_adaptation.py --run mind_25k_full_news_ts  # one thesis run
    python scripts/run_mind_adaptation.py --size 10k --objective full --matched-steps --news-objective ts

A news run first trains the adaptation it starts from. To reuse adaptations and run only news fine-tuning,
--recorded-adaptation selects the saved thesis checkpoints; --from-adaptation DIR uses DIR/<adaptation run>.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from topic_segmentation import launch

REPOSITORY = Path(__file__).resolve().parents[1]

CONFIG = json.loads((REPOSITORY / "configs/mind_adaptation.json").read_text())


def command(run, model, output, prepared_root=None):
    """Training command of one adaptation run (on MIND silver labels) or news run (on the news dataset)."""
    adaptation = run["stage"] == "adaptation"
    profile = dict(CONFIG["adaptation" if adaptation else "news_finetuning"])
    profile["per_device_batch_size"] = run.get("batch_size", profile["per_device_batch_size"])  # same effective batch
    profile["gradient_accumulation_steps"] = run.get("accumulation", profile["gradient_accumulation_steps"])
    if adaptation:
        dataset, data = "mind_sent", REPOSITORY / (prepared_root or CONFIG["prepared_root"]) / run["size"]
    else:
        dataset, data = "news", REPOSITORY / "data/news"
    steps = CONFIG["adaptation"]["matched_steps"] if run.get("matched_steps") else None
    cache = REPOSITORY / "cache/mind/reproductions" / output.name
    objective = CONFIG["objectives"][run["objective"]]
    return launch.command(model, dataset, data, profile, objective, output, cache, run.get("length"), steps)


def starting_model(run, root, recorded, source, c4_model=None):
    """The checkpoint a run starts from: the boundary-pretrained encoder for an adaptation; for a news run, its
    adaptation from `source`, the saved one (`recorded`), or the one rerun under `root`."""
    if run["stage"] == "adaptation":
        return REPOSITORY / (c4_model or CONFIG["initialization"])
    adaptation = run["initialization"]
    if source:
        return source / adaptation
    if recorded:
        return REPOSITORY / CONFIG["runs"][adaptation]["record"]
    return root / adaptation


def jobs(runs, root, recorded, source=None, *, c4_model=None, prepared_root=None):
    """(name, command) in order: unless saved adaptations are used, each news run follows the adaptation it starts from."""
    order = {}
    for name, run in runs.items():
        if run["stage"] == "adaptation" and (recorded or source):
            continue
        if run["stage"] == "news" and not (recorded or source):
            order.setdefault(run["initialization"], runs.get(run["initialization"]) or CONFIG["runs"][run["initialization"]])
        order.setdefault(name, run)
    return [(name, command(run, starting_model(run, root, recorded, source, c4_model), root / name, prepared_root))
            for name, run in order.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", nargs="+", choices=list(CONFIG["experiments"]), help="Experiment groups; shared runs train once")
    parser.add_argument("--run", choices=list(CONFIG["runs"]), help="One thesis run")
    parser.add_argument("--size", choices=["2k", "5k", "10k", "25k"], default="25k", help="MIND articles for adaptation")
    parser.add_argument("--objective", choices=list(CONFIG["objectives"]), default="full", help="Adaptation objective")
    parser.add_argument("--matched-steps", action="store_true", help="Stop adaptation after the matched number of steps")
    parser.add_argument("--news-objective", choices=list(CONFIG["objectives"]), default="ts", help="News fine-tuning objective")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--recorded-adaptation", action="store_true", help="Start news runs from the saved adaptation")
    source.add_argument("--from-adaptation", type=Path, help="Directory containing the required adaptation checkpoints")
    parser.add_argument("--c4-model", type=Path, help="Boundary-pretrained checkpoint for MIND adaptation")
    parser.add_argument("--prepared-root", type=Path, help="Prepared MIND splits, with one folder per subset size")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, help="Parent directory of the run outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()
    if args.experiment:
        runs = {name: CONFIG["runs"][name] for experiment in args.experiment for name in CONFIG["experiments"][experiment]}
    elif args.run:
        runs = {args.run: CONFIG["runs"][args.run]}
    else:
        adapted = f"mind_{args.size}_{args.objective}" + ("_matched" if args.matched_steps else "")
        runs = {adapted: {"stage": "adaptation", "size": args.size, "objective": args.objective, "matched_steps": args.matched_steps},
                f"{adapted}_news_{args.news_objective}": {"stage": "news", "size": args.size, "objective": args.news_objective,
                                                          "initialization": adapted}}
    if (args.recorded_adaptation or args.from_adaptation) and not any(run["stage"] == "news" for run in runs.values()):
        parser.error("--recorded-adaptation and --from-adaptation require news fine-tuning runs")
    root = args.output_dir or Path("runs/mind_adaptation/reproductions") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    root = (REPOSITORY / root).resolve()
    source = REPOSITORY / args.from_adaptation if args.from_adaptation else None
    for name, arguments in jobs(runs, root, args.recorded_adaptation, source,
                                c4_model=args.c4_model, prepared_root=args.prepared_root):
        print(" ".join(map(str, arguments)), flush=True)
        if not args.dry_run:
            (root / name).mkdir(parents=True)
            subprocess.run(arguments, cwd=REPOSITORY, env=launch.environment(args.gpu), check=True)


if __name__ == "__main__":
    main()
