"""Stage 1: prepare the datasets of the segmentation experiments (all four by default, in this order).

    python scripts/prepare_data.py           # news, c4, mind, mimic
    python scripts/prepare_data.py news     # rebuild IDEA-seg from the saved pages into runs/news_construction/
    python scripts/prepare_data.py mind     # MIND sentences, teacher boundaries (silver labels), training splits
    python scripts/prepare_data.py c4       # paragraph-boundary inputs for boundary pre-training
    python scripts/prepare_data.py mimic    # MIMIC-IV evaluation sample, segments, inputs, fine-tuning splits

Add --dry-run to print the commands without running them.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
MIND = json.loads((REPOSITORY / "configs/mind_adaptation.json").read_text())
DATASETS = ["news", "c4", "mind", "mimic"]


def commands(dataset, gpu, teacher_model=None, output_dir=None):
    if dataset == "news":
        return [[REPOSITORY / "scripts/build_news.py"]]
    if dataset == "mind":
        module = ["-m", "topic_segmentation.data.build_mind_silver_labels"]
        options = ["--output-dir", output_dir] if output_dir else []
        sentences = output_dir / Path(MIND["sentence_file"]).name if output_dir else REPOSITORY / MIND["sentence_file"]
        predictions = output_dir / Path(MIND["teacher_predictions"]).name if output_dir else REPOSITORY / MIND["teacher_predictions"]
        teacher = ["-m", "topic_segmentation.inference", "--model", REPOSITORY / (teacher_model or MIND["teacher_checkpoint"]),
                   "--input", sentences, "--output", predictions,
                   "--window", "1024", "--stride", "512", "--batch-size", "32", "--no-protect-window-end",
                   "--device", f"cuda:{gpu}"]  # the news-trained teacher marks topic boundaries in MIND articles
        return [[*module, "sentences", *options], teacher, [*module, "subsets", *options], [*module, "training-data", *options]]
    if dataset == "c4":
        return [["-m", "topic_segmentation.data.build_c4_boundaries"]]
    return mimic_commands(REPOSITORY / "data/mimic")


def mimic_commands(root):
    """Evaluation sample, then the 5,000-note and 101,000-note fine-tuning pools with their nested splits."""
    module = "topic_segmentation.data.mimic."
    test = root / "eval_sets/random_seed_42"
    pool = root / "finetune_sets/discharge_bhc_seed42_n5000"
    large = root / "finetune_sets/discharge_bhc_seed42_n101000"
    caches = ["--cache", pool / "discharge_random5000_by_id.json"]
    large_caches = caches + ["--cache", large / "discharge_random96000_by_id.json"]
    return [
        # 1,000 test notes, their block and subheading segments, and the inference inputs
        ["-m", module + "sample", "--seed", "42", "--n", "1000", "--out-dir", test],
        ["-m", module + "segments", "--cache", test / "discharge_random1000_by_id.json", "--out", test / "mimic_discharge_segments.json"],
        ["-m", module + "inputs", "--segments", test / "mimic_discharge_segments.json", "--out-dir", test / "inference_inputs"],
        # 5,000-note pool without test notes or patients: labels, validation set, and 8 to 4,000 training notes
        ["-m", module + "sample", "--seed", "42", "--n", "5000", "--out-dir", pool,
         "--exclude-sample-json", test / "discharge_random1000.json"],
        ["-m", module + "segments", *caches, "--out", pool / "discharge_random5000_segments.json"],
        ["-m", module + "inputs", "--segments", pool / "discharge_random5000_segments.json", "--out-dir", pool / "segmentation_labels"],
        ["-m", module + "splits", pool, "--train-sizes", "50", "100", "300", "1000", "4000", "--prefix-sizes", "8", "16", "32"],
        # 96,000 more notes for 10,000 to 100,000 training notes, keeping the validation set and the 4,000 notes
        ["-m", module + "sample", "--seed", "43", "--n", "96000", "--out-dir", large,
         "--exclude-sample-json", test / "discharge_random1000.json", "--exclude-sample-json", pool / "discharge_random5000.json"],
        ["-m", module + "segments", *large_caches, "--out", large / "discharge_random101000_segments.json"],
        ["-m", module + "inputs", "--segments", large / "discharge_random101000_segments.json", "--out-dir", large / "segmentation_labels"],
        ["-m", module + "splits", large, "--extends", pool, "--train-sizes", "10000", "30000", "100000"],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="{news,c4,mind,mimic}", help="Datasets to prepare (default: all)")
    parser.add_argument("--gpu", default="0", help="GPU for the MIND teacher")
    parser.add_argument("--teacher-model", type=Path, help="News checkpoint used to label MIND")
    parser.add_argument("--output-dir", type=Path, help="Directory for MIND sentences, silver labels, subsets, and prepared splits")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if set(args.datasets) - set(DATASETS):  # argparse in Python 3.10 rejects an empty list with choices
        parser.error(f"choose datasets from {', '.join(DATASETS)}")
    output = (REPOSITORY / args.output_dir).resolve() if args.output_dir else None
    for command in [command for dataset in args.datasets or DATASETS
                    for command in commands(dataset, args.gpu, args.teacher_model, output)]:
        command = [sys.executable, *map(str, command)]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
