"""Prepare MIND sentences, silver labels, and training splits.

    python -m topic_segmentation.data.build_mind_silver_labels sentences
    python -m topic_segmentation.data.build_mind_silver_labels subsets
    python -m topic_segmentation.data.build_mind_silver_labels training-data

scripts/prepare_data.py mind runs the news teacher between sentences and subsets.
Settings come from configs/mind_adaptation.json. Subsets are prefixes of one shuffle;
the test set is the IDEA-seg news test set.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import re

from nltk.tokenize import sent_tokenize

from topic_segmentation.paths import REPOSITORY

csv.field_size_limit(10_000_000)

CONFIG = json.loads((REPOSITORY / "configs/mind_adaptation.json").read_text())
MIND_DIR = REPOSITORY / CONFIG["raw_dataset"]
SELECTION = CONFIG["selection"]  # Subset filters and split sizes
# Articles excluded by the original subset's prompt-length filter.
EXCLUDED_ARTICLES = {"N60020", "N62386", "N39350"}

# Body footers, most specific first; each is cut only when it starts in the second half.
FOOTER_PATTERNS = [
    re.compile(r'\s*-{2,3}\s+More AP\b.*$', re.IGNORECASE | re.DOTALL),
    re.compile(r'\s*Distributed by\b.*$', re.IGNORECASE | re.DOTALL),
    re.compile(r'\s+\S+@\S+\.\S{2,}\s*$'),
    re.compile(r'\s+Follow\s+\S.*?\bon Twitter\s+at\s*$', re.IGNORECASE | re.DOTALL),
    re.compile(r'\s+Follow\s+\S.*?\bon Twitter:?\s+@\S*\s*$', re.IGNORECASE | re.DOTALL),
    re.compile(r'\s+(?:©|\(c\))?\s*20\d{2}\s+\w[\w\s,\.]+(?:Visit|at)\s+.*$', re.IGNORECASE | re.DOTALL),
    re.compile(r'\s+\w+\s+on TripAdvisor\s*$', re.IGNORECASE),
]


def clean_body(text):
    text = text.strip()
    for pattern in FOOTER_PATTERNS:
        match = pattern.search(text)
        if match and match.start() > len(text) * 0.5:
            text = text[:match.start()]
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r' ([,\.!?;:\)])', r'\1', text)
    return text.strip()


def write_sentences(input_path, output_path):
    """Clean bodies, split sentences, and keep articles within the sentence limits."""
    with open(input_path, encoding="utf-8", newline="") as source, open(output_path, "w", encoding="utf-8") as out:
        for row in csv.DictReader(source):
            body = row.get("body", "")
            if not body:
                continue
            sentences = [s.strip() for s in sent_tokenize(clean_body(body)) if len(s.strip()) >= SELECTION["minimum_sentence_characters"]]
            if SELECTION["minimum_sentences"] <= len(sentences) <= SELECTION["maximum_sentences"]:
                out.write(json.dumps({"articleId": row["news_id"], "sentences": sentences}, ensure_ascii=False) + "\n")


def write_subsets(predictions_path, filtered_path, subsets_dir):
    """Keep articles within the segment limits, shuffle once, and write nested index files."""
    records = []
    with open(predictions_path) as fin, open(filtered_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            if (SELECTION["minimum_segments"] <= sum(d["predictions"]) <= SELECTION["maximum_segments"]
                    and d["articleId"] not in EXCLUDED_ARTICLES):
                fout.write(line)
                records.append({"articleId": d["articleId"]})
    random.seed(SELECTION["seed"])
    random.shuffle(records)
    for name, size in SELECTION["subset_sizes"].items():
        with open(subsets_dir / f"subset_{name}.json", "w") as f:
            json.dump(records[:size], f)


def write_training_data(filtered_path, subsets_dir, news_data, out_root):
    """Split each subset into train/val; the test split is the news test set."""
    with open(filtered_path) as f:
        predictions = {d["articleId"]: d for d in map(json.loads, f)}
    with open(news_data / "test.jsonl") as f:
        news_test = f.readlines()
    rng = random.Random(SELECTION["seed"])  # Share one RNG stream across subset sizes.
    for key in SELECTION["subset_sizes"]:
        with open(subsets_dir / f"subset_{key}.json") as f:
            articles = [predictions[e["articleId"]] for e in json.load(f)]
        rng.shuffle(articles)
        n_val = max(1, int(len(articles) * SELECTION["validation_ratio"]))
        out = out_root / key
        out.mkdir(parents=True, exist_ok=True)
        for name, part in (("train", articles[n_val:]), ("val", articles[:n_val])):
            with open(out / f"{name}.jsonl", "w") as f:
                for art in part:
                    f.write(json.dumps({"sentences": art["sentences"], "labels": art["predictions"]}) + "\n")
        with open(out / "test.jsonl", "w") as f:
            f.writelines(line if line.endswith("\n") else line + "\n" for line in news_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["sentences", "subsets", "training-data"])
    parser.add_argument("--output-dir", type=Path, help="Directory for this preparation run; prepared splits go in its prepared/ folder")
    args = parser.parse_args()
    paths = {key: REPOSITORY / CONFIG[key] for key in ("sentence_file", "teacher_predictions", "filtered_predictions", "prepared_root")}
    subsets = MIND_DIR
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {key: args.output_dir / value.name for key, value in paths.items()}
        paths["prepared_root"] = args.output_dir / "prepared"
        subsets = args.output_dir
    if args.step == "sentences":
        write_sentences(MIND_DIR / "news.csv", paths["sentence_file"])
    elif args.step == "subsets":
        write_subsets(paths["teacher_predictions"], paths["filtered_predictions"], subsets)
    else:
        write_training_data(paths["filtered_predictions"], subsets, REPOSITORY / "data/news", paths["prepared_root"])


if __name__ == "__main__":
    main()
