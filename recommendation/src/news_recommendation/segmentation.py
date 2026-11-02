"""Segment every article body of the MIND full-text catalog with the fixed news segmenter.

    PYTHONPATH=src python -m news_recommendation.segmentation --output runs/segmentation/segmented.jsonl

Run from recommendation/ in the topic-segmentation environment. Short bodies stay whole; missing bodies have no segments.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

from .paths import REPOSITORY


def sentences(body, splitter, min_words):
    """The units of a body and whether the model segments them: short bodies stay whole, missing ones are empty."""
    if not body.strip() or body.strip().lower() in ("nan", "none", "null", "n/a"):
        return [], False
    if len(body.split()) < min_words:
        return [body], False
    return [body[start:end] for start, end in splitter.span_tokenize(body)], True


def join_segments(units, labels):
    """Group sentences into segments, each ending at a boundary label of 1."""
    segments, current = [], []
    for unit, boundary in zip(units, labels):
        current.append(unit)
        if boundary:
            segments.append({"segment_id": len(segments) + 1, "text": " ".join(current)})
            current = []
    return segments


def main():
    from topic_segmentation.inference import ModernBertSegmenter

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Segmented catalog, relative to recommendation/")
    parser.add_argument("--config", default="configs/segmentation.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads((REPOSITORY / args.config).read_text())
    with (REPOSITORY / config["punkt"]).open("rb") as stream:
        splitter = pickle.load(stream)
    segmenter = ModernBertSegmenter(REPOSITORY / config["model"], args.device, config["window"], config["stride"],
                                    config["batch_size"], config["protect_window_end"])
    csv.field_size_limit(10_000_000)
    with (REPOSITORY / config["source"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = REPOSITORY / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as result:
        for offset in range(0, len(rows), 512):
            batch = rows[offset:offset + 512]
            units, modeled = zip(*(sentences(row["body"], splitter, config["min_words"]) for row in batch))
            segmented = [i for i in range(len(batch)) if modeled[i]]
            labels = dict(zip(segmented, segmenter.segment_batch([units[i] for i in segmented])))
            for i, row in enumerate(batch):
                record = {key: row[key] for key in ("news_id", "title", "abstract", "url")}
                record |= {"full_body": row["body"],
                           "segments": join_segments(units[i], labels.get(i, [1] * len(units[i])))}
                result.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
