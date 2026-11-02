"""Filter MIND-large to articles with full bodies, split its impressions by day, and pack the three representations.

    python -m news_recommendation.preparation all --output runs/data

`split` filters articles and impressions, reserving the last two days for validation and test.
`inputs` packs each title with an abstract, segment summaries, or an article summary into 512 BERT tokens.
Segment summaries are dropped whole when they do not fit: by the saved selection when replaying,
otherwise at random with a per-article seed. See README.md for filtering criteria.
"""
import argparse
import csv
from datetime import datetime, timedelta
import json
import random

import numpy as np
from transformers import BertTokenizerFast

from .paths import REPOSITORY

MIN_BODY_TOKENS = 600
MAX_SEGMENTS = 10
MIN_HISTORY = 5
MIN_CANDIDATES = 5
SELECTION_SEED = 42
VARIANTS = ("title_abstract_512", "title_section_summary_512", "title_article_summary_512")


def read_jsonl(paths):
    return {row["news_id"]: row for path in paths for row in map(json.loads, path.open())}


def select_news(paths, tokenizer):
    """MIND-large train and dev news to keep, and the matched body of each."""
    news = {}
    for split in ("train", "dev"):
        with (paths["mind"] / split / "news.tsv").open() as stream:
            news |= {fields[0]: fields for fields in (line.rstrip("\r\n").split("\t") for line in stream)}
    by_url = {}
    for row in read_jsonl([paths["segmented"]]).values():
        by_url.setdefault(row["url"].strip(), []).append(row)
    candidates = {}
    for key, fields in news.items():
        matches = by_url.get(fields[5].strip(), []) if fields[5].strip() else []
        if len(matches) == 1 and 0 < len(matches[0]["segments"]) <= MAX_SEGMENTS and fields[4].strip():
            candidates[key] = matches[0]
    keys = sorted(candidates)
    lengths = [len(ids) for start in range(0, len(keys), 1000) for ids in tokenizer(
        [candidates[key]["full_body"].lower() for key in keys[start:start + 1000]], add_special_tokens=False)["input_ids"]]
    kept = {key: {"news_id": key, "body_source_id": candidates[key]["news_id"], "body_bert_tokens": length,
                  "body_sections": len(candidates[key]["segments"])}
            for key, length in zip(keys, lengths) if length >= MIN_BODY_TOKENS}
    return {key: news[key] for key in kept}, kept


def read_impressions(paths, news):
    """Train then dev impressions restricted to `news`, renumbered from 1."""
    records = []
    for split in ("train", "dev"):
        with (paths["mind"] / split / "behaviors.tsv").open("rb") as stream:
            for line in stream:
                _, user, timestamp, history, candidates = line.decode().rstrip("\r\n").split("\t")
                history = [key for key in history.split() if key in news]
                candidates = [item for item in candidates.split() if item.rsplit("-", 1)[0] in news]
                if (len(history) >= MIN_HISTORY and len(candidates) >= MIN_CANDIDATES
                        and {item[-1] for item in candidates} == {"0", "1"}):
                    records.append([user, timestamp, " ".join(history), " ".join(candidates)])
    return ["\t".join([str(i), *record]) + "\n" for i, record in enumerate(records, 1)]


def split_by_day(records):
    """Training, validation, and test records: the last day is test, the day before validation."""
    days = [datetime.strptime(record.split("\t")[2], "%m/%d/%Y %I:%M:%S %p").date() for record in records]
    validation_day = max(days) - timedelta(days=1)
    return {"train": [r for r, day in zip(records, days) if day < validation_day],
            "validation": [r for r, day in zip(records, days) if day == validation_day],
            "test": [r for r, day in zip(records, days) if day > validation_day]}


def prepare_split(destination, paths):
    news, metadata = select_news(paths, BertTokenizerFast.from_pretrained(paths["tokenizer"]))
    destination.mkdir(parents=True)
    (destination / "news.tsv").write_text("".join("\t".join(news[key]) + "\n" for key in sorted(news)))
    with (destination / "news_body_metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["news_id", "body_source_id", "body_bert_tokens", "body_sections"])
        writer.writeheader()
        writer.writerows(metadata[key] for key in sorted(metadata))
    for name, records in split_by_day(read_impressions(paths, news)).items():
        (destination / f"{name}.tsv").write_text("".join(records))


def pack_text_input(tokenizer, title, text, length=512):
    """[CLS] title [SEP] text [SEP], cutting the end of the text to fit."""
    ids = [tokenizer.cls_token_id] + tokenizer.encode(title.lower(), add_special_tokens=False)[:length - 3]
    ids.append(tokenizer.sep_token_id)
    return ids + tokenizer.encode(text.lower(), add_special_tokens=False)[:length - len(ids) - 1] + [tokenizer.sep_token_id]


def pack_section_input(tokenizer, title, summaries, news_id, length=512, selected_segment_ids=None):
    """[CLS] title [SEP] summary [SEP] ... with whole summaries dropped to fit; returns the ids and kept segment IDs."""
    ids = [tokenizer.cls_token_id] + tokenizer.encode(title.lower(), add_special_tokens=False)[:length - 2]
    ids.append(tokenizer.sep_token_id)
    sections = [(row["segment_id"], tokenizer.encode(row["summary"].lower(), add_special_tokens=False))
                for row in summaries]
    if selected_segment_ids is None:
        rng = random.Random(int.from_bytes(f"{SELECTION_SEED}:{news_id}".encode(), "big"))
        while len(ids) + sum(len(tokens) + 1 for _, tokens in sections) > length:
            sections.pop(rng.randrange(len(sections)))
    else:
        unknown = set(selected_segment_ids) - {sid for sid, _ in sections}
        if unknown:
            raise ValueError(f"Unknown saved segment IDs for {news_id}: {sorted(unknown)}")
        sections = [(sid, tokens) for sid, tokens in sections if sid in selected_segment_ids]
        if len(ids) + sum(len(tokens) + 1 for _, tokens in sections) > length:
            raise ValueError(f"Saved segment selection exceeds {length} tokens for {news_id}")
    for _, tokens in sections:
        ids += tokens + [tokenizer.sep_token_id]
    return ids, [sid for sid, _ in sections]


def prepare_inputs(destination, paths):
    """inputs/<variant>/features.npz for each representation, and the segment selection of the summary variant."""
    tokenizer = BertTokenizerFast.from_pretrained(paths["tokenizer"])
    with (destination / "news.tsv").open() as stream:
        news = {fields[0]: fields for fields in (line.rstrip("\n").split("\t") for line in stream)}
    with (destination / "news_body_metadata.csv").open(newline="") as stream:
        sources = {row["news_id"]: row["body_source_id"] for row in csv.DictReader(stream)}
    sections = read_jsonl([paths["sections"]])
    articles = read_jsonl(sorted(paths["articles"].glob("*/outputs.jsonl")))
    saved = ({row["news_id"]: row["selected_segment_ids"] for row in map(json.loads, paths["selection"].open())}
             if "selection" in paths else None)
    if saved is not None:
        for key in news:
            if not isinstance(saved.get(key), list):
                raise ValueError(f"Missing or invalid saved segment selection for {key}")
    keys = sorted(news)
    for variant in VARIANTS:
        folder = destination / "inputs" / variant
        folder.mkdir(parents=True)
        # Token ids, token types (all zero), and attention mask, side by side.
        packed = np.zeros((len(keys), 3 * 512), dtype=np.int32)
        selections = []
        for position, key in enumerate(keys):
            title, abstract, source = news[key][3], news[key][4], sources[key]
            if variant == "title_section_summary_512":
                ids, selected = pack_section_input(tokenizer, title, sections[source]["parsed"]["summaries"], key,
                                                   selected_segment_ids=saved[key] if saved is not None else None)
                selections.append({"news_id": key, "selected_segment_ids": selected})
            else:
                text = abstract if variant == "title_abstract_512" else articles[source]["summary"]
                ids = pack_text_input(tokenizer, title, text)
            packed[position, :512] = tokenizer.pad_token_id
            packed[position, :len(ids)] = ids
            packed[position, 1024:1024 + len(ids)] = 1
        np.savez(folder / "features.npz", keys=np.asarray(keys), packed=packed)
        if selections:
            (folder / "selection.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selections))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("all", "split", "inputs"))
    parser.add_argument("--output", required=True, help="Prepared-data directory, relative to recommendation/")
    parser.add_argument("--config", default="configs/data_preparation.json")
    parser.add_argument("--segmented", help="Segmented article catalog")
    parser.add_argument("--section-summaries", help="Segment summaries (outputs.jsonl)")
    parser.add_argument("--article-summaries", help="Article-summary run directory")
    parser.add_argument("--new-selection", action="store_true",
                        help="Drop segment summaries at random instead of replaying the saved selection")
    args = parser.parse_args()
    config = json.loads((REPOSITORY / args.config).read_text())
    config |= {key: value for key, value in (("segmented", args.segmented), ("sections", args.section_summaries),
                                             ("articles", args.article_summaries)) if value}
    if args.new_selection:
        del config["selection"]
    paths = {key: REPOSITORY / value for key, value in config.items()}
    destination = REPOSITORY / args.output
    if args.stage in ("all", "split"):
        prepare_split(destination, paths)
    if args.stage in ("all", "inputs"):
        prepare_inputs(destination, paths)


if __name__ == "__main__":
    main()
