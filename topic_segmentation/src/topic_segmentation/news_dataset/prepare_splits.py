"""Convert the editorial dataset into paragraph and sentence splits.

Remove images and headings, placing each heading's boundary on the preceding content.
For sentence inputs, place each paragraph's boundary on its final sentence.
Use the fixed 80/10/10 splits per outlet in configs/news_splits.json.
"""
import argparse
import json
import os

from nltk.tokenize import sent_tokenize

from .paths import REPOSITORY, ROOT

SPLITS = json.loads((REPOSITORY / "configs/news_splits.json").read_text())  # article IDs of train, val, and test
VARIANTS = (("news", False), ("news_sent", True))


def content_rows(rec):
    """Return content and heading rows, moving heading boundaries to the preceding content."""
    block_types = rec.get("block_types", ["element"] * len(rec["body"]))
    rows = [(text, label, heading)
            for text, label, heading, kind in zip(rec["body"], rec["labels"], rec["title_labels"], block_types)
            if kind != "image"]
    if not rows:
        return None
    texts, labels, headings = map(list, zip(*rows))
    for i, heading in enumerate(headings):
        if heading == 1:
            if i > 0 and headings[i - 1] == 0:
                labels[i - 1] = 1
            labels[i] = -100
    if headings[-1] == 0 and labels[-1] != 1:
        labels[-1] = 1
    return list(zip(texts, labels, headings))


def sentence_rows(rows):
    out = []
    for text, label, heading in rows:
        if heading == 1:
            out.append((text, -100, 1))
            continue
        parts = [s.strip() for s in sent_tokenize(text, language="english") if s.strip()]
        out.extend((part, label if j == len(parts) - 1 else 0, 0) for j, part in enumerate(parts))
    return out


def build_record(rec, sentences):
    rows = content_rows(rec)
    if rows and sentences:
        rows = sentence_rows(rows)
    if not rows:
        return None
    kept = [(text, 0 if label == -100 else label) for text, label, heading in rows if heading == 0]
    return {
        "articleId": rec["articleId"],
        "outlet": rec["outlet"],
        "sentences": [text for text, _ in kept],
        "labels": [label for _, label in kept],
    }


def write_jsonl(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/dataset.jsonl"))
    ap.add_argument("--out_root", default=str(ROOT / "prepared"))
    args = ap.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    for name, sentences in VARIANTS:
        records = {r["articleId"]: r for r in (build_record(rec, sentences) for rec in raw) if r is not None}
        assert records.keys() == {article for ids in SPLITS.values() for article in ids}
        for split, ids in SPLITS.items():
            write_jsonl(os.path.join(args.out_root, name, f"{split}.jsonl"), [
                {"articleId": article, "sentences": records[article]["sentences"], "labels": records[article]["labels"],
                 "title_labels": [0] * len(records[article]["sentences"]),
                 "section_topic_labels": [], "sentence_topic_labels": []}
                for article in ids])
        print(f"{name}: " + ", ".join(f"{split}={len(ids)}" for split, ids in SPLITS.items()))


if __name__ == "__main__":
    main()
