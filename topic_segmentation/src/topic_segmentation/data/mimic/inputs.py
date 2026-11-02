"""Build sentence-level MIMIC inputs for block and subheading segmentation.

    python -m topic_segmentation.data.mimic.inputs --segments SEGMENTS.json --out-dir DIR

Write one note per JSONL row to `mimic_<table>_<level>.jsonl`. Label 1 marks the
last sentence of a block, or of a prelude or heading section at subheading level.
"""
import argparse
import json
from pathlib import Path

LEVELS = ("block", "subheading")


def add_unit(sentences, labels, unit):
    """Append non-empty sentences and mark the last as a boundary."""
    unit = [sentence.strip() for sentence in unit if sentence.strip()]
    if unit:
        sentences.extend(unit)
        labels.extend([0] * (len(unit) - 1) + [1])


def flatten_note(note, level):
    sentences, labels = [], []
    for block in note["blocks"]:
        prelude = block["prelude"]["sentences"] if block["prelude"] else []
        sections = [section["sentences"] for section in block["sections"]]
        units = [prelude + [sentence for section in sections for sentence in section]] if level == "block" else [prelude, *sections]
        for unit in units:
            add_unit(sentences, labels, unit)
    if labels:
        labels[-1] = 1
    return sentences, labels


def rows(notes, level):
    output = []
    for note in notes:
        sentences, labels = flatten_note(note, level)
        if sentences:
            output.append({"articleId": f"{note['note_id']}::{level}", "note_id": note["note_id"],
                           "sentences": sentences, "labels": labels})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", type=Path, required=True, help="Segments file of data.mimic.segments")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.segments.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for table, note_ids in payload["order"].items():
        notes = [payload["notes"][note_id] for note_id in note_ids]
        for level in LEVELS:
            output = rows(notes, level)
            with (args.out_dir / f"mimic_{table}_{level}.jsonl").open("w", encoding="utf-8") as stream:
                stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in output)


if __name__ == "__main__":
    main()
