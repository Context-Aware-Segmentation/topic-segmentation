"""Sample MIMIC-IV discharge notes that contain a hospital course section.

    python -m topic_segmentation.data.mimic.sample --seed 42 --n 1000 --out-dir OUT [--exclude-sample-json EARLIER.json ...]

Reservoir sampling over discharge.csv.gz; notes and patients of earlier samples are excluded. Writes
`OUT/discharge_random<n>.json` (note and patient IDs) and `OUT/discharge_random<n>_by_id.json` (the notes).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from pathlib import Path

from topic_segmentation.paths import REPOSITORY

NOTES = REPOSITORY / "data/mimic/raw/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz"
COURSE_HEADING_RE = re.compile(r"^\s*(?:Brief\s+)?Hospital\s+Course\s*:", flags=re.IGNORECASE | re.MULTILINE)


def excluded_ids(paths):
    """Collect note and patient IDs from earlier samples."""
    notes, subjects = set(), set()
    for path in paths:
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]:
            notes.add(row["note_id"])
            subjects.add(row["subject_id"])
    return notes, subjects


def reservoir_sample(size, rng, excluded_notes, excluded_subjects):
    """Return a uniform sample sorted by note ID."""
    sample, eligible = [], 0
    with gzip.open(NOTES, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not COURSE_HEADING_RE.search(row["text"]):
                continue
            if row["note_id"] in excluded_notes or row["subject_id"] in excluded_subjects:
                continue
            eligible += 1
            slot = len(sample)
            if slot >= size:
                slot = rng.randrange(eligible)
                if slot >= size:
                    continue
            item = {"table": "discharge", "note_id": row["note_id"],
                    "subject_id": row["subject_id"], "text": row["text"]}
            if slot == len(sample):
                sample.append(item)
            else:
                sample[slot] = item
    sample.sort(key=lambda note: note["note_id"])
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude-sample-json", type=Path, action="append", default=[],
                        help="Earlier sample whose notes and patients are excluded; may be repeated")
    args = parser.parse_args()

    name = f"discharge_random{args.n}"
    notes = reservoir_sample(args.n, random.Random(args.seed), *excluded_ids(args.exclude_sample_json))
    if len(notes) != args.n:
        raise RuntimeError(f"Requested {args.n} notes but only {len(notes)} were eligible")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"rows": [{"note_id": note["note_id"], "subject_id": note["subject_id"]} for note in notes]}
    (args.out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cache = {"order": [note["note_id"] for note in notes], "notes": {note["note_id"]: note for note in notes}}
    (args.out_dir / f"{name}_by_id.json").write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
