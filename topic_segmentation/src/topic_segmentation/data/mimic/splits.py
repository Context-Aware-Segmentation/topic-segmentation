"""Create nested, patient-disjoint MIMIC training and validation sets.

    python -m topic_segmentation.data.mimic.splits POOL --train-sizes 50 100 300 1000 4000 --prefix-sizes 8 16 32
    python -m topic_segmentation.data.mimic.splits LARGE_POOL --extends POOL --train-sizes 10000 30000 100000

Read note caches and segmentation_labels/ from POOL; write to POOL/splits_valid1000_seed42.
Reserve 1,000 notes for validation and keep patients separate across train, validation,
and test. --prefix-sizes selects whole-patient prefixes of the smallest training set.
--extends retains the earlier pool's validation set and largest training subset.
"""
import argparse
import json
import random
from pathlib import Path

from topic_segmentation.paths import REPOSITORY

TEST = REPOSITORY / "data/mimic/eval_sets/random_seed_42/discharge_random1000_by_id.json"
SPLITS = "splits_valid1000_seed42"
LEVELS = ("block", "subheading")
VALID_SIZE = 1000
SEED = 42


def read_ids(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def note_ids(patients, notes_of):
    return {note_id for patient in patients for note_id in notes_of[patient]}


def take_patients(candidates, notes_of, size):
    """Select whole patients in order to reach size notes, skipping groups that would exceed it."""
    taken, rest, count = [], [], 0
    for patient in candidates:
        if count + len(notes_of[patient]) <= size:
            taken.append(patient)
            count += len(notes_of[patient])
        else:
            rest.append(patient)
    if count != size:
        raise ValueError(f"no patient-disjoint split of exactly {size} notes; reached {count}")
    return taken, rest


def whole_patients(ids, subject, notes_of):
    """Return patient IDs, checking that ids contains each patient's full set of notes."""
    patients = sorted({subject[note_id] for note_id in ids})
    if note_ids(patients, notes_of) != set(ids):
        raise ValueError("note ids do not cover whole patients")
    return patients


def patient_prefix(ids, size, subject):
    """Select the first size notes grouped by patient, requiring a whole-patient prefix."""
    by_patient = {}
    for note_id in ids:
        by_patient.setdefault(subject[note_id], []).append(note_id)
    chosen = []
    for notes in by_patient.values():  # patients in order of their first note
        if len(chosen) >= size:
            break
        chosen.extend(notes)
    if len(chosen) != size:
        raise ValueError(f"no patient-complete prefix of {size} notes")
    chosen = set(chosen)
    return [note_id for note_id in ids if note_id in chosen]


def write_split(directory, name, ids, rows):
    """Write `ids/<name>.txt` and `<level>/<name>.jsonl`."""
    (directory / "ids").mkdir(parents=True, exist_ok=True)
    (directory / "ids" / f"{name}.txt").write_text("".join(f"{note_id}\n" for note_id in ids), encoding="utf-8")
    for level, by_id in rows.items():
        (directory / level).mkdir(parents=True, exist_ok=True)
        with (directory / level / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(by_id[note_id], ensure_ascii=False) + "\n" for note_id in ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool", type=Path, help="Folder with the note cache and segmentation_labels/")
    parser.add_argument("--train-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--prefix-sizes", type=int, nargs="+", default=[])
    parser.add_argument("--extends", type=Path, help="Smaller pool whose validation set and largest training subset are kept")
    args = parser.parse_args()

    order, subject = [], {}
    for pool in ([args.extends] if args.extends else []) + [args.pool]:
        [cache_path] = pool.glob("*_by_id.json")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for note_id in cache["order"]:
            if note_id in subject:
                raise ValueError(f"note {note_id} is in two caches")
            order.append(note_id)
            subject[note_id] = str(cache["notes"][note_id]["subject_id"])
    notes_of = {}
    for note_id in order:
        notes_of.setdefault(subject[note_id], []).append(note_id)
    train_sizes = sorted(set(args.train_sizes))
    if VALID_SIZE + train_sizes[-1] != len(order):
        raise ValueError(f"{VALID_SIZE} validation and {train_sizes[-1]} training notes must make up the {len(order)} notes")
    test = json.loads(TEST.read_text(encoding="utf-8"))["notes"]
    if set(order) & set(test) or set(notes_of) & {str(note["subject_id"]) for note in test.values()}:
        raise AssertionError("the pool overlaps the test sample")

    if args.extends:
        kept = max((args.extends / SPLITS / "ids").glob("train_*.txt"), key=lambda path: int(path.stem.split("_")[1]))
        valid_patients = whole_patients(read_ids(args.extends / SPLITS / "ids/valid_1000.txt"), subject, notes_of)
        excluded = set(valid_patients)
        train_patients = [patient for patient in notes_of if patient not in excluded]
        nested = whole_patients(read_ids(kept), subject, notes_of)
    else:
        shuffled = list(notes_of)
        random.Random(SEED).shuffle(shuffled)
        valid_patients, train_patients = take_patients(shuffled, notes_of, VALID_SIZE)
        nested = []
    excluded = set(nested)
    available = [patient for patient in train_patients if patient not in excluded]
    random.Random(SEED + 1).shuffle(available)
    subsets, size = {}, len(note_ids(nested, notes_of))
    for train_size in train_sizes:
        added, available = take_patients(available, notes_of, train_size - size)
        nested += added
        selected = note_ids(nested, notes_of)
        subsets[train_size], size = [note_id for note_id in order if note_id in selected], train_size
    for prefix_size in args.prefix_sizes:
        subsets[prefix_size] = patient_prefix(subsets[train_sizes[0]], prefix_size, subject)
    subsets = dict(sorted(subsets.items()))
    sizes = list(subsets)
    if any(not set(subsets[small]) < set(subsets[large]) for small, large in zip(sizes, sizes[1:])):
        raise AssertionError("training subsets are not nested")
    selected = note_ids(valid_patients, notes_of)
    valid = [note_id for note_id in order if note_id in selected]

    rows = {}
    for level in LEVELS:
        with (args.pool / f"segmentation_labels/mimic_discharge_{level}.jsonl").open(encoding="utf-8") as stream:
            rows[level] = {row["note_id"]: row for row in map(json.loads, stream)}
        if set(rows[level]) != set(order):
            raise ValueError(f"{level} labels do not match the pool notes")
    out = args.pool / SPLITS
    write_split(out, "valid_1000", valid, rows)
    for train_size, ids in subsets.items():
        write_split(out, f"train_{train_size:04d}", ids, rows)


if __name__ == "__main__":
    main()
