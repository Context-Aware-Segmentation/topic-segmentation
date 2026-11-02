"""MedCAT model-pack inference with offset-preserving output, and unsupervised context adaptation on the 80 training notes.

python -m knowledge_graph.clinical.medcat [--evaluate-only]   # writes runs/clinical/medcat/
"""

import argparse
import random
from pathlib import Path

from knowledge_graph.clinical.data import CONDITIONS, CLINICAL_RUNS, documents, segmenter, segments, split
from knowledge_graph.clinical.scoring import score_ner
from knowledge_graph.common import EarlyStopping, write_json
from knowledge_graph.paths import REPOSITORY

PACK = REPOSITORY / "checkpoints" / "medcat" / "v1_Snomed2025_MIMIC_IV.zip"
CLINICAL_TYPE_NAMES = frozenset({
    "body structure", "cell", "cell structure", "clinical drug", "disorder", "finding", "medicinal product",
    "morphologic abnormality", "observable entity", "organism", "physical object", "procedure", "product",
    "product name", "regime/therapy", "specimen", "substance",
})


def load_pack(pack):
    from medcat.cat import CAT

    return CAT.load_model_pack(str(pack))


def clinical_entities(cat, segment):
    """MedCAT entities of one segment at note offsets, kept when a model-pack type is a clinical SNOMED category."""
    names = {str(type_id): str(name).casefold() for type_id, name in cat.cdb.addl_info["type_id2name"].items()}
    rows = []
    for raw in cat.get_entities(segment["text"])["entities"].values():
        types = [str(value) for value in raw.get("type_ids", [])]
        if not any(names.get(type_id, "") in CLINICAL_TYPE_NAMES for type_id in types):
            continue
        begin, end = int(raw["start"]), int(raw["end"])
        rows.append({
            "begin": segment["begin"] + begin, "end": segment["begin"] + end, "text": segment["text"][begin:end],
            "concept_id": str(raw["cui"]) if raw.get("cui") is not None else None,
            "concept_name": raw.get("pretty_name"), "semantic_type_ids": types,
            "confidence": float(raw["acc"]) if raw.get("acc") is not None else None,
            "detected_name": raw.get("detected_name"), "segment_id": segment["segment_id"],
        })
    return rows


def predict(cat, docs, nlp, condition):
    rows = {document["id"]: [row for segment in segments(nlp, document["text"], document["reference_segments"], condition)
                             for row in clinical_entities(cat, segment)]
            for document in docs}
    return score_ner(docs, rows), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=CLINICAL_RUNS / "medcat")
    parser.add_argument("--evaluate-only", action="store_true", help="Score the saved best CDB without adapting.")
    args = parser.parse_args()
    import numpy as np
    from medcat.cat import CAT
    from medcat.cdb import CDB

    random.seed(42)
    np.random.seed(42)
    folds, out = split(), args.out
    out.mkdir(parents=True, exist_ok=True)
    docs = documents()
    train = [d for d in docs if d["id"] in folds["train_ids"]]
    valid = [d for d in docs if d["id"] in folds["validation_ids"]]
    test = [d for d in docs if d["id"] in folds["test_ids"]]
    cat, nlp = load_pack(PACK), segmenter()
    if not args.evaluate_only:
        # Unsupervised context updates, selected on validation exact-span F1 with fixed256 inputs.
        selection = EarlyStopping()
        for _ in range(10):
            random.shuffle(train)
            for document in train:
                cat(document["text"], do_train=True)
            cat.config.linking.train = False
            metrics, _ = predict(cat, valid, nlp, "fixed256")
            if selection.update(metrics["f1"],
                                lambda: cat.cdb.save(str(out / "best_cdb.dat"))):
                break
    cat = CAT(CDB.load(str(out / "best_cdb.dat")), vocab=cat.vocab, config=cat.config, meta_cats=cat._meta_cats,
              rel_cats=cat._rel_cats, addl_ner=cat._addl_ner)
    results = {}
    for condition in CONDITIONS:
        metrics, rows = predict(cat, test, nlp, condition)
        write_json(out / f"test_{condition}.json", {"metrics": {"ner": metrics},
                                                    "documents": [{"id": n, "predictions": p} for n, p in rows.items()]})
        results[condition] = {"ner": metrics}
    write_json(out / "test_results.json", results)


if __name__ == "__main__":
    main()
