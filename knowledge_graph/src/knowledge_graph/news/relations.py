"""Predict and score DWIE relations with ATLOP or DREEAM, using reference or predicted entities."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from knowledge_graph.common import count_metrics, load_json, precision_recall_f1, write_json
from knowledge_graph.news.docre import IMPLEMENTATIONS, DocREPredictor
from knowledge_graph.news.inputs import (
    CONDITIONS,
    ENCODER,
    document_groups,
    observable_relations,
)

ENCODER_WINDOWS = "encoder_windows"  # test_document.json read through overlapping encoder windows


def score_key(row: dict) -> tuple:
    """Set key for scoring: unmapped relations never match a gold relation."""
    key = (row["document_id"], row["subject"], row["relation"], row["object"])
    return key if row.get("mapped", True) else (row["document_id"], "unmapped", *key[1:])


def to_gold(local: set[tuple], mappings: list[dict]) -> list[dict]:
    """Relations between gold concepts, once per document; a predicted-mention relation with an endpoint outside the
    gold mentions keeps its prediction ids and is marked unmapped."""
    by_title = {mapping["title"]: mapping for mapping in mappings}
    rows = {}
    for title, head, relation, tail in local:
        mapping = by_title[title]
        gold = mapping["local_to_gold"]
        mapped = gold[head] is not None and gold[tail] is not None
        ids = gold if mapped else mapping["local_to_prediction"]
        row = {"document_id": mapping["document_id"], "subject": ids[head], "relation": relation, "object": ids[tail]}
        if not mapped:
            row["mapped"] = False
        rows[score_key(row)] = row
    return sorted(rows.values(), key=lambda row: (not row.get("mapped", True), score_key(row)))


def evaluate(predictions: dict[str, list[dict]], documents: list[dict], groups: dict[str, set[str]]) -> dict:
    """Scores of every condition on each document group, and the document-level row when both of its parts exist."""
    gold = observable_relations(documents)
    results = {}
    for condition, rows in predictions.items():
        found = {score_key(row) for row in rows}
        results[condition] = {
            group: precision_recall_f1({r for r in gold if r[0] in ids}, {r for r in found if r[0] in ids})
            for group, ids in groups.items()
        }
    if ENCODER_WINDOWS in results and "chunks1024" in results:
        long, short = results[ENCODER_WINDOWS]["long"], results["chunks1024"]["short"]
        tp, predicted = long["tp"] + short["tp"], long["predicted"] + short["predicted"]
        results["document_level"] = {"all": count_metrics(tp, len(gold), predicted), "long": long, "short": short}
    return results


def write_scores(predictions_path, gold_path, out, tokenizer):
    documents = load_json(gold_path)["documents"]
    write_json(out, evaluate(load_json(predictions_path), documents, document_groups(documents, tokenizer)), indent=2)


def score(args: argparse.Namespace) -> None:
    write_scores(args.predictions, args.gold, args.out, AutoTokenizer.from_pretrained(ENCODER))


def predict(args: argparse.Namespace) -> None:
    """Write predictions and scores for the selected input conditions."""
    model = DocREPredictor(
        args.implementation, args.inputs / "rel2id.json", args.model, single_pass=args.single_pass_2048,
    )
    mappings = load_json(args.inputs / "mappings.json")
    predictions = {}
    for name in CONDITIONS:
        filename = f"test_{name}.json"
        if filename in mappings:  # predicted-mention inputs cover fewer conditions
            documents = load_json(args.inputs / filename)
            if name == "document" and args.single_pass_2048:
                local = model.predict(documents, max_length=2048)
            elif name == "document":
                local = model.predict_with_windows(documents)
                name = ENCODER_WINDOWS
            else:
                local = model.predict(documents)
            predictions[name] = to_gold(local, mappings[filename])
    path = args.out_dir / "predictions.json"
    write_json(path, predictions)
    write_scores(path, args.gold, args.out_dir / "results.json", model.tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="Run a trained model on DocRED inputs and score its relations.")
    p.add_argument("--implementation", choices=list(IMPLEMENTATIONS), required=True)
    p.add_argument("--inputs", type=Path, required=True, help="Gold-entity or predicted-mention DocRED inputs.")
    p.add_argument("--model", type=Path, required=True, help="best.pt of the implementation.")
    p.add_argument(
        "--single-pass-2048", action="store_true",
        help="Load a 2,048-subword checkpoint and encode each input in one pass; document inputs use a 2,048 cap.",
    )
    p.add_argument("--gold", type=Path, required=True, help="dwie_test_gold.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=predict)
    p = sub.add_parser("score", help="Score saved predictions.")
    p.add_argument("--gold", type=Path, required=True, help="dwie_test_gold.json")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
