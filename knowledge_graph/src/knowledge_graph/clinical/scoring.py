"""Scores of the clinical systems: exact-span NER, pooled relations, and the shared P/R/F1 counts."""

from knowledge_graph.common import precision_recall_f1


def metric(gold, predicted):
    """tp/fp/fn and P/R/F1: common.precision_recall_f1 without the set sizes."""
    return {k: v for k, v in precision_recall_f1(gold, predicted).items() if k not in ("gold", "predicted")}


def score_ner(documents, predictions):
    """Micro exact character-span NER against all reference mentions."""
    predicted = {(d["id"], int(row["begin"]), int(row["end"])) for d in documents for row in predictions.get(d["id"], [])}
    gold = {(d["id"], mention["begin"], mention["end"]) for d in documents for mention in d["gold_mentions"]}
    return metric(gold, predicted)


def reachable_relations(annotation, available_entities):
    """Reference relations whose endpoints coexist in at least one model input."""
    return {(r["head"], r["type"], r["tail"]) for r in annotation["relations"]
            if any(r["head"] in entities and r["tail"] in entities for entities in available_entities)}


def score_relations(documents):
    """Relation P/R/F1 pooled over notes."""
    found = {(d["note_id"], *relation) for d in documents for relation in d["predicted_relation_keys"]}
    gold = {(d["note_id"], *relation) for d in documents for relation in d["gold_relation_keys"]}
    return metric(gold, found)
