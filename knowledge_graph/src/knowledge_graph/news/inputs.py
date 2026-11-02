"""Build DocRED inputs from DWIE reference entities or predicted singleton mentions."""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from knowledge_graph.common import load_json, write_json
from knowledge_graph.news.data import dwie_documents, load_partitions
from knowledge_graph.news.docred import tokenize_document
from knowledge_graph.news.policies import COMBINED_MENTIONS, POLICIES

TOKEN_RE = re.compile(r"\w+(?:[\-'’]\w+)*|[^\w\s]", re.UNICODE)
SEED, DEV_SIZE = 42, 70  # DWIE training documents held out for model selection
ENCODER = "roberta-base"  # tokenizer and encoder of ATLOP and DREEAM
MAX_LENGTH = 1024  # subwords per DocRED input
SPECIAL_TOKENS = 2  # <s> and </s> of every RoBERTa input
WINDOW = MAX_LENGTH - SPECIAL_TOKENS  # subwords of a semantic window


def tokenize(text: str, begin: int, end: int) -> tuple[list[str], list[tuple[int, int]]]:
    matches = list(TOKEN_RE.finditer(text, begin, end))
    return [m.group() for m in matches], [(m.start(), m.end()) for m in matches]


def sentence_of(mention: dict, sentences: list[dict]) -> int | None:
    for index, sentence in enumerate(sentences):
        if sentence["begin"] <= mention["begin"] < mention["end"] <= sentence["end"]:
            return index
    return None


def evidence_sentences(subject_mentions: list[dict], object_mentions: list[dict]) -> list[int]:
    """Shared sentences, otherwise the nearest pair; break ties by subject then object sentence ID."""
    subject_ids = {item["sent_id"] for item in subject_mentions}
    object_ids = {item["sent_id"] for item in object_mentions}
    if shared := sorted(subject_ids & object_ids):
        return shared
    if not subject_ids or not object_ids:
        return []
    left, right = min(
        ((left, right) for left in subject_ids for right in object_ids),
        key=lambda pair: (abs(pair[0] - pair[1]), pair[0], pair[1]),
    )
    return sorted({left, right})


def to_docred(doc: dict) -> tuple[dict, list]:
    """Tokenize a document, align its mentions, and map entities to document concepts."""
    sents, token_spans = [], []
    for sentence in doc["sentences"]:
        tokens, spans = tokenize(doc["text"], sentence["begin"], sentence["end"])
        sents.append(tokens)
        token_spans.append(spans)
    mentions: dict[int, list[dict]] = defaultdict(list)
    for mention in doc["mentions"]:
        sent_id = sentence_of(mention, doc["sentences"])
        if sent_id is None:
            continue
        overlapping = [
            i
            for i, (start, end) in enumerate(token_spans[sent_id])
            if start < mention["end"] and mention["begin"] < end
        ]
        if overlapping:
            mentions[mention["concept"]].append({"sent_id": sent_id, "pos": [min(overlapping), max(overlapping) + 1]})
    local_to_concept = sorted(mentions)
    local = {concept: index for index, concept in enumerate(local_to_concept)}
    vertex_set = [mentions[concept] for concept in local_to_concept]
    labels = []
    for relation in doc["relations"]:
        # ATLOP/DREEAM enumerate only ordered pairs of distinct entities.
        if relation["s"] in local and relation["o"] in local and relation["s"] != relation["o"]:
            head, tail = local[relation["s"]], local[relation["o"]]
            evidence = evidence_sentences(vertex_set[head], vertex_set[tail])
            labels.append({"r": relation["p"], "h": head, "t": tail, "evidence": evidence})
    return {"title": doc["id"], "sents": sents, "vertexSet": vertex_set, "labels": labels}, local_to_concept


def select_sentences(document: dict, sentence_ids, title: str) -> tuple[dict, list[int]]:
    """Select sentences from a DocRED document and return their surviving entity indices."""
    sentence_index = {old: new for new, old in enumerate(sentence_ids)}
    entities = [
        [
            {**mention, "sent_id": sentence_index[mention["sent_id"]]}
            for mention in entity if mention["sent_id"] in sentence_index
        ]
        for entity in document["vertexSet"]
    ]
    kept = [i for i, mentions in enumerate(entities) if mentions]
    local = {old: new for new, old in enumerate(kept)}
    labels = [
        {
            "r": label["r"], "h": local[label["h"]], "t": local[label["t"]],
            "evidence": evidence_sentences(entities[label["h"]], entities[label["t"]]),
        }
        for label in document["labels"] if label["h"] in local and label["t"] in local
    ]
    return {
        "title": title,
        "sents": [document["sents"][i] for i in sentence_ids],
        "vertexSet": [entities[i] for i in kept],
        "labels": labels,
    }, kept


def subword_lengths(doc: dict, tokenizer) -> list[int]:
    """Subwords per DocRED sentence, including the entity markers used by the models."""
    _, _, spans = tokenize_document(doc, tokenizer)
    return [end - start for start, end in spans]


def pack_sentences(sentence_ids: list[int], lengths: list[int], limit: int) -> list[list[int]]:
    """Consecutive sentences packed greedily up to ``limit`` subwords; a longer sentence stands alone."""
    groups, current, total = [], [], 0
    for sentence_id in sentence_ids:
        if current and total + lengths[sentence_id] > limit:
            groups.append(current)
            current, total = [], 0
        current.append(sentence_id)
        total += lengths[sentence_id]
    return groups + [current] if current else groups


def semantic_windows(units: list[list[int]], lengths: list[int]) -> list[list[int]]:
    """Pack consecutive segments into windows; each window repeats the previous window's last segment."""
    size = [sum(lengths[i] for i in unit) for unit in units]
    groups, start = [], 0
    while start < len(units):
        end, total = start, 0
        while end < len(units) and (end == start or total + size[end] <= WINDOW):
            total += size[end]
            end += 1
        groups.append([i for unit in units[start:end] for i in unit])
        if end >= len(units):
            break
        start = end - 1 if end - 1 > start else end
    return groups


# Each condition selects a partition and a content-token limit.
CONDITIONS = {
    "document": (None, None),
    "chunks256": (None, 256 - SPECIAL_TOKENS),
    "chunks512": (None, 512 - SPECIAL_TOKENS),
    "chunks1024": (None, 1024 - SPECIAL_TOKENS),
    "semantic512": ("semantic", 512 - SPECIAL_TOKENS),
    "equal512": ("equal", 512 - SPECIAL_TOKENS),
    "random512": ("random", 512 - SPECIAL_TOKENS),
    "semantic1024": ("semantic", 1024 - SPECIAL_TOKENS),
    "semantic_windows": ("semantic", WINDOW),
}
PREDICTED_CONDITIONS = ["document", "chunks256", "chunks512", "chunks1024", "semantic1024", "semantic_windows"]


def sentence_groups(condition: str, partitions: dict, lengths: list[int]) -> list:
    """Sentence groups for one DocRED input condition."""
    partition, limit = CONDITIONS[condition]
    if limit is None:
        return [range(len(lengths))]
    if partition is None:
        return pack_sentences(range(len(lengths)), lengths, limit)
    units = [
        group
        for segment in partitions[partition]
        for group in pack_sentences(range(segment["sentence_start"], segment["sentence_end"]), lengths, limit)
    ]
    return semantic_windows(units, lengths) if condition == "semantic_windows" else units


def condition_inputs(documents: dict, condition: str, partitions: dict, lengths: dict) -> list[tuple]:
    """(document id, DocRED input, document concepts of its entities) of one condition; inputs need two entities."""
    inputs = []
    for doc_id, (document, concepts) in documents.items():
        for index, sentence_ids in enumerate(sentence_groups(condition, partitions[doc_id], lengths[doc_id])):
            item, kept = select_sentences(document, sentence_ids, f"{doc_id}::{condition}::{index:03d}")
            if len(kept) >= 2:
                inputs.append((doc_id, item, [concepts[i] for i in kept]))
    return inputs


def document_groups(documents, tokenizer) -> dict[str, set[str]]:
    """All documents, those longer than 1,024 subwords, and the others."""
    ids = {doc["id"] for doc in documents}
    long_ids = {
        doc["id"]
        for doc in documents
        if len(tokenizer(doc["text"], add_special_tokens=True, truncation=False)["input_ids"]) > MAX_LENGTH
    }
    return {"all": ids, "long": long_ids, "short": ids - long_ids}


def write_conditions(out_dir, documents, conditions, partitions, tokenizer, gold_of=None):
    """test_<condition>.json for every condition, and mappings.json from input entities to gold concepts; ``gold_of``
    maps the predicted concepts of each document to gold concepts (gold-entity inputs need none)."""
    documents = {doc_id: to_docred(doc) for doc_id, doc in documents.items()}
    lengths = {doc_id: subword_lengths(item, tokenizer) for doc_id, (item, _) in documents.items()}
    mappings = {}
    for condition in conditions:
        inputs = condition_inputs(documents, condition, partitions, lengths)
        write_json(out_dir / f"test_{condition}.json", [item for _, item, _ in inputs])
        mappings[f"test_{condition}.json"] = [
            {"title": item["title"], "document_id": doc_id, "local_to_gold": concepts} if gold_of is None else
            {"title": item["title"], "document_id": doc_id, "local_to_prediction": concepts,
             "local_to_gold": [gold_of[doc_id][concept] for concept in concepts]}
            for doc_id, item, concepts in inputs]
    write_json(out_dir / "mappings.json", mappings)


def observable_relations(documents) -> set[tuple]:
    """Gold relations whose two distinct endpoints both have mentions."""
    relations = set()
    for doc in documents:
        mentioned = {mention["concept"] for mention in doc["mentions"]}
        relations.update(
            (doc["id"], r["s"], r["p"], r["o"])
            for r in doc["relations"]
            if r["s"] in mentioned and r["o"] in mentioned and r["s"] != r["o"]
        )
    return relations


def singleton_document(gold_doc: dict, spans: set[tuple]) -> dict:
    """The gold document with each predicted span inside a sentence as a singleton concept and no relations."""
    mentions = []
    for concept, (begin, end) in enumerate(sorted(spans)):
        mention = {"begin": begin, "end": end, "concept": concept}
        if 0 <= begin < end <= len(gold_doc["text"]) and sentence_of(mention, gold_doc["sentences"]) is not None:
            mentions.append(mention)
    return {**{key: gold_doc[key] for key in ("id", "text", "sentences")}, "mentions": mentions, "relations": []}


def gold_concept(gold_doc: dict, predicted_doc: dict) -> dict[int, int | None]:
    """Gold concept of each predicted mention with an exact, unambiguous span match (scoring only)."""
    by_span: dict[tuple[int, int], set[int]] = {}
    for mention in gold_doc["mentions"]:
        by_span.setdefault((mention["begin"], mention["end"]), set()).add(mention["concept"])
    output = {}
    for mention in predicted_doc["mentions"]:
        candidates = by_span.get((mention["begin"], mention["end"]), set())
        output[mention["concept"]] = next(iter(candidates)) if len(candidates) == 1 else None
    return output


def gold_entities(args: argparse.Namespace) -> None:
    """Train and dev inputs and one test file per input condition, with gold entity clusters and DREEAM evidence."""
    from transformers import AutoTokenizer

    documents = dwie_documents()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    train_ids = sorted(doc_id for doc_id, doc in documents.items() if "train" in doc["tags"])
    random.Random(SEED).shuffle(train_ids)
    for split, ids in (("dev", train_ids[:DEV_SIZE]), ("train", train_ids[DEV_SIZE:])):
        pairs = [to_docred(documents[doc_id]) for doc_id in ids]
        write_json(args.out_dir / f"{split}.json", [item for item, concepts in pairs if len(concepts) >= 2])
    test = {doc_id: documents[doc_id] for doc_id in sorted(documents) if "test" in documents[doc_id]["tags"]}
    write_conditions(args.out_dir, test, CONDITIONS, load_partitions(args.partitions), tokenizer)
    # Keep the one test-only label in the classifier vocabulary so evaluation counts it as unrecoverable.
    labels = sorted({relation["p"] for doc in documents.values() for relation in doc["relations"]})
    write_json(args.out_dir / "rel2id.json", {"Na": 0, **{label: index + 1 for index, label in enumerate(labels)}})


def spans_by_document(payload):
    """(begin, end) entity spans of every document."""
    return {doc["id"]: {(int(e["begin"]), int(e["end"])) for e in doc["entities"]} for doc in payload["documents"]}


def predicted_mentions(args: argparse.Namespace) -> None:
    """Build singleton-mention inputs and exact-span gold mappings for scoring."""
    from transformers import AutoTokenizer

    gold = {doc["id"]: doc for doc in load_json(args.gold)["documents"]}
    partitions = load_partitions(args.partitions)
    spacy_spans, relik_spans = spans_by_document(load_json(args.spacy)), spans_by_document(load_json(args.relik))
    combine = POLICIES[COMBINED_MENTIONS]
    documents = {
        doc_id: singleton_document(doc, combine(spacy_spans[doc_id], relik_spans[doc_id]))
        for doc_id, doc in gold.items()
    }
    gold_of = {doc_id: gold_concept(gold[doc_id], doc) for doc_id, doc in documents.items()}
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    write_conditions(args.out_dir, documents, PREDICTED_CONDITIONS, partitions, tokenizer, gold_of)
    write_json(args.out_dir / "rel2id.json", load_json(args.rel2id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gold-entities", help="Train, dev, and test inputs with gold entity clusters.")
    p.add_argument("--partitions", type=Path, required=True, help="dwie_test_partitions.json (news.data partitions)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=gold_entities)
    p = sub.add_parser("predicted-mentions", help="Test inputs with the combined mentions as singleton entities.")
    p.add_argument("--gold", type=Path, required=True, help="dwie_test_gold.json")
    p.add_argument("--partitions", type=Path, required=True, help="dwie_test_partitions.json (news.data partitions)")
    p.add_argument("--spacy", type=Path, required=True, help="spaCy predictions on whole DWIE documents.")
    p.add_argument("--relik", type=Path, required=True, help="ReLiK predictions on whole DWIE documents.")
    p.add_argument("--rel2id", type=Path, required=True, help="rel2id.json of the gold-entity inputs.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=predicted_mentions)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
