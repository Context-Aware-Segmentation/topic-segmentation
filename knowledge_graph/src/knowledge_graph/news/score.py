"""Score exact news entity spans, span-QID links, and normalized Wikipedia titles."""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from knowledge_graph.common import load_json, normalize_space, precision_recall_f1, write_json
from knowledge_graph.news.data import MIND_GOLD, lookup_qids
from knowledge_graph.news.policies import COMBINED_MENTIONS, LINK_ASSIGNMENTS, POLICIES, default_links


def normalize_title(value: str) -> str:
    value = unquote(value or "").replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", value)


def title_key(entity):
    """Cache key of a ReLiK entity: its Wikipedia title with whitespace collapsed."""
    return normalize_space(entity.get("wikipedia_title") or entity.get("relik_label"))


def stored_qid(entity):
    """The QID written into the entity (gold files and spaCy predictions)."""
    return entity.get("wikidataId") or entity.get("WikidataId")


def title_qids(cache):
    """ReLiK QIDs from resolving the predicted Wikipedia titles through the title-to-QID cache."""
    return lambda entity: stored_qid(entity) or cache.get(title_key(entity))


def candidate_qids(cache):
    """ReLiK QIDs stored with the retrieved candidate when the index provides them, else from the title cache."""
    return lambda entity: stored_qid(entity) or entity.get("relik_candidate_qid") or cache.get(title_key(entity))


def spans_and_links(payload, qid_of=stored_qid):
    """Return {doc: spans} and {doc: {span: QIDs}} from entity predictions."""
    spans, links = defaultdict(set), defaultdict(lambda: defaultdict(set))
    for doc in payload["documents"]:
        for entity in doc.get("gold", doc.get("entities", [])):
            span = (int(entity["begin"]), int(entity["end"]))
            spans[doc["id"]].add(span)
            if (qid := qid_of(entity)) and str(qid).startswith("Q"):
                links[doc["id"]][span].add(qid)
    return spans, links


def flatten_spans(spans):
    return {(doc, *span) for doc, doc_spans in spans.items() for span in doc_spans}


def flatten_links(links):
    return {(doc, *span, qid) for doc, doc_links in links.items() for span, qids in doc_links.items() for qid in qids}


def assign_links(selected, spacy_links, relik_links, links):
    """QIDs of the selected spans under one link assignment."""
    assign = LINK_ASSIGNMENTS[links]
    return {
        doc: {
            span: assign(spacy_links.get(doc, {}).get(span, set()), relik_links.get(doc, {}).get(span, set()))
            for span in spans
        }
        for doc, spans in selected.items()
    }


def score_policies(gold, spacy, relik):
    """NER of every policy and EL of its default links; the combined mentions get every link assignment."""
    gold_spans, gold_links = gold
    spacy_spans, spacy_links = spacy
    relik_spans, relik_links = relik
    ids = spacy_spans.keys() | relik_spans.keys()
    scores = {}
    for policy, select in POLICIES.items():
        selected = {doc: select(spacy_spans.get(doc, set()), relik_spans.get(doc, set())) for doc in ids}
        assignments = LINK_ASSIGNMENTS if policy == COMBINED_MENTIONS else [default_links(policy)]
        scores[policy] = {
            "ner": precision_recall_f1(gold_spans, flatten_spans(selected)),
            "el": {
                links: precision_recall_f1(
                    gold_links, flatten_links(assign_links(selected, spacy_links, relik_links, links))
                )
                for links in assignments
            },
        }
    return scores


def mind_gold(path):
    spans, links = spans_and_links(load_json(path))
    return flatten_spans(spans), flatten_links(links)


def dwie_gold(path):
    """Gold mention spans and span-QID links of the DWIE QID gold."""
    spans, links = set(), set()
    for doc in load_json(path)["documents"]:
        for entity in doc["entities"]:
            for mention in entity["mentions"]:
                span = (doc["id"], mention["begin"], mention["end"])
                spans.add(span)
                if entity["wikidata_qid"]:
                    links.add((*span, entity["wikidata_qid"]))
    return spans, links


def cached_title_qids(path, relik_paths, provided_qid=stored_qid):
    """QIDs of the predicted titles without a QID from the selected link source, from the cache or looked up online."""
    titles = {title_key(entity) for path_in in relik_paths for document in load_json(path_in)["documents"]
              for entity in document["entities"] if not provided_qid(entity)}
    return lookup_qids(titles, path)


def policies(args):
    """Every span policy on each MIND-50 and DWIE input."""
    runs = {"MIND-50": args.mind, "DWIE": args.dwie}
    cache = cached_title_qids(args.cache, [relik for inputs in runs.values() for _, _, relik in inputs])
    golds = {"MIND-50": mind_gold(args.mind_gold), "DWIE": dwie_gold(args.dwie_gold)}
    results = {}
    for dataset, inputs in runs.items():
        results[dataset] = {}
        for condition, spacy_path, relik_path in inputs:
            spacy = spans_and_links(load_json(spacy_path))
            relik = spans_and_links(load_json(relik_path), title_qids(cache))
            results[dataset][condition] = score_policies(golds[dataset], spacy, relik)
    write_json(args.out, results, indent=2)


def indexes(args):
    """ReLiK alone on MIND-50 with each retrieval index."""
    files = dict(args.relik)
    cache = cached_title_qids(args.cache, list(files.values()), provided_qid=candidate_qids({}))
    gold_spans, gold_links = mind_gold(args.mind_gold)
    results = {}
    for name, path in files.items():
        spans, links = spans_and_links(load_json(path), candidate_qids(cache))
        results[name] = {
            "ner": precision_recall_f1(gold_spans, flatten_spans(spans)),
            "el": precision_recall_f1(gold_links, flatten_links(links)),
        }
    write_json(args.out, results, indent=2)


def titles(args):
    """Exact span and Wikipedia title EL, excluding spans annotated only with a QID."""
    gold, qid_spans = set(), set()
    for doc in load_json(args.gold)["documents"]:
        concepts = {item["concept"]: item for item in doc["concepts"]}
        for mention in doc["mentions"]:
            if link := concepts[mention["concept"]]["link"]:
                span = (doc["id"], mention["begin"], mention["end"])
                if link.startswith("wikidata:"):
                    qid_spans.add(span)
                else:
                    gold.add((*span, normalize_title(link)))
    excluded = qid_spans - {item[:3] for item in gold}
    results = {}
    for condition, path in args.relik:
        predicted = {
            (doc["id"], int(e["begin"]), int(e["end"]), title)
            for doc in load_json(path)["documents"]
            for e in doc["entities"]
            if (doc["id"], int(e["begin"]), int(e["end"])) not in excluded
            if (title := normalize_title(e.get("wikipedia_title") or ""))
        }
        results[condition] = precision_recall_f1(gold, predicted)
    write_json(args.out, results, indent=2)


def add_qid_arguments(parser):
    parser.add_argument("--cache", type=Path, required=True, help="Title-to-QID cache of ReLiK predictions; missing titles are looked up online.")
    parser.add_argument("--mind-gold", type=Path, default=MIND_GOLD)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("policies")
    add_qid_arguments(p)
    p.add_argument("--mind", nargs=3, action="append", required=True, metavar=("INPUT", "SPACY", "RELIK"))
    p.add_argument("--dwie", nargs=3, action="append", required=True, metavar=("INPUT", "SPACY", "RELIK"))
    p.add_argument("--dwie-gold", type=Path, required=True, help="dwie_test_qid_gold.json")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=policies)
    p = sub.add_parser("indexes")
    add_qid_arguments(p)
    p.add_argument("--relik", nargs=2, action="append", required=True, metavar=("INDEX", "RELIK"))
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=indexes)
    p = sub.add_parser("titles")
    p.add_argument("--gold", type=Path, required=True, help="dwie_test_gold.json")
    p.add_argument("--relik", nargs=2, action="append", required=True, metavar=("INPUT", "RELIK"))
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=titles)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
