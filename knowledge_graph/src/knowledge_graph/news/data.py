"""Prepare DWIE and MIND-50 documents, segmentation inputs, partitions, and Wikidata links."""

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from knowledge_graph.common import load_json, normalize_space, read_jsonl, write_json, write_jsonl
from knowledge_graph.paths import DATA, SEGMENTATION

DWIE_PARQUET = DATA / "dwie/train.parquet"  # the full DWIE mirror; documents tagged test form the test split
MIND_GOLD = DATA / "mind50/mind_nerl_gold_qid.json"
MIND_SENTENCES = SEGMENTATION / "data/mind/full/news_seg_predictions.jsonl"
SEED = 42  # random partitions
TITLE_QIDS = DATA / "dwie/dwie_title_qid_cache.json"
PARTITIONS = ("document", "semantic", "equal", "random")  # semantic: predicted topic segments


def lookup_qids(titles, cache_path):
    """Wikidata QIDs of Wikipedia titles through the MediaWiki API (redirects followed), cached on disk."""
    cache_path = Path(cache_path)
    cache = load_json(cache_path) if cache_path.exists() else {}
    titles = {title for value in titles if (title := normalize_space(value))}
    missing = sorted(titles - cache.keys())
    for start in range(0, len(missing), 45):
        batch = missing[start : start + 45]
        params = {"action": "query", "prop": "pageprops", "titles": "|".join(batch), "redirects": "1", "format": "json"}
        request = urllib.request.Request(
            "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params),
            headers={"User-Agent": "knowledge-graph/1.0 (research)"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            query = json.load(response)["query"]
        aliases = {title: title for title in batch}
        for row in query.get("normalized", []) or []:
            if row.get("from") in aliases:
                aliases[row["from"]] = row.get("to", aliases[row["from"]])
        for row in query.get("redirects", []) or []:
            for original, current in list(aliases.items()):
                if current == row.get("from"):
                    aliases[original] = row.get("to")
        page_qids = {
            page.get("title"): qid
            for page in query["pages"].values()
            if (qid := (page.get("pageprops") or {}).get("wikibase_item"))
        }
        for original in batch:
            cache[original] = page_qids.get(aliases[original])
        write_json(cache_path, cache)
        time.sleep(0.1)
    return cache


def parse_tags(value):
    """DWIE tags, stored either as a list or as its string form."""
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if text.startswith("["):
        return [str(item) for item in json.loads(text.replace("'", '"'))]
    return [text] if text else []


def sentences(nlp, text):
    """Sentence spans without surrounding whitespace."""
    spans = []
    for sent in nlp(text).sents:
        begin, end = sent.start_char, sent.end_char
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if begin < end:
            spans.append({"begin": begin, "end": end, "text": text[begin:end]})
    return spans


def parse_row(row, nlp):
    """One DWIE parquet row as text, tags, sentences, mentions, concepts, and relations."""
    text = row.get("content") or ""
    return {
        "id": row["id"],
        "text": text,
        "tags": parse_tags(row.get("tags")),
        "sentences": sentences(nlp, text),
        "mentions": [
            {"begin": m["begin"], "end": m["end"], "text": m.get("text") or "", "concept": m["concept"]}
            for m in row.get("mentions") or []
        ],
        "concepts": [
            {
                "concept": c["concept"],
                "text": c.get("text") or "",
                "link": c.get("link") or "",
                "tags": c.get("tags") or [],
                "count": c.get("count") or 0,
            }
            for c in row.get("concepts") or []
        ],
        "relations": row.get("relations") or [],
    }


def dwie_documents():
    """Every document of the DWIE parquet mirror by id, split into sentences by the rule-based sentencizer."""
    import pyarrow.parquet as pq
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?"]})
    return {row["id"]: parse_row(row, nlp) for row in pq.read_table(DWIE_PARQUET).to_pylist()}


def qid_gold(gold, title_qids):
    """DWIE entity concepts with their mentions and the QID of their original Wikipedia link (QID literals kept)."""
    documents = []
    for source in gold["documents"]:
        mentions = defaultdict(list)
        for m in source["mentions"]:
            mentions[m["concept"]].append({"begin": m["begin"], "end": m["end"], "text": m["text"]})
        entities = [
            {
                "concept_id": concept["concept"],
                "wikipedia_title": concept["link"] or None,
                "wikidata_qid": title_qids.get(normalize_space(concept["link"])) if concept["link"] else None,
                "mentions": mentions[concept["concept"]],
            }
            for concept in sorted(source["concepts"], key=lambda c: c["concept"])
            if "type::entity" in concept["tags"]
        ]
        documents.append({"id": source["id"], "entities": entities})
    return {"documents": documents}


def dwie(args: argparse.Namespace) -> None:
    """The DWIE test documents with sentences, mentions, concepts, and relations, and their segmentation input."""
    documents = [doc for doc in dwie_documents().values() if "test" in doc["tags"]]
    write_json(args.out_dir / "dwie_test_gold.json", {"split": "test", "documents": documents})
    write_jsonl(
        args.out_dir / "dwie_test_segmentation_input.jsonl",
        [{"articleId": doc["id"], "sentences": [sent["text"] for sent in doc["sentences"]]} for doc in documents],
    )


def qids(args: argparse.Namespace) -> None:
    """QID gold of the DWIE test entities; titles missing from the cache are looked up online."""
    gold = load_json(args.gold)
    titles = {concept["link"] for doc in gold["documents"] for concept in doc["concepts"] if concept["link"]}
    literal = re.compile(r"wikidata:Q\d+")
    direct = {title: title.removeprefix("wikidata:").upper() for title in titles if literal.fullmatch(title)}
    write_json(args.out, qid_gold(gold, lookup_qids(titles - direct.keys(), TITLE_QIDS) | direct))


def segment_ends(boundaries: list) -> list[int]:
    """Sentence index after each predicted segment; the last sentence closes the last segment."""
    n = len(boundaries)
    return [i + 1 for i, boundary in enumerate(boundaries) if boundary and i + 1 < n] + [n] if n else []


def load_partitions(path) -> dict[str, dict[str, list[dict]]]:
    """Segments of every partition by document id (written by partitions)."""
    return {row["id"]: row["partitions"] for row in load_json(path)["documents"]}


def whole_documents(documents):
    return [(doc, [{"text": doc["text"], "begin": 0, "end": len(doc["text"]), "segment_id": 0}]) for doc in documents]


def locate_segments(doc, prediction):
    """Segments ending at the predicted boundaries, found in the text by sequential sentence search."""
    text, cursor, spans = doc["text"], 0, []
    for sentence in prediction["sentences"]:
        start = text.find(sentence, cursor)
        cursor = start + len(sentence)
        spans.append((start, cursor))
    segments, first = [], 0
    for end in segment_ends(prediction["predictions"]):
        begin, stop = spans[first][0], spans[end - 1][1]
        segments.append({"text": text[begin:stop], "begin": begin, "end": stop, "segment_id": len(segments)})
        first = end
    return segments


def predicted_segments(documents, predictions):
    """Predicted topic segments of every document from the segmentation output."""
    by_id = {item["articleId"]: item for item in read_jsonl(predictions)}
    return [(doc, locate_segments(doc, by_id[doc["id"]])) for doc in documents]


def partition_segments(documents, partitions, partition):
    """One DWIE partition (document, semantic, equal, or random) of every document."""
    by_id, fields = {doc["id"]: doc for doc in documents}, ("text", "begin", "end", "segment_id")
    partitions = load_partitions(partitions)
    return [
        (by_id[doc_id], [{key: unit[key] for key in fields} for unit in segments[partition]])
        for doc_id, segments in partitions.items()
        if doc_id in by_id
    ]


def partitions(args: argparse.Namespace) -> None:
    """Semantic partitions at the predicted boundaries; equal-length and random partitions with as many segments."""
    gold = load_json(args.gold)
    predictions = {item["articleId"]: item for item in read_jsonl(args.segments)}
    rng = random.Random(SEED)
    rows = []
    for document in gold["documents"]:
        prediction, sentences = predictions[document["id"]], document["sentences"]
        n = len(sentences)
        ends = segment_ends(prediction["predictions"])
        k = len(ends)
        boundaries = {
            "document": [n],
            "semantic": ends,
            "equal": [round(i * n / k) for i in range(1, k)] + [n],
            "random": sorted(rng.sample(range(1, n), k - 1)) + [n],
        }
        segments = {}
        for name, partition_ends in boundaries.items():
            starts = [0, *partition_ends[:-1]]
            segments[name] = [
                {
                    "segment_id": index,
                    "sentence_start": start,
                    "sentence_end": end,
                    "begin": sentences[start]["begin"],
                    "end": sentences[end - 1]["end"],
                    "text": document["text"][sentences[start]["begin"] : sentences[end - 1]["end"]],
                }
                for index, (start, end) in enumerate(zip(starts, partition_ends))
            ]
        rows.append({"id": document["id"], "partitions": segments})
    write_json(args.out, {"documents": rows}, indent=2)


def mind_input(args: argparse.Namespace) -> None:
    """Write MIND-50 segmentation inputs in the source file's order."""
    ids = {doc["id"] for doc in load_json(MIND_GOLD)["documents"]}
    rows = []
    with MIND_SENTENCES.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["articleId"] in ids:
                rows.append({"articleId": item["articleId"], "sentences": item["sentences"]})
    write_jsonl(args.out, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dwie", help="Convert the DWIE parquet mirror into gold and segmentation input files.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=dwie)
    p = sub.add_parser("qids", help="Resolve the gold entities' Wikipedia titles to QIDs (cached).")
    p.add_argument("--gold", type=Path, required=True, help="dwie_test_gold.json")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=qids)
    p = sub.add_parser("partitions", help="Build semantic, equal, and random partitions from segmentation output.")
    p.add_argument("--gold", type=Path, required=True, help="dwie_test_gold.json")
    p.add_argument(
        "--segments", type=Path, required=True, help="Segmentation output (articleId, sentences, predictions)."
    )
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=partitions)
    p = sub.add_parser("mind-input", help="Write the MIND-50 segmentation input.")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=mind_input)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
