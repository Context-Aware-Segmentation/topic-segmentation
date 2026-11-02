"""Recognize and link news entities with spaCy or ReLiK, using document or segment inputs."""

import argparse
import re
from pathlib import Path

from knowledge_graph.common import load_json, normalize_label, normalize_space, write_json
from knowledge_graph.news.data import PARTITIONS, partition_segments, predicted_segments, whole_documents
from knowledge_graph.news.relik_loading import load_relik

SPACY = "en_core_web_lg"
MIN_OVERLAP = 0.8
SPACY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART"}
# spaCy spans that are generic words or dates rather than proper names.
STOP_SURFACES = set(
    """a an the this that these those he she it they we i you me him her them us people person residents resident
    students student children child kids kid company companies team teams city state country world community center
    school looking making going including according construction government federal championship car life look vice
    pdf news media press investment investments llc inc corp ltd co""".split()
) | {"community center", "the news", "more than", "looking for", "monday night"}
DATE_WORDS = set(
    """monday tuesday wednesday thursday friday saturday sunday january february march april may june july august
    september october november december today tomorrow yesterday""".split()
)
NUMBER = re.compile(r"^\$?\d+(?:[.,:/-]\d+)*(?:%|[kKmMbB])?$")
ACRONYM = re.compile(r"^[A-Z0-9][A-Z0-9&.+-]{1,12}$")
QID = re.compile(r"^Q\d+$")


def to_document_offsets(entity, segment):
    return {
        **entity,
        "begin": segment["begin"] + int(entity["begin"]),
        "end": segment["begin"] + int(entity["end"]),
        "segment_id": segment["segment_id"],
    }


def is_proper_name(surface):
    """Whether a spaCy span looks like a proper name rather than a generic word, date, or number."""
    surface = normalize_space(surface)
    lower = surface.lower()
    if not surface or lower in STOP_SURFACES or lower in DATE_WORDS or NUMBER.fullmatch(surface):
        return False
    if len(surface) <= 2 and not ACRONYM.fullmatch(surface):
        return False
    letters = [char for char in surface if char.isalpha()]
    if letters and all(not char.isupper() for char in letters):
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9&.'+-]*", surface)
    return bool(
        words
        and (
            ACRONYM.fullmatch(surface)
            or any(c.isupper() for word in words for c in word)
            or any(c.isdigit() for c in surface)
        )
    )


def linker_qid(value):
    """The linker reports numeric Wikidata IDs; return them as QIDs."""
    value = str(value).strip() if value is not None else ""
    if value.startswith("Q") and value[1:].isdigit():
        return value
    return f"Q{value}" if value.isdigit() else None


def linker_spans(doc):
    """Linker spans that carry a Wikidata QID, deduplicated by span and QID."""
    found = {}
    for linked in doc._.linkedEntities:
        span, qid = linked.get_span(), linker_qid(linked.get_id())
        if span is None or span.start_char == span.end_char or not qid:
            continue
        key = (span.start_char, span.end_char, qid)
        if key not in found:
            found[key] = {"begin": key[0], "end": key[1], "wikidataId": qid, "prior": float(linked.get_prior() or 0.0)}
    return sorted(found.values(), key=lambda e: (e["begin"], e["end"], e["wikidataId"]))


def proper_names(doc):
    """Proper-name spaCy spans, each with the QID of the best linker span overlapping 80% of the longer span."""
    candidates, found = linker_spans(doc), []
    for ent in doc.ents:
        if ent.label_ not in SPACY_LABELS or not is_proper_name(ent.text):
            continue
        best, best_rank = None, ()
        for link in candidates:
            overlap = min(ent.end_char, link["end"]) - max(ent.start_char, link["begin"])
            if overlap / max(ent.end_char - ent.start_char, link["end"] - link["begin"]) < MIN_OVERLAP:
                continue
            exact = int(ent.start_char == link["begin"] and ent.end_char == link["end"])
            rank = (exact, overlap, link["end"] - link["begin"], link["prior"])
            if rank > best_rank:
                best, best_rank = link, rank
        found.append(
            {
                "surface": ent.text,
                "begin": ent.start_char,
                "end": ent.end_char,
                "wikidataId": best["wikidataId"] if best else None,
            }
        )
    return sorted(found, key=lambda e: (e["begin"], e["end"]))


class SpacyLinker:
    """spaCy proper names linked to Wikidata."""

    def __init__(self):
        import spacy
        import spacy_entity_linker  # noqa: F401 - registers entityLinker

        self.nlp = spacy.load(SPACY)
        self.nlp.add_pipe("entityLinker", last=True)

    def predict(self, document, segments):
        self.nlp.max_length = max(2_000_000, len(document["text"]) + 100)
        entities = []
        parsed_segments = self.nlp.pipe([segment["text"] for segment in segments], batch_size=8)
        for segment, parsed in zip(segments, parsed_segments):
            entities += [to_document_offsets(entity, segment) for entity in proper_names(parsed)]
        return sorted(entities, key=lambda entity: (entity["begin"], entity["end"]))


def qid_of_candidate(candidate):
    """QID of a retrieved candidate, from its ID or its metadata (index-dependent)."""
    if candidate is None:
        return None
    if QID.fullmatch(value := normalize_space(getattr(candidate, "id", None))):
        return value
    metadata = getattr(candidate, "metadata", None) or {}
    for key in ("wikidataId", "WikidataId", "wikidata_id", "wikidata", "qid", "QID", "kb_id", "entity_id"):
        if QID.fullmatch(value := normalize_space(metadata.get(key))):
            return value
    return None


def candidate_qids_by_title(result):
    """QIDs of retrieved candidates (e.g. of the 2M-popular index) by normalized title; the first candidate wins."""
    qids = {}
    candidates = getattr(result, "candidates", None)
    if candidates is None:
        return qids
    for window in candidates.span or []:
        for group in window:
            for candidate in group:
                key = normalize_label(getattr(candidate, "text", ""))
                if key and key not in qids:
                    qids[key] = qid_of_candidate(candidate)
    return qids


class ReLiK:
    """ReLiK mentions with Wikipedia titles and available candidate QIDs."""

    def __init__(self, index=None):
        self.model = load_relik(index)

    def predict(self, document, segments):
        """ReLiK spans of every segment at document offsets, with the candidate QID when the index provides one."""
        results = self.model(
            [segment["text"] for segment in segments],
            annotation_type="char",
            top_k=50,
            retriever_batch_size=16,
            reader_batch_size=16,
            progress_bar=False,
        )
        if len(segments) == 1:
            results = [results]
        entities = []
        for segment, result in zip(segments, results):
            qids = candidate_qids_by_title(result)
            for span in result.spans:
                title = str(span.label)
                entity = {"surface": str(span.text), "begin": span.start, "end": span.end, "wikipedia_title": title}
                if qid := qids.get(normalize_label(title)):
                    entity["relik_candidate_qid"] = qid
                entities.append(to_document_offsets(entity, segment))
        return entities


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["spacy", "relik"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="Documents JSON with id and text.")
    parser.add_argument("--out", type=Path, required=True)
    units = parser.add_mutually_exclusive_group()
    units.add_argument("--segments", type=Path, help="Segmentation output (articleId, sentences, predictions).")
    units.add_argument("--partitions", type=Path, help="DWIE partitions (news.data partitions).")
    parser.add_argument("--partition", choices=PARTITIONS, help="The DWIE partition to process.")
    parser.add_argument("--index", help="ReLiK retrieval index (default: the model's own).")
    args = parser.parse_args()
    documents = load_json(args.input)["documents"]
    if args.partitions:
        units = partition_segments(documents, args.partitions, args.partition)
    elif args.segments:
        units = predicted_segments(documents, args.segments)
    else:
        units = whole_documents(documents)
    model = SpacyLinker() if args.backend == "spacy" else ReLiK(args.index)
    write_json(args.out, {"documents": [{"id": document["id"], "text": document["text"],
                                         "entities": model.predict(document, segments)}
                                        for document, segments in units]})


if __name__ == "__main__":
    main()
