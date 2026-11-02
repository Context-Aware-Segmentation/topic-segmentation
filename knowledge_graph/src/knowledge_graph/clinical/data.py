"""Synthetic-100 notes, the 80/10/10 split, and input segmentation."""

import random

from knowledge_graph.common import load_json
from knowledge_graph.paths import DATA, RUNS, SYNTHETIC100

CORPUS = SYNTHETIC100
CLINICAL_RUNS = RUNS / "clinical"
HPO_OBO = DATA / "resources" / "hpo" / "hp.obo"
CONDITIONS = ["fixed128", "fixed256", "fixed512", "sentence", "block", "section"]


def note(note_id):
    """Source note (text and reference segments) and its reviewed annotation."""
    return load_json(CORPUS / "sources" / f"{note_id}.json"), load_json(CORPUS / "reviews" / f"{note_id}.json")


def note_ids():
    return [item["note_id"] for item in load_json(CORPUS / "manifest.json")["documents"]]


def split(seed=42):
    """Note-level 80/10/10 split of the manifest, shared by every fine-tuned model."""
    ids = sorted(note_ids())
    random.Random(seed).shuffle(ids)
    return {"train_ids": sorted(ids[:80]), "validation_ids": sorted(ids[80:90]), "test_ids": sorted(ids[90:])}


def documents(ids=None):
    """Notes with their reference mentions for exact-span NER scoring, in manifest order."""
    output = []
    for note_id in note_ids():
        if ids is not None and note_id not in ids:
            continue
        source, annotation = note(note_id)
        mentions = [
            {
                "id": mention["mention_id"],
                "entity_id": entity["entity_id"],
                "begin": int(mention["begin"]),
                "end": int(mention["end"]),
                "text": mention["text"],
                "type": entity["type"],
                "assertion": mention.get("assertion"),
            }
            for entity in annotation["entities"]
            for mention in entity["mentions"]
        ]
        output.append(
            {"id": note_id, "text": source["text"], "gold_mentions": mentions, "reference_segments": source["segments"]}
        )
    return output


def segmenter():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def token_pieces(nlp, text, begin, end, limit):
    doc = nlp.make_doc(text[begin:end])
    if not doc:
        return []
    pieces = []
    for token_start in range(0, len(doc), limit):
        token_end = min(len(doc), token_start + limit)
        local_begin = doc[token_start].idx
        local_end = doc[token_end - 1].idx + len(doc[token_end - 1])
        pieces.append({"begin": begin + local_begin, "end": begin + local_end})
    return pieces


def sentence_segments(nlp, text):
    return [
        {"begin": sentence.start_char, "end": sentence.end_char}
        for sentence in nlp(text).sents
        if sentence.text.strip()
    ]


def pack_sentences(nlp, text, sentences, limit):
    """Pack whole sentences up to the token limit; a longer sentence is split at token boundaries."""
    output, current, current_tokens = [], [], 0
    for sentence in sentences:
        count = len(nlp.make_doc(text[sentence["begin"] : sentence["end"]]))
        if count > limit:
            if current:
                output.append({"begin": current[0]["begin"], "end": current[-1]["end"]})
                current, current_tokens = [], 0
            output.extend(token_pieces(nlp, text, sentence["begin"], sentence["end"], limit))
            continue
        if current and current_tokens + count > limit:
            output.append({"begin": current[0]["begin"], "end": current[-1]["end"]})
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += count
    if current:
        output.append({"begin": current[0]["begin"], "end": current[-1]["end"]})
    return output


def cap_segments(nlp, text, sources, limit):
    output = []
    for source_index, source in enumerate(sources):
        begin, end = int(source["begin"]), int(source["end"])
        doc = nlp(text[begin:end])
        if len(doc) <= limit:
            pieces = [{"begin": begin, "end": end}]
        else:
            local = [
                {"begin": begin + sent.start_char, "end": begin + sent.end_char}
                for sent in doc.sents
                if sent.text.strip()
            ]
            pieces = pack_sentences(nlp, text, local, limit)
        for piece_index, piece in enumerate(pieces):
            output.append(
                {
                    "segment_id": f"r{source_index:04d}_p{piece_index:03d}",
                    "source_index": source_index,
                    "begin": piece["begin"],
                    "end": piece["end"],
                    "label": source.get("label"),
                    "text": text[piece["begin"] : piece["end"]],
                }
            )
    return output


def segments(nlp, text, reference, condition):
    """Model inputs of one condition: fixed token budgets, sentences, or reference blocks/sections (512-token cap)."""
    if condition.startswith("fixed"):
        limit = int(condition.removeprefix("fixed"))
        return cap_segments(nlp, text, pack_sentences(nlp, text, sentence_segments(nlp, text), limit), limit)
    if condition == "sentence":
        return cap_segments(nlp, text, sentence_segments(nlp, text), 512)
    return cap_segments(nlp, text, reference[condition], 512)


def keep_best(table, key, row):
    """Keep the highest-scoring row per key (first seen wins ties)."""
    if key not in table or row["score"] > table[key]["score"]:
        table[key] = row


def deduplicate(rows):
    """One prediction per document span and concept ID, keeping the highest score (1.0 when a model gives none)."""
    def score(row):
        return 1.0 if row.get("score") is None else float(row["score"])

    best = {}
    for row in rows:
        key = (int(row["begin"]), int(row["end"]), row.get("concept_id"))
        if key not in best or score(row) > score(best[key]):
            best[key] = row
    return sorted(best.values(), key=lambda row: (row["begin"], row["end"], row.get("concept_id") or ""))


def reference_map(annotation):
    """Reference entity IDs by exact span and by exact typed span."""
    spans, typed = {}, {}
    for entity in annotation["entities"]:
        for mention in entity["mentions"]:
            spans.setdefault((mention["begin"], mention["end"]), set()).add(entity["entity_id"])
            typed.setdefault((mention["begin"], mention["end"], entity["type"]), set()).add(entity["entity_id"])
    return spans, typed


def align(begin, end, kind, spans, typed):
    """Reference entity of an exact predicted span (typed match first); unmatched spans keep their own ID."""
    candidates = typed.get((begin, end, kind), set())
    if len(candidates) != 1:
        candidates = spans.get((begin, end), set())
    return next(iter(candidates)) if len(candidates) == 1 else f"pred:{begin}:{end}:{kind}"
