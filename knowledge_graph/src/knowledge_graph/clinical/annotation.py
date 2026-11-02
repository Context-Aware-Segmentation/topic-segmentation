"""Synthetic-100 reference annotation: note preparation, two annotation passes, and the deterministic validator.

    python -m knowledge_graph.clinical.annotation prepare    # notes -> sources/, empty annotations/, manifest.json
    python -m knowledge_graph.clinical.annotation annotate   # pass 1 -> annotations/
    python -m knowledge_graph.clinical.annotation review     # independent pass 2 -> reviews/, the reference annotation
    python -m knowledge_graph.clinical.annotation validate   # offsets, labels, signatures, and evidence of reviews/

HPO links of the SYMPTOM clusters follow in knowledge_graph.clinical.hpo_annotation.
"""

import argparse
import json
from pathlib import Path

from knowledge_graph.clinical.agents import agent, bullet_feedback, read_response, run
from knowledge_graph.clinical.data import CORPUS
from knowledge_graph.clinical.schema import ASSERTIONS, ENTITY_TYPES, POLICY, SCHEMA, SIGNATURES
from knowledge_graph.common import load_json, write_json
from knowledge_graph.paths import SYNTHETIC100

NOTES = SYNTHETIC100 / "generation/terra_mimic_formatted.jsonl"
EXCLUSIONS = [
    "Do not annotate dose, route, frequency, normal demographic fields, or headings.",
    "Do not infer relations from medical knowledge or co-occurrence.",
    "Do not output ontology identifiers in this pass.",
]


ANNOTATE = """Read input.json and annotate its single synthetic discharge summary.

Follow the supplied Clinical-10 annotation policy exactly. Return one annotation
object matching schema.json. Use exact Python character offsets into input.text.
Group repeated mentions of the same real clinical concept into one entity.
Review the entire note for repeated mentions. Keep absent, possible, historical,
and conditional mentions with the correct assertion value.

Create only relations explicitly stated by the note. A relation must use an
allowed type and compatible endpoint types, and evidence_mentions must contain
at least one mention from each endpoint. Do not infer relations from medical
knowledge, section membership, lists, or co-occurrence. Do not create relations
from absent or conditional evidence. Do not guess ontology identifiers in this
pass. Return JSON only."""
REVIEW = """Independently review the Clinical-10 annotation in input.json against
the complete synthetic note. Return a corrected full annotation matching
schema.json. Check every paragraph for omitted clinical mentions and every
repeated mention of an existing entity. Verify exact offsets, maximal clinical
concept boundaries, assertion status, entity grouping, relation direction,
relation signature, and evidence mentions. Delete relations that are inferred
from medical knowledge, lists, section membership, or co-occurrence. Add only
relations explicitly stated by the note. Do not guess ontology IDs. Return JSON
only; this must be a substantive second pass rather than a schema-only check."""
# Pass -> job folder, output folder, prompt, feedback header.
PASSES = {
    "annotate": (
        "agent_jobs",
        "annotations",
        ANNOTATE,
        "Previous output failed validation. Correct every item:",
    ),
    "review": ("review_jobs", "reviews", REVIEW, "Correct every validation error:"),
}


def segments(text):
    """Whole note, blocks, and colon-headed sections with character offsets (lines numbered from 1)."""
    from topic_segmentation.data.mimic.rules import split_blocks, split_subsections

    starts, cursor = [None], 0  # starts[n] = offset of line n
    for line in text.splitlines(keepends=True):
        starts.append(cursor)
        cursor += len(line)
    starts.append(len(text))

    def span(first, last):
        begin, end = starts[first], starts[last + 1]
        while end > begin and text[end - 1] in "\r\n":
            end -= 1
        return begin, end

    blocks, sections = [], []
    for block_id, (first, last, lines) in enumerate(split_blocks(text)):
        begin, end = span(first, last)
        blocks.append({"segment_id": f"block_{block_id:03d}", "begin": begin, "end": end, "text": text[begin:end]})
        for index, (label, header, body) in enumerate(split_subsections(lines)):
            numbers = [n for n, _ in body] + ([header] if header is not None else [])
            if not numbers:
                continue
            begin, end = span(min(numbers), max(numbers))
            sections.append(
                {
                    "segment_id": f"block_{block_id:03d}_section_{index:03d}",
                    "block_id": block_id,
                    "label": label,
                    "begin": begin,
                    "end": end,
                    "text": text[begin:end],
                }
            )
    return {
        "document": [{"segment_id": "document", "begin": 0, "end": len(text), "text": text}],
        "block": blocks,
        "section": sections,
    }


def prepare(notes, out):
    """sources/ with each note's text and block/section spans, empty annotations/, and manifest.json."""
    rows = [json.loads(line) for line in notes.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents = []
    for row in rows:
        note_id, text = str(row["synthetic_id"]), str(row["text"])
        write_json(out / "sources" / f"{note_id}.json", {"note_id": note_id, "text": text, "segments": segments(text)})
        annotation = out / "annotations" / f"{note_id}.json"
        if not annotation.exists():  # rerunning prepare keeps existing annotations
            write_json(annotation, {"note_id": note_id, "annotator": "", "entities": [], "relations": [], "notes": []})
        documents.append({"note_id": note_id, "source": f"sources/{note_id}.json",
                          "annotation": f"annotations/{note_id}.json"})
    write_json(out / "manifest.json", {"documents": documents})
    return documents


def validate(source: dict, annotation: dict) -> list[str]:
    """Check annotation offsets, text, identifiers, labels, and relations against the source note."""
    text, errors = source["text"], []
    if annotation.get("note_id") != source.get("note_id"):
        errors.append("annotation note_id does not match source note_id")
    if not str(annotation.get("annotator") or "").strip():
        errors.append("annotator is empty")
    if not annotation.get("entities"):
        errors.append("annotation contains no entities")
    entities, mentions, spans = {}, {}, set()  # mentions: mention_id -> (entity_id, mention)
    for entity in annotation.get("entities", []):
        entity_id = entity.get("entity_id")
        if not entity_id or entity_id in entities:
            errors.append(f"invalid or duplicate entity_id: {entity_id!r}")
            continue
        entities[entity_id] = entity
        if entity.get("type") not in ENTITY_TYPES:
            errors.append(f"{entity_id}: unsupported entity type {entity.get('type')!r}")
        if not entity.get("mentions"):
            errors.append(f"{entity_id}: entity has no mentions")
        for mention in entity.get("mentions", []):
            mention_id = mention.get("mention_id")
            if not mention_id or mention_id in mentions:
                errors.append(f"{entity_id}: invalid or duplicate mention_id {mention_id!r}")
                continue
            mentions[mention_id] = (entity_id, mention)
            begin, end, surface = mention.get("begin"), mention.get("end"), mention.get("text")
            if not isinstance(begin, int) or not isinstance(end, int) or not 0 <= begin < end <= len(text):
                errors.append(f"{mention_id}: invalid offset {begin!r}:{end!r}")
            elif text[begin:end] != surface:
                errors.append(f"{mention_id}: offset text mismatch: source={text[begin:end]!r}, annotation={surface!r}")
            if (begin, end, entity_id) in spans:
                errors.append(f"{mention_id}: duplicate mention span in {entity_id}")
            spans.add((begin, end, entity_id))
            if mention.get("assertion") not in ASSERTIONS:
                errors.append(f"{mention_id}: invalid assertion {mention.get('assertion')!r}")
    relation_ids, keys = set(), set()
    for relation in annotation.get("relations", []):
        relation_id = relation.get("relation_id")
        if not relation_id or relation_id in relation_ids:
            errors.append(f"invalid or duplicate relation_id: {relation_id!r}")
        relation_ids.add(relation_id)
        head, tail, kind = relation.get("head"), relation.get("tail"), relation.get("type")
        if head not in entities or tail not in entities:
            errors.append(f"{relation_id}: unknown endpoint {head!r} -> {tail!r}")
            continue
        if head == tail:
            errors.append(f"{relation_id}: self-loop relation")
        if kind not in SIGNATURES:
            errors.append(f"{relation_id}: unsupported relation type {kind!r}")
        elif (
            entities[head].get("type") not in SIGNATURES[kind][0]
            or entities[tail].get("type") not in SIGNATURES[kind][1]
        ):
            errors.append(
                f"{relation_id}: incompatible signature {entities[head].get('type')} -{kind}-> {entities[tail].get('type')}"
            )
        if (head, kind, tail) in keys:
            errors.append(f"{relation_id}: duplicate relation {(head, kind, tail)}")
        keys.add((head, kind, tail))
        evidence = relation.get("evidence_mentions") or []
        if not evidence:
            errors.append(f"{relation_id}: missing evidence_mentions")
        if unknown := [m for m in evidence if m not in mentions]:
            errors.append(f"{relation_id}: unknown evidence mentions {unknown}")
        if not {head, tail} <= {mentions[m][0] for m in evidence if m in mentions}:
            errors.append(f"{relation_id}: evidence does not contain mentions of both endpoints")
        if negated := [
            m for m in evidence if m in mentions and mentions[m][1].get("assertion") in {"absent", "conditional"}
        ]:
            errors.append(f"{relation_id}: relation uses non-asserted evidence mentions {negated}")
    return errors


def response_checker(source, document, generation, output):
    """Record the generation settings and save a response that passes annotation validation."""

    def check(response):
        candidate, errors = read_response(response)
        if errors:
            return errors
        candidate.update(annotator=generation["model"], generation=generation)
        write_json(response, candidate)
        errors = validate(source, candidate)
        if not errors:
            write_json(output, candidate)
            entities, relations = len(candidate["entities"]), len(candidate["relations"])
            print(f"{generation['stage']} complete: {document['note_id']} entities={entities} relations={relations}", flush=True)
        return errors

    return check


def already_valid(output, source):
    candidate, errors = read_response(output)
    return not (errors or validate(source, candidate))


def agent_pass(args):
    """Pass 1 (annotate) or the independent pass 2 (review) for every selected note without a valid output."""
    data = args.data_dir.resolve()
    folder, outputs, prompt, feedback = PASSES[args.command]
    jobs = []
    for document in selected_notes(data, args.note_ids):
        note_id = document["note_id"]
        source = load_json(data / document["source"])
        inputs = {"note_id": note_id, "text": source["text"], "policy": POLICY}
        if args.command == "review":
            inputs["pass1_annotation"] = load_json(data / document["annotation"])
        else:
            inputs["important_exclusions"] = EXCLUSIONS
        job = data / folder / note_id
        write_json(job / "input.json", inputs)
        write_json(job / "schema.json", SCHEMA)
        output = data / outputs / f"{note_id}.json"
        if already_valid(output, source):
            continue
        generation = {"model": args.model, "reasoning_effort": args.reasoning_effort, "stage": args.command}
        check = response_checker(source, document, generation, output)
        jobs.append(
            lambda s, job=job, check=check: agent(
                job, prompt, check, args, s, attempts=args.max_attempts, followup=bullet_feedback(feedback, 80)
            )
        )
    return run(jobs, args)


def selected_notes(data, note_ids):
    documents = load_json(data / "manifest.json")["documents"]
    return [d for d in documents if not note_ids or d["note_id"] in note_ids.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Package the 100 synthetic notes with their block and section spans.")
    p.add_argument("--notes", type=Path, default=NOTES)
    p.add_argument("--data-dir", type=Path, default=CORPUS)
    for name in PASSES:
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, default=CORPUS)
        p.add_argument("--model", default="gpt-5.6-luna")
        p.add_argument("--reasoning-effort", default="max")
        p.add_argument("--max-parallel", type=int, default=5)
        p.add_argument("--max-attempts", type=int, default=3)
        p.add_argument("--note-ids", help="Comma-separated note IDs (default: all).")
    p = sub.add_parser("validate", help="Validate every reviewed annotation.")
    p.add_argument("--data-dir", type=Path, default=CORPUS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.notes, args.data_dir)
        return 0
    if args.command == "validate":
        documents = load_json(args.data_dir / "manifest.json")["documents"]
        failed = 0
        for document in documents:
            errors = validate(
                load_json(args.data_dir / document["source"]),
                load_json(args.data_dir / "reviews" / f"{document['note_id']}.json"),
            )
            failed += bool(errors)
            for error in errors:
                print(f"{document['note_id']}: {error}")
        print(f"validated {len(documents)} notes, {failed} with errors")
        return int(bool(failed))
    return agent_pass(args)


if __name__ == "__main__":
    raise SystemExit(main())
