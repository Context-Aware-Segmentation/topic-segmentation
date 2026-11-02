"""HPO links of the Synthetic-100 SYMPTOM clusters: GPT-5.6-luna adjudication over lexical candidates, then an independent review.

python -m knowledge_graph.clinical.hpo_annotation adjudicate   # reviews/ -> hpo_adjudication/review_v2/
python -m knowledge_graph.clinical.hpo_annotation review       # review_v2/ -> hpo_adjudication/review_final/, the reference links
"""

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from knowledge_graph.clinical.agents import agent, bullet_feedback, read_response, run
from knowledge_graph.clinical.annotation import selected_notes
from knowledge_graph.clinical.data import CORPUS, HPO_OBO
from knowledge_graph.clinical.hpo import HpoOntology, normalize_surface
from knowledge_graph.clinical.schema import object_schema
from knowledge_graph.common import load_json, write_json

FORBIDDEN_FALLBACK_IDS = {"HP:0000118"}
ADJUDICATE = """Adjudicate every symptom entity in input.json to the Human Phenotype
Ontology. Use the complete mention cluster and full note context, not lexical
score alone. Choose a current, specific HPO concept only when clinically
defensible; otherwise return NIL-HPO. Never use generic HP:0000118. Candidate
rows are retrieval suggestions, not gold. If none is adequate, you may search
the local hp.obo named in input.json, but every linked ID and canonical label
must occur there. Preserve entity IDs and mention objects exactly. Provide a
short clinical rationale for every decision. Return JSON only."""
REVIEW = """Independently review every proposed HPO decision in input.json.
Verify each SYMPTOM cluster against the full note context and the local hp.obo.
Correct lexically tempting but clinically wrong, anatomically wrong, overly
broad, overly specific, assertion-confused, obsolete, or noncanonical links.
Use NIL-HPO when no defensible HPO concept exists and never use HP:0000118.
Preserve entity IDs and mention objects exactly. Return a complete corrected
decision file as JSON only."""


def word_set(value: str) -> set[str]:
    return set(normalize_surface(value).split())


def candidate_score(query: str, surface: str) -> float:
    query_norm, surface_norm = normalize_surface(query), normalize_surface(surface)
    if query_norm == surface_norm:
        return 1.0
    query_tokens, surface_tokens = word_set(query_norm), word_set(surface_norm)
    overlap = len(query_tokens & surface_tokens) / max(len(query_tokens | surface_tokens), 1)
    return 0.65 * overlap + 0.35 * SequenceMatcher(None, query_norm, surface_norm).ratio()


def surface_index(ontology: HpoOntology) -> dict[str, list[tuple[str, str, str]]]:
    index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for identifier, term in ontology.terms.items():
        for surface in (term.label, *term.synonyms):
            for token in word_set(surface):
                index[token].append((identifier, term.label, surface))
    return index


def candidates(
    entity: dict[str, Any], index: dict[str, list[tuple[str, str, str]]], limit: int
) -> list[dict[str, Any]]:
    """Top lexical HPO candidates over all surfaces of a mention cluster (retrieval suggestions, not decisions)."""
    scores: dict[str, tuple[float, str, str]] = {}
    for query in sorted({mention["text"] for mention in entity["mentions"]}):
        for identifier, label, surface in sorted({row for token in word_set(query) for row in index.get(token, [])}):
            score = candidate_score(query, surface)
            if identifier not in scores or score > scores[identifier][0]:
                scores[identifier] = (score, label, surface)
    ranked = sorted(scores.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
    return [
        {"hpo_id": identifier, "hpo_label": label, "matched_surface": matched, "lexical_score": round(score, 4)}
        for identifier, (score, label, matched) in ranked
    ]


def symptom_entities(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entity["entity_id"]: entity for entity in annotation["entities"] if entity.get("type") == "SYMPTOM"}


def validate(decision: dict[str, Any], annotation: dict[str, Any], names: dict[str, str]) -> list[str]:
    """Coverage of every SYMPTOM cluster, known canonical HPO IDs and labels, NIL rules, and unchanged mentions."""
    note_id = annotation["note_id"]
    if decision.get("note_id") != note_id:
        return [f"note_id mismatch: expected {note_id}, got {decision.get('note_id')}"]
    source = symptom_entities(annotation)
    rows = decision.get("symptom_entities")
    if not isinstance(rows, list):
        return ["symptom_entities must be a list"]
    errors: list[str] = []
    observed = [row.get("entity_id") for row in rows]
    if len(observed) != len(set(observed)):
        errors.append("duplicate entity_id in adjudication")
    if set(observed) != set(source):
        errors.append(
            f"entity coverage mismatch: missing={sorted(set(source) - set(observed))}; "
            f"unexpected={sorted(set(observed) - set(source))}"
        )
    required = {"entity_id", "hpo_id", "hpo_label", "hpo_status", "assertion_summary", "mentions"}
    for row in rows:
        entity_id = row.get("entity_id", "<missing>")
        if missing := sorted(required - set(row)):
            errors.append(f"{entity_id}: missing fields {missing}")
            continue
        status, hpo_id, hpo_label = row["hpo_status"], row["hpo_id"], row["hpo_label"]
        if status not in {"linked", "NIL-HPO"}:
            errors.append(f"{entity_id}: invalid hpo_status {status!r}")
        if status == "linked":
            if not isinstance(hpo_id, str) or hpo_id not in names:
                errors.append(f"{entity_id}: unknown HPO ID {hpo_id!r}")
            elif hpo_id in FORBIDDEN_FALLBACK_IDS:
                errors.append(f"{entity_id}: generic fallback HPO ID {hpo_id} is not permitted")
            elif hpo_label != names[hpo_id]:
                errors.append(
                    f"{entity_id}: HPO label mismatch for {hpo_id}: expected {names[hpo_id]!r}, got {hpo_label!r}"
                )
        elif hpo_id is not None or hpo_label is not None:
            errors.append(f"{entity_id}: NIL-HPO requires null hpo_id and hpo_label")
        if entity_id in source:
            expected = [
                {key: m[key] for key in ("mention_id", "text", "begin", "end")} for m in source[entity_id]["mentions"]
            ]
            if row["mentions"] != expected:
                errors.append(f"{entity_id}: mention list differs from source annotation")
    return errors


def output_schema() -> dict[str, Any]:
    mention = object_schema(
        {
            "mention_id": {"type": "string"},
            "text": {"type": "string"},
            "begin": {"type": "integer"},
            "end": {"type": "integer"},
        }
    )
    entity = object_schema(
        {
            "entity_id": {"type": "string"},
            "hpo_id": {"type": ["string", "null"]},
            "hpo_label": {"type": ["string", "null"]},
            "hpo_status": {"enum": ["linked", "NIL-HPO"]},
            "assertion_summary": {"type": "string"},
            "mentions": {"type": "array", "items": mention},
            "rationale": {"type": "string"},
        }
    )
    return object_schema(
        {
            "note_id": {"type": "string"},
            "adjudicator": {"type": "string"},
            "symptom_entities": {"type": "array", "items": entity},
        }
    )


def adjudication_job(data: Path, annotation: dict[str, Any], ontology: HpoOntology, index, limit: int) -> Path:
    note_id = annotation["note_id"]
    text = load_json(data / "sources" / f"{note_id}.json")["text"]
    entities = [
        {
            "entity_id": entity["entity_id"],
            "mentions": [{key: m[key] for key in ("mention_id", "text", "begin", "end")} for m in entity["mentions"]],
            "assertions": sorted({m["assertion"] for m in entity["mentions"]}),
            "context": [text[max(0, m["begin"] - 140) : m["end"] + 140] for m in entity["mentions"][:3]],
            "candidates": candidates(entity, index, limit),
        }
        for entity in annotation["entities"]
        if entity["type"] == "SYMPTOM"
    ]
    job = data / "hpo_adjudication" / "jobs" / note_id
    job.mkdir(parents=True, exist_ok=True)
    write_json(
        job / "input.json",
        {
            "note_id": note_id,
            "text": text,
            "ontology": {
                "name": "Human Phenotype Ontology",
                "version": ontology.version,
                "source_obo": str(ontology.path),
            },
            "policy": {
                "scope": "Every SYMPTOM entity in the reviewed Clinical-10 annotation",
                "linked": "Choose the most specific defensible current HPO concept",
                "nil": "Use NIL-HPO when no candidate is clinically defensible",
                "forbidden_generic_id": "HP:0000118",
                "important": "Lexical scores retrieve candidates; they are not gold decisions",
            },
            "symptom_entities": entities,
        },
    )
    write_json(job / "schema.json", output_schema())
    return job


def review_job(data: Path, note_id: str) -> Path:
    job = data / "hpo_adjudication" / "review_jobs" / note_id
    job.mkdir(parents=True, exist_ok=True)
    write_json(
        job / "input.json",
        {
            "note_id": note_id,
            "hpo_obo": str(HPO_OBO),
            "text": load_json(data / "sources" / f"{note_id}.json")["text"],
            "annotation": load_json(data / "reviews" / f"{note_id}.json"),
            "proposed_decision": load_json(data / "hpo_adjudication" / "review_v2" / f"{note_id}.json"),
        },
    )
    write_json(job / "schema.json", output_schema())
    return job


def response_checker(annotation: dict[str, Any], names: dict[str, str], generation: dict, output: Path):
    def check(response: Path) -> list[str]:
        candidate, errors = read_response(response)
        if errors:
            return errors
        candidate.update(adjudicator=generation["model"], generation=generation)
        write_json(response, candidate)
        errors = validate(candidate, annotation, names)
        if not errors:
            write_json(output, candidate)
            linked = sum(entity["hpo_status"] == "linked" for entity in candidate["symptom_entities"])
            print(
                f"{generation['stage']} complete: {annotation['note_id']} linked={linked} "
                f"nil={len(candidate['symptom_entities']) - linked}",
                flush=True,
            )
        return errors

    return check


def already_valid(output, annotation, names):
    candidate, errors = read_response(output)
    return not (errors or validate(candidate, annotation, names))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["adjudicate", "review"])
    parser.add_argument("--data-dir", type=Path, default=CORPUS)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--note-ids", help="Comma-separated note IDs (default: all).")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    documents = selected_notes(data, args.note_ids)
    ontology = HpoOntology(HPO_OBO)
    index = surface_index(ontology) if args.stage == "adjudicate" else None
    jobs = []
    for document in sorted(documents, key=lambda d: d["note_id"]):
        note_id = document["note_id"]
        name = f"{note_id}.json"
        annotation = load_json(data / "reviews" / name)
        if args.stage == "adjudicate":
            output, prompt = (
                data / "hpo_adjudication/review_v2" / name,
                ADJUDICATE,
            )
            job = adjudication_job(data, annotation, ontology, index, args.candidate_limit)
        else:
            output, prompt = (
                data / "hpo_adjudication/review_final" / name,
                REVIEW,
            )
            job = review_job(data, note_id)
        if already_valid(output, annotation, ontology.names):
            continue
        generation = {"model": args.model, "reasoning_effort": args.reasoning_effort, "stage": args.stage}
        check = response_checker(annotation, ontology.names, generation, output)
        jobs.append(
            lambda s, job=job, prompt=prompt, check=check: agent(
                job,
                prompt,
                check,
                args,
                s,
                attempts=args.max_attempts,
                followup=bullet_feedback("Correct every validation error:", 100),
            )
        )
    return run(jobs, args)


if __name__ == "__main__":
    raise SystemExit(main())
