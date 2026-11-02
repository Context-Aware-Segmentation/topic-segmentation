"""GLiNER-Relex on Synthetic-100: joint entity and relation extraction, and relations over reference entities.

    python -m knowledge_graph.clinical.gliner_relex joint --out DIR [--notes test]                  # zero-shot joint
    python -m knowledge_graph.clinical.gliner_relex reference-entities --out DIR [--model CHECKPOINT --notes test]
"""

import argparse
from pathlib import Path
from types import MethodType

import torch

from knowledge_graph.clinical.data import CONDITIONS, align, keep_best, note, note_ids, reference_map, segmenter, segments, split
from knowledge_graph.clinical.schema import ENTITY_PROMPTS, PROMPT_TO_RELATION
from knowledge_graph.clinical.scoring import reachable_relations, score_relations
from knowledge_graph.common import write_json

RELEX = "knowledgator/gliner-relex-large-v1.0"
THRESHOLD = 0.3  # entity, adjacency, and relation threshold
JOINT_BATCH, REFERENCE_BATCH = 4, 8  # segments per batch


def resolve_endpoint(entity, by_span, segment, spans, typed):
    """Document span, type, and aligned reference entity of a predicted entity or relation endpoint."""
    if "label" not in entity:  # relation endpoints carry only offsets
        entity = by_span[(int(entity["start"]), int(entity["end"]))]
    begin, end = segment["begin"] + int(entity["start"]), segment["begin"] + int(entity["end"])
    kind = ENTITY_PROMPTS[entity["label"]]
    return begin, end, kind, align(begin, end, kind, spans, typed)


def gold_keys(annotation):
    return sorted({(r["head"], r["type"], r["tail"]) for r in annotation["relations"]})


def joint_predictions(model, texts, batch_size):
    """Joint entity and relation inference with the shared clinical prompts and thresholds."""
    return model.inference(
        texts=texts,
        labels=list(ENTITY_PROMPTS),
        relations=list(PROMPT_TO_RELATION),
        flat_ner=False,
        threshold=THRESHOLD,
        adjacency_threshold=THRESHOLD,
        relation_threshold=THRESHOLD,
        batch_size=batch_size,
        return_relations=True,
    )


def decode_joint(segment, spans, typed, entities, relations):
    """Decode one input into document spans and relations aligned to reference entities for scoring."""
    by_span = {(int(e["start"]), int(e["end"])): e for e in sorted(entities, key=lambda e: e["score"])}

    def endpoint(entity):
        begin, end, kind, key = resolve_endpoint(entity, by_span, segment, spans, typed)
        return {
            "begin": begin,
            "end": end,
            "type": kind,
            "entity_id": key,
            "text": segment["text"][begin - segment["begin"] : end - segment["begin"]],
        }

    decoded_entities = [
        {**endpoint(e), "native_label": e["label"], "score": float(e["score"])} for e in entities
    ]
    decoded_relations = [
        {
            "head": endpoint(r["head"]),
            "tail": endpoint(r["tail"]),
            "type": PROMPT_TO_RELATION[r["relation"]],
            "score": float(r["score"]),
        }
        for r in relations
    ]
    return decoded_entities, decoded_relations


def joint_note(model, nlp, note_id, condition, batch_size):
    """Joint predictions pooled over one note's inputs."""
    source, annotation = note(note_id)
    spans, typed = reference_map(annotation)
    inputs = segments(nlp, source["text"], source["segments"], condition)
    entities, relations = {}, {}
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        entity_batches, relation_batches = joint_predictions(model, [s["text"] for s in batch], batch_size)
        for segment, predicted_entities, predicted_relations in zip(batch, entity_batches, relation_batches):
            decoded_entities, decoded_relations = decode_joint(segment, spans, typed, predicted_entities, predicted_relations)
            for entity in decoded_entities:
                keep_best(entities, (entity["begin"], entity["end"], entity["type"]),
                          {k: entity[k] for k in ("begin", "end", "type", "score", "text")})
            for relation in decoded_relations:
                head, tail, kind = relation["head"], relation["tail"], relation["type"]
                keep_best(relations, (head["entity_id"], kind, tail["entity_id"]),
                          {"head": head["entity_id"], "type": kind, "tail": tail["entity_id"], "score": relation["score"],
                           "head_begin": head["begin"], "head_end": head["end"], "tail_begin": tail["begin"],
                           "tail_end": tail["end"], "segment_id": segment["segment_id"]})
    return {"note_id": note_id, "predicted_entities": list(entities.values()),
            "predicted_relations": list(relations.values()), "predicted_relation_keys": sorted(relations),
            "gold_relation_keys": gold_keys(annotation)}


def joint(args):
    from gliner import GLiNER

    model = GLiNER.from_pretrained(RELEX).to("cuda").eval()
    nlp = segmenter()
    ids = split()["test_ids"] if args.notes == "test" else note_ids()
    for condition in CONDITIONS:
        records = [joint_note(model, nlp, note_id, condition, JOINT_BATCH) for note_id in ids]
        payload = {"model": RELEX, "condition": condition, "documents": records, "evaluation": score_relations(records)}
        write_json(args.out / f"gliner_relex_{condition}.json", payload)
        print(condition, payload["evaluation"], flush=True)


def token_bounds(prepared, entities):
    """First and last model token of every reference entity."""
    result = []
    for i, mentions in enumerate(entities):
        starts, ends = prepared["start_token_map"][i], prepared["end_token_map"][i]
        rows = []
        for entity in mentions:
            hits = [j for j in range(len(starts)) if starts[j] < entity["end"] and ends[j] > entity["start"]]
            rows.append((hits[0], hits[-1]))
        result.append(rows)
    return result


def span_representations(layer, words, indices):
    """The markerV0 span representation for given endpoints, without its fixed-width grid reshape."""
    start, end = layer.project_start(words), layer.project_end(words)
    batch = torch.arange(words.shape[0], device=words.device)[:, None]
    return layer.out_project(torch.cat([start[batch, indices[:, :, 0]], end[batch, indices[:, :, 1]]], -1).relu())


@torch.inference_mode()
def predict_reference_entities(model, texts, entities, threshold=THRESHOLD):
    """Relation scores of GLiNER-Relex over given entity spans: the span layer is replaced, NER is bypassed."""
    prepared = model.prepare_batch(texts, list(ENTITY_PROMPTS), relations=list(PROMPT_TO_RELATION))
    bounds = token_bounds(prepared, entities)
    batch = model.collate_batch(prepared["input_x"], prepared["entity_types"], relation_types=prepared["relation_types"])
    core = model.model
    layer, original = core.span_rep_layer.span_rep_layer, core.represent_spans

    def represent(self, words, mask, prompts, span_idx=None, span_mask=None, labels=None, threshold=0.3):
        width = max(1, max(map(len, bounds)))
        indices = torch.zeros((len(bounds), width, 2), dtype=torch.long, device=words.device)
        valid = torch.zeros((len(bounds), width), dtype=torch.long, device=words.device)
        for i, rows in enumerate(bounds):
            if rows:
                indices[i, :len(rows)] = torch.tensor(rows, device=words.device)
                valid[i, :len(rows)] = 1
        scores = words.new_zeros((len(bounds), words.shape[1], model.config.max_width, len(ENTITY_PROMPTS)))
        return scores, span_representations(layer, words, indices), valid, indices

    core.represent_spans = MethodType(represent, core)
    out = model.run_batch(batch, threshold=THRESHOLD, adjacency_threshold=THRESHOLD)
    core.represent_spans = original
    classes, probabilities = batch["rel_id_to_classes"], out.rel_logits.sigmoid().cpu().tolist()
    pairs, valid = out.rel_idx.cpu().tolist(), out.rel_mask.cpu().tolist()
    outputs = []
    for i in range(len(entities)):
        names = classes[i] if isinstance(classes, list) else classes
        outputs.append([{"head": int(pairs[i][j][0]), "tail": int(pairs[i][j][1]), "relation": names[c + 1], "score": float(p)}
                        for j in range(len(pairs[i])) if valid[i][j]
                        for c, p in enumerate(probabilities[i][j]) if float(p) > threshold and c + 1 in names])
    return outputs


def reference_entities(annotation, segment):
    """Reference mentions inside one input, with input-local and document offsets."""
    unique = {}
    for entity in annotation["entities"]:
        for m in entity["mentions"]:
            if segment["begin"] <= m["begin"] < m["end"] <= segment["end"]:
                unique[(m["begin"], m["end"], entity["entity_id"], entity["type"])] = {
                    "start": m["begin"] - segment["begin"], "end": m["end"] - segment["begin"], "begin": m["begin"],
                    "absolute_end": m["end"], "entity_id": entity["entity_id"], "type": entity["type"], "text": m["text"]}
    return [unique[k] for k in sorted(unique)]


def reference_note(model, nlp, note_id, condition, batch_size):
    """Relations of one note over its reference entities, pooled over its inputs."""
    source, annotation = note(note_id)
    inputs = segments(nlp, source["text"], source["segments"], condition)
    mentions = [reference_entities(annotation, s) for s in inputs]
    best = {}
    for start in range(0, len(inputs), batch_size):
        batch, batch_mentions = inputs[start:start + batch_size], mentions[start:start + batch_size]
        predictions = predict_reference_entities(model, [s["text"] for s in batch], batch_mentions)
        for segment, entities, found in zip(batch, batch_mentions, predictions):
            for relation in found:
                head, tail = entities[relation["head"]], entities[relation["tail"]]
                kind = PROMPT_TO_RELATION[relation["relation"]]
                keep_best(best, (head["entity_id"], kind, tail["entity_id"]),
                          {"head": head["entity_id"], "type": kind, "tail": tail["entity_id"], "score": relation["score"],
                           "segment_id": segment["segment_id"]})
    return {
        "note_id": note_id,
        "segments": [{k: v for k, v in s.items() if k != "text"} for s in inputs],
        "reference_inputs": [{"segment_id": s["segment_id"], "entities": e} for s, e in zip(inputs, mentions)],
        "predicted_relations": list(best.values()),
        "predicted_relation_keys": sorted(best),
        "gold_relation_keys": gold_keys(annotation),
        "reachable_relation_keys": sorted(reachable_relations(annotation, [{e["entity_id"] for e in m} for m in mentions])),
    }


def evaluate_reference_entities(args):
    """Relations over the reference entities: entity endpoints project onto fixed markerV0 spans, with no NER filtering
    and no gold relations as input."""
    from gliner import GLiNER

    model = GLiNER.from_pretrained(args.model).to("cuda").eval()
    nlp = segmenter()
    ids = split()["test_ids"] if args.notes == "test" else note_ids()
    for condition in CONDITIONS:
        records = [reference_note(model, nlp, note_id, condition, REFERENCE_BATCH) for note_id in ids]
        payload = {"model": str(args.model), "condition": condition, "documents": records,
                   "evaluation": score_relations(records)}
        write_json(args.out / f"gliner_relex_gold_{condition}.json", payload)
        print(condition, payload["evaluation"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("joint", help="Zero-shot joint entity and relation extraction.").set_defaults(run=joint)
    p = sub.add_parser("reference-entities", help="Relations over reference entities.")
    p.add_argument("--model", default=RELEX, help="Hugging Face name or fine-tuned checkpoint directory.")
    p.set_defaults(run=evaluate_reference_entities)
    for p in sub.choices.values():
        p.add_argument("--notes", choices=["all", "test"], default="all")
        p.add_argument("--out", type=Path, required=True, help="Directory for predictions and scores.")
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
