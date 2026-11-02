"""Clinical relation extraction with GLiREL: inference and scoring for any checkpoint (predict), and adaptation (finetune).

    python -m knowledge_graph.clinical.glirel predict --entities gold --out-dir DIR            # zero-shot, 100 notes
    python -m knowledge_graph.clinical.glirel predict --entities predicted --notes test ...    # fine-tuned GLiNER mentions
    python -m knowledge_graph.clinical.glirel finetune --output-dir DIR                         # train on 80 notes, select on 10

Every run covers the six input conditions; relations predicted in different inputs are pooled per note.
"""

import argparse
import json
import random
from pathlib import Path
from typing import NamedTuple

from knowledge_graph.clinical.data import (
    CONDITIONS, CLINICAL_RUNS, align, keep_best, note, note_ids, reference_map, segmenter, segments, split,
)
from knowledge_graph.clinical.schema import ENTITY_PROMPTS, PROMPT_TO_RELATION, RELATION_PROMPTS
from knowledge_graph.clinical.scoring import metric, score_relations
from knowledge_graph.common import EarlyStopping, load_json, write_json

ZERO_SHOT = "jackboyla/glirel-large-v0"
THRESHOLD, TOP_K = 0.5, 1  # GLiREL relation score cutoff and labels per pair
NER_THRESHOLD = 0.3  # GLiNER mention score cutoff for predicted entities
SEED, EPOCHS, PATIENCE = 42, 10, 3  # GLiREL adaptation; the seed also fixes the note split


def predict_relations(model, tokens, entities):
    """GLiREL relations between token-level entity spans (inclusive ends)."""
    return model.predict_relations(tokens, list(PROMPT_TO_RELATION), threshold=THRESHOLD, ner=entities, top_k=TOP_K)


class Mention(NamedTuple):
    """A candidate entity mention at note offsets."""

    rank: float  # lower wins when two mentions cover the same tokens: 0 for reference mentions, -score for predictions
    begin: int
    end: int
    key: str  # reference entity id, or the id of an unmatched prediction
    type: str
    text: str


def glirel_input(doc, segment, mentions):
    """Align mentions to a tokenized segment; return token text, NER spans, and the entity key of each token span.

    Mentions outside the segment are skipped, other boundaries expand to whole tokens, and two mentions on the same
    tokens keep the exact one, then the lower rank, then the earlier span.
    """
    starts = {token.idx: token.i for token in doc}
    ends = {token.idx + len(token.text): token.i + 1 for token in doc}
    aligned = []
    for mention in mentions:
        if not segment["begin"] <= mention.begin < mention.end <= segment["end"]:
            continue
        local_begin, local_end = mention.begin - segment["begin"], mention.end - segment["begin"]
        exact = local_begin in starts and local_end in ends
        if exact:
            first, last = starts[local_begin], ends[local_end]
        elif (span := doc.char_span(local_begin, local_end, alignment_mode="expand")) is not None:
            first, last = span.start, span.end
        else:
            continue
        aligned.append((not exact, mention.rank, first, last, mention))
    ner, mapping = [], {}
    for _, _, first, last, mention in sorted(aligned, key=lambda row: row[:4]):
        if (first, last) not in mapping:
            mapping[first, last] = mention.key
            ner.append([first, last - 1, mention.type, mention.text])
    return [token.text for token in doc], ner, mapping


def reference_mentions(annotation):
    """Reference mentions keyed by their entity."""
    return [
        Mention(0, m["begin"], m["end"], e["entity_id"], e["type"], m["text"])
        for e in annotation["entities"]
        for m in e["mentions"]
    ]


def predicted_mentions(annotation, predictions, threshold):
    """Predicted mentions above the threshold; exact spans map to reference entities for scoring."""
    exact, typed = reference_map(annotation)
    rows = []
    for prediction in predictions:
        if prediction["score"] < threshold:
            continue
        kind = prediction["type"] if "type" in prediction else ENTITY_PROMPTS[prediction["native_label"]]
        begin, end = int(prediction["begin"]), int(prediction["end"])
        rows.append(Mention(-float(prediction["score"]), begin, end, align(begin, end, kind, exact, typed), kind,
                            prediction["text"]))
    return rows


def predict_note(model, nlp, note_id, condition, ner):
    """Relations of one note pooled over its inputs; mentions are the reference ones when ner is None."""
    source, annotation = note(note_id)
    mentions = reference_mentions(annotation) if ner is None else predicted_mentions(annotation, ner, NER_THRESHOLD)
    best = {}
    for segment in segments(nlp, source["text"], source["segments"], condition):
        tokens, ner_spans, token_to_entity = glirel_input(nlp.make_doc(segment["text"]), segment, mentions)
        if len(ner_spans) < 2:
            continue
        for prediction in predict_relations(model, tokens, ner_spans):
            head = token_to_entity[tuple(prediction["head_pos"])]
            tail = token_to_entity[tuple(prediction["tail_pos"])]
            label = PROMPT_TO_RELATION[prediction["label"]]
            keep_best(best, (head, label, tail), {
                "head": head, "type": label, "tail": tail, "score": float(prediction["score"]),
                "head_text": " ".join(prediction.get("head_text") or []),
                "tail_text": " ".join(prediction.get("tail_text") or []), "segment_id": segment["segment_id"]})
    return {
        "note_id": note_id,
        "predicted_relations": sorted(best.values(), key=lambda row: (row["head"], row["type"], row["tail"])),
        "predicted_relation_keys": sorted(best),
        "gold_relation_keys": sorted({(r["head"], r["type"], r["tail"]) for r in annotation["relations"]}),
    }


def predict(args):
    """Relations of every note and input condition with one checkpoint, scored per condition."""
    from glirel import GLiREL

    ids = split(SEED)["test_ids"] if args.notes == "test" else note_ids()
    nlp = segmenter()
    model = GLiREL.from_pretrained(str(args.model), strict=True).to("cuda")
    for condition in CONDITIONS:
        ner = None
        if args.entities == "predicted":
            ner = {row["id"]: row["predictions"] for row in load_json(args.ner_dir / f"test_{condition}.json")["documents"]}
        documents = [predict_note(model, nlp, note_id, condition, ner and ner[note_id]) for note_id in ids]
        payload = {"condition": condition, "model": str(args.model), "entity_source": args.entities,
                   "documents": documents, "evaluation": score_relations(documents)}
        write_json(args.out_dir / f"glirel_{args.entities}_entities_{condition}.json", payload)
        print(condition, json.dumps(payload["evaluation"]), flush=True)


def training_examples(ids, nlp, condition="fixed256"):
    """Reference-mention training examples with multi-hot relation labels per directed mention pair."""
    examples, gold = [], set()
    for note_id in ids:
        source, annotation = note(note_id)
        mentions = reference_mentions(annotation)
        relations = {(r["head"], r["type"], r["tail"]) for r in annotation["relations"]}
        gold.update((note_id, *r) for r in relations)
        labels_by_pair = {}
        for head, label, tail in relations:
            labels_by_pair.setdefault((head, tail), []).append(label)
        for segment in segments(nlp, source["text"], source["segments"], condition):
            tokens, ner, mapping = glirel_input(nlp.make_doc(segment["text"]), segment, mentions)
            if len(ner) < 2:
                continue
            # GLiREL uses inclusive span ends internally; the shared scorer uses exclusive ends.
            pair_labels = {}
            for h, head in mapping.items():
                for t, tail in mapping.items():
                    if h != t and (labels := labels_by_pair.get((head, tail))):
                        pair_labels[(h[0], h[1] - 1, t[0], t[1] - 1)] = labels
            examples.append({"note_id": note_id, "tokenized_text": tokens, "ner": ner, "mapping": mapping,
                             "pair_labels": pair_labels})
    return examples, gold


def example_loss(model, example):
    import torch

    # Keep every ontology label available, including labels absent from this segment.
    batch = model.collate_fn(
        [{"tokenized_text": example["tokenized_text"], "ner": example["ner"]}],
        relation_types=list(RELATION_PROMPTS.values()),
        device=model.device,
    )
    logits, _, _, _ = model.compute_score(batch)
    targets = torch.zeros_like(logits)
    classes = batch["classes_to_id"][0]
    for j, pair in enumerate(batch["relations_idx"][0].tolist()):
        for label in example["pair_labels"].get((*pair[0], *pair[1]), []):
            targets[0, j, classes[RELATION_PROMPTS[label]] - 1] = 1
    # Checkpoint loss and positive/negative weighting with a multi-hot target, so several
    # annotated relation types on one pair are not overwritten.
    losses = model.compute_relation_loss(logits, targets)
    weights = targets * model.positive_weight + (1 - targets) * model.negative_weight
    return (losses * weights).mean()


def validation_scores(model, examples, gold):
    import torch

    model.eval()
    with torch.no_grad():
        predictions = {(example["note_id"], example["mapping"][tuple(p["head_pos"])], PROMPT_TO_RELATION[p["label"]],
                        example["mapping"][tuple(p["tail_pos"])])
                       for example in examples
                       for p in predict_relations(model, example["tokenized_text"], example["ner"])}
    return metric(gold, predictions)


def finetune(args):
    """Train on fixed256 inputs with reference mentions; keep the best pooled validation relation-F1 checkpoint."""
    import torch
    from glirel import GLiREL

    random.seed(SEED)
    torch.manual_seed(SEED)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    folds = split(SEED)
    nlp = segmenter()
    train, _ = training_examples(folds["train_ids"], nlp)
    valid, valid_gold = training_examples(folds["validation_ids"], nlp)
    model = GLiREL.from_pretrained(ZERO_SHOT, strict=True).to("cuda")
    model.base_config.fixed_relation_types = True
    encoder, other = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith("token_rep_layer.") else other).append(parameter)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": 1e-5}, {"params": other, "lr": 5e-5}], weight_decay=0.01)
    selection = EarlyStopping(PATIENCE)
    for _ in range(EPOCHS):
        model.train()
        random.shuffle(train)
        optimizer.zero_grad(set_to_none=True)
        for i, example in enumerate(train):
            loss = example_loss(model, example)
            (loss / min(4, len(train) - (i // 4) * 4)).backward()  # mean over the accumulation group
            if (i + 1) % 4 == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = validation_scores(model, valid, valid_gold)
        if selection.update(metrics["f1"], lambda: model.save_pretrained(out / "best")):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="Relations for every condition with one checkpoint.")
    p.add_argument("--model", default=ZERO_SHOT, help="Hugging Face name or fine-tuned checkpoint directory.")
    p.add_argument("--entities", choices=["gold", "predicted"], default="gold")
    p.add_argument("--notes", choices=["all", "test"], default="all")
    p.add_argument("--ner-dir", type=Path, default=CLINICAL_RUNS / "gliner_relex",
                   help="Fine-tuned GLiNER test_<condition>.json files.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=predict)
    p = sub.add_parser("finetune", help="Adapt GLiREL on the training notes and select on the validation notes.")
    p.add_argument("--output-dir", type=Path, default=CLINICAL_RUNS / "glirel_adapted")
    p.set_defaults(run=finetune)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
