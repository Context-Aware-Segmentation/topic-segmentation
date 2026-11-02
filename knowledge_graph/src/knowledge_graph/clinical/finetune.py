"""Adapt GLiNER-Relex or HunFlair2 on the training notes, select on the validation notes, and score the test notes.

    python -m knowledge_graph.clinical.finetune --model gliner_relex   # writes runs/clinical/gliner_relex/
    python -m knowledge_graph.clinical.finetune --model hunflair2
    python -m knowledge_graph.clinical.finetune --model gliner_relex --evaluate-only
"""

import argparse
import random
from pathlib import Path

import torch

from knowledge_graph.clinical.data import CONDITIONS, CLINICAL_RUNS, note, reference_map, segmenter, split
from knowledge_graph.clinical.data import segments as inputs
from knowledge_graph.clinical.gliner_relex import RELEX, decode_joint, gold_keys, joint_predictions
from knowledge_graph.clinical.ner import hunflair_entities
from knowledge_graph.clinical.schema import ENTITY_PROMPTS, RELATION_PROMPTS
from knowledge_graph.clinical.scoring import metric
from knowledge_graph.common import EarlyStopping, write_json

HUNFLAIR_TYPES = {"CONDITION": "Disease", "SYMPTOM": "Disease", "DRUG": "Chemical"}
EPOCHS = 10  # maximum; early stopping on the validation notes


def segments(nlp, source, condition):
    return inputs(nlp, source["text"], source["segments"], condition)


def reference_keys(notes):
    """Reference mention spans and relations of the notes."""
    spans = {(s["note_id"], m["begin"], m["end"]) for s, a in notes for e in a["entities"] for m in e["mentions"]}
    relations = {(s["note_id"], r["head"], r["type"], r["tail"]) for s, a in notes for r in a["relations"]}
    return spans, relations


def gliner_examples(model, nlp, notes):
    """GLiNER-Relex training inputs: word-aligned reference entities and multi-label relations per fixed256 input."""
    examples = []
    prompts = {kind: prompt for prompt, kind in ENTITY_PROMPTS.items()}
    for source, annotation in notes:
        by_pair = {}
        for r in annotation["relations"]:
            by_pair.setdefault((r["head"], r["tail"]), set()).add(r["type"])
        for segment in segments(nlp, source, "fixed256"):
            words = list(model.data_processor.words_splitter(segment["text"]))
            if not words:
                continue
            entities, entity_ids, seen = [], [], set()
            for entity in annotation["entities"]:
                for m in entity["mentions"]:
                    if not segment["begin"] <= m["begin"] < m["end"] <= segment["end"]:
                        continue
                    begin, end = m["begin"] - segment["begin"], m["end"] - segment["begin"]
                    indices = [
                        i for i, (_, word_begin, word_end) in enumerate(words) if word_begin < end and word_end > begin
                    ]
                    if not indices or (indices[0], indices[-1]) in seen:
                        continue
                    first, last = indices[0], indices[-1]
                    seen.add((first, last))
                    if last - first + 1 > model.config.max_width:
                        continue
                    entities.append([first, last, prompts[entity["type"]]])
                    entity_ids.append(entity["entity_id"])
            relations = [
                [i, j, RELATION_PROMPTS[kind]]
                for i, head in enumerate(entity_ids)
                for j, tail in enumerate(entity_ids)
                if i != j
                for kind in sorted(by_pair.get((head, tail), set()))
            ]
            examples.append({"tokenized_text": [w[0] for w in words], "ner": entities, "relations": relations})
    return examples


def flair_examples(nlp, notes):
    """HunFlair2 training sentences: Disease/Chemical reference spans, longest first, without token overlap."""
    from flair.data import Sentence

    examples = []
    for source, annotation in notes:
        for segment in segments(nlp, source, "fixed256"):
            sentence = Sentence(segment["text"])
            candidates, used = [], set()
            for entity in annotation["entities"]:
                if entity["type"] not in HUNFLAIR_TYPES:
                    continue
                for m in entity["mentions"]:
                    if segment["begin"] <= m["begin"] < m["end"] <= segment["end"]:
                        begin, end = m["begin"] - segment["begin"], m["end"] - segment["begin"]
                        tokens = [
                            i for i, t in enumerate(sentence) if t.start_position < end and t.end_position > begin
                        ]
                        if tokens:
                            candidates.append((tokens, HUNFLAIR_TYPES[entity["type"]], begin, end))
            for tokens, label, begin, end in sorted(candidates, key=lambda x: (-(x[0][-1] - x[0][0]), x[0][0])):
                if used.intersection(tokens):
                    continue
                used.update(tokens)
                sentence[tokens[0] : tokens[-1] + 1].add_label("ner", label)
            examples.append(sentence)
    return examples


def gliner_predictions(model, source, annotation, segments):
    """Joint predictions for a note's inputs, using one reference map and one segment per model batch."""
    spans, typed = reference_map(annotation)
    rows, relations = [], set()
    for segment in segments:
        entity_batches, relation_batches = joint_predictions(model, [segment["text"]], batch_size=1)
        entities, found = decode_joint(segment, spans, typed, entity_batches[0], relation_batches[0])
        rows.extend({k: e[k] for k in ("begin", "end", "native_label", "text", "score")} for e in entities)
        relations.update(
            (source["note_id"], r["head"]["entity_id"], r["type"], r["tail"]["entity_id"]) for r in found
        )
    return rows, relations


def flair_predictions(model, source, segment):
    from flair.data import Sentence

    sentence = Sentence(segment["text"])
    model.predict(sentence, verbose=False)
    return hunflair_entities(sentence, source["text"], segment["begin"])


class GlinerRelex:
    """Joint GLiNER-Relex extraction, selected on the mean of validation NER and joint RE micro-F1."""

    joint = True

    def __init__(self, path=RELEX):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(path).to("cuda")
        self.collator = self.model._create_data_collator()

    def examples(self, nlp, notes):
        return gliner_examples(self.model, nlp, notes)

    def optimizer(self):
        encoder, other = [], []
        for name, parameter in self.model.named_parameters():
            (encoder if "token_rep_layer" in name else other).append(parameter)
        return torch.optim.AdamW([{"params": encoder, "lr": 1e-5}, {"params": other, "lr": 5e-5}], weight_decay=0.01)

    def loss(self, example):
        batch = self.collator(
            [example], entity_types=[list(ENTITY_PROMPTS)], relation_types=[list(RELATION_PROMPTS.values())]
        )
        return self.model(**{k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}).loss

    def release(self, example):
        pass

    def predict(self, source, annotation, segments):
        return gliner_predictions(self.model, source, annotation, segments)

    def save(self, out):
        self.model.save_pretrained(out / "best")

    @classmethod
    def best(cls, out):
        return cls(out / "best")


class HunFlair2:
    """HunFlair2: Disease and Chemical spans only; selected on validation
    exact-span NER micro-F1 over all six gold types."""

    joint = False

    def __init__(self, path="hunflair2"):
        import flair
        from flair.nn import Classifier

        flair.device = torch.device("cuda")
        self.model = Classifier.load(path)

    def examples(self, nlp, notes):
        return flair_examples(nlp, notes)

    def optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=5e-6, weight_decay=0.01)

    def loss(self, example):
        loss, count = self.model.forward_loss([example])
        return loss / max(count, 1)

    def release(self, example):
        example.clear_embeddings()  # Flair keeps the embeddings on the sentence

    def predict(self, source, annotation, segments):
        rows = [row for segment in segments for row in flair_predictions(self.model, source, segment)]
        return rows, set()

    def save(self, out):
        self.model.save(out / "best.pt")

    @classmethod
    def best(cls, out):
        return cls(out / "best.pt")


MODELS = {"gliner_relex": GlinerRelex, "hunflair2": HunFlair2}


@torch.no_grad()
def predict(system, nlp, notes, condition, path=None):
    """Exact-span NER (and joint RE for GLiNER-Relex) of the notes under one condition."""
    system.model.eval()
    spans, relations, output = set(), set(), []
    for source, annotation in notes:
        rows, note_relations = system.predict(source, annotation, segments(nlp, source, condition))
        spans.update((source["note_id"], row["begin"], row["end"]) for row in rows)
        relations |= note_relations
        record = {"id": source["note_id"], "predictions": rows}
        if system.joint:
            record.update(
                note_id=source["note_id"],
                predicted_relation_keys=sorted(r[1:] for r in note_relations),
                gold_relation_keys=gold_keys(annotation),
            )
        output.append(record)
    reference_spans, reference_relations = reference_keys(notes)
    result = {"ner": metric(reference_spans, spans)}
    if system.joint:
        result["re"] = metric(reference_relations, relations)
    if path:
        write_json(path, {"metrics": result, "documents": output})
    return result


def train(system, examples, valid, nlp, out):
    """Train on fixed256 inputs with early stopping on validation."""
    optimizer = system.optimizer()
    selection = EarlyStopping()
    for _ in range(EPOCHS):
        random.shuffle(examples)
        system.model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, example in enumerate(examples):
            loss = system.loss(example)
            (loss / min(4, len(examples) - (i // 4) * 4)).backward()  # mean over the accumulation group
            if (i + 1) % 4 == 0 or i + 1 == len(examples):
                torch.nn.utils.clip_grad_norm_(system.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            system.release(example)
        metrics = predict(system, nlp, valid, "fixed256")
        score = sum(v["f1"] for v in metrics.values()) / len(metrics)
        if selection.update(score, lambda: system.save(out)):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--out", type=Path, help="Output directory (default: runs/clinical/MODEL)")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the saved best checkpoint without training.")
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    out = args.out or CLINICAL_RUNS / args.model
    out.mkdir(parents=True, exist_ok=True)
    nlp = segmenter()
    folds = split()
    if not args.evaluate_only:
        train_notes = [note(n) for n in folds["train_ids"]]
        valid = [note(n) for n in folds["validation_ids"]]
        system = MODELS[args.model]()
        train(system, system.examples(nlp, train_notes), valid, nlp, out)
        del system
        torch.cuda.empty_cache()
    best = MODELS[args.model].best(out)
    test = [note(n) for n in folds["test_ids"]]
    results = {condition: predict(best, nlp, test, condition, out / f"test_{condition}.json") for condition in CONDITIONS}
    write_json(out / "test_results.json", results)


if __name__ == "__main__":
    main()
