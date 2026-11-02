"""PhenoBERT: official API loading and offset-preserving parsing, and fine-tuning of its three-way HPO phrase matcher.

python -m knowledge_graph.clinical.phenobert   # fine-tune on the training notes, score the test notes
"""

import argparse
import functools
import importlib
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from string import punctuation

from knowledge_graph.clinical.data import CONDITIONS, CORPUS, CLINICAL_RUNS, documents, segmenter, segments, split
from knowledge_graph.clinical.scoring import metric, score_ner
from knowledge_graph.common import EarlyStopping, load_json, write_json
from knowledge_graph.paths import DATA

# The released PhenoBERT pickles name their classes by top-level module (model, fastNLP), so this directory joins sys.path.
PHENOBERT = Path(__file__).resolve().parent / "vendor" / "phenobert"


@contextmanager
def full_unpickling():
    """torch.load with weights_only=False while PhenoBERT runs: its released models and the Stanza 1.1 resources it
    loads on demand are pickled objects, which PyTorch 2.6 refuses by default."""
    import torch

    load = torch.load
    torch.load = functools.partial(load, weights_only=False)
    try:
        yield
    finally:
        torch.load = load


class PhenoBERTBackend:
    """Own a PhenoBERT model and reuse its classifiers across predictions."""

    def __init__(self) -> None:
        import nltk
        import stanza
        from stanza.resources.common import DEFAULT_MODEL_DIR

        nltk.data.path.insert(0, str(DATA / "resources" / "nltk"))  # stopwords, wordnet, and the POS tagger
        if not (Path(DEFAULT_MODEL_DIR) / "en" / "ner" / "i2b2.pt").exists():  # Stanza 1.1 does not download by itself
            stanza.download("en", package="mimic", processors={"ner": "i2b2"}, verbose=False)
        if str(PHENOBERT) not in sys.path:
            sys.path.insert(0, str(PHENOBERT))
        with full_unpickling():
            self.api = importlib.import_module("api").PhenoBERTAPI()
        self.bert_model, self.hpo_tree = self.api.bert_model, self.api.hpo_tree

    def annotate_text(self, text: str) -> str:
        with full_unpickling():
            return self.api.annotate_text(text)

    def get_most_related_HPO_term(self, phrases: list[str]) -> list:
        with full_unpickling():
            return self.api.get_most_related_HPO_term(phrases)


def normalised_offset_map(text):
    """PhenoBERT's normalised text, which drops hyphens between an uppercase letter and a digit, and a map from its
    character boundaries to those of `text`."""
    chars, boundaries = [], [0]
    deleted = {match.start() for match in re.finditer(r"(?<=[A-Z])-(?=[\d])", text)}
    for index, char in enumerate(text):
        if index in deleted:
            boundaries[-1] = index + 1
            continue
        chars.append(char)
        boundaries.append(index + 1)
    return "".join(chars), boundaries


def parse_predictions(raw_output, segment, source_text):
    """Mentions of PhenoBERT's tab-separated output at source-note offsets."""
    normalized, boundary_map = normalised_offset_map(segment["text"])
    rows, remaining = [], raw_output
    while remaining:
        start, stop, remaining = remaining.split("\t", 2)
        local_begin, local_end = int(start), int(stop)
        # The mention text may contain tabs or newlines; its offsets give its length.
        trailer, _, remaining = remaining[local_end - local_begin + 1:].partition("\n")
        fields = trailer.split("\t")
        begin, end = segment["begin"] + boundary_map[local_begin], segment["begin"] + boundary_map[local_end]
        rows.append({"begin": begin, "end": end, "text": source_text[begin:end], "hpo_id": fields[0],
                     "score": float(fields[1]), "negative": len(fields) == 3, "segment_id": segment["segment_id"]})
    return rows


def pairs(ids, ontology):
    """Build phrase pairs; conflicting inputs prefer same concept (2), parent (1), then unrelated (0)."""
    rng = random.Random(42)
    keys = sorted(ontology.terms)
    ancestors = {}

    def related(a, b):
        for key in (a, b):
            if key not in ancestors:
                ancestors[key] = ontology.ancestors(key)
        return b in ancestors[a] or a in ancestors[b]

    output = set()
    for note_id in ids:
        for entity in load_json(CORPUS / "hpo_adjudication/review_final" / f"{note_id}.json")["symptom_entities"]:
            if entity["hpo_status"] != "linked":
                continue
            key = ontology.canonical_id(entity["hpo_id"])
            term = ontology.terms[key]
            for mention in entity["mentions"]:
                phrase = " ".join(mention["text"].split())
                output.add((phrase, term.label, 2))
                parents = [p for p in ontology.parents.get(key, []) if p in ontology.terms]
                if parents:
                    output.add((phrase, ontology.terms[parents[0]].label, 1))
                negative = rng.choice(keys)
                while negative == key or related(key, negative):
                    negative = rng.choice(keys)
                output.add((phrase, ontology.terms[negative].label, 0))
    # Match addWords: lowercase, replace ASCII punctuation, and split whitespace.
    spaces = str.maketrans({char: " " for char in punctuation})

    def input_key(row):
        fields = (row[0] + "::" + row[1]).lower().split("::")[:2]
        return tuple(tuple(field.translate(spaces).split()) for field in fields)

    targets = {}
    for row in output:
        key = input_key(row)
        targets[key] = max(targets.get(key, 0), row[2])
    return sorted(row for row in output if row[2] == targets[input_key(row)])


def dataset(rows):
    from fastNLP import DataSet
    from my_bert_match import addSeqlen, addWordPiece, addWords, processItem, processNum

    d = DataSet({"raw_words": [a + "::" + b for a, b, c in rows], "target": [c for a, b, c in rows]})
    for func, name in [
        (addWords, "p_words"),
        (addWordPiece, "t_words"),
        (processItem, "word_pieces"),
        (processNum, "word_nums"),
        (addSeqlen, "seq_len"),
    ]:
        d.apply(func, new_field_name=name)
    for k in ["word_pieces", "word_nums", "seq_len"]:
        d.field_arrays[k].is_input = True
    d.field_arrays["target"].is_target = True
    return d


def fit_matcher(api, train_rows, valid_rows, out):
    """Fine-tune on training pairs and restore the checkpoint selected by validation macro F1."""
    import torch

    from fastNLP import DataSetIter, RandomSampler  # the vendored copy is importable once PhenoBERTBackend is loaded
    from fastNLP.core.utils import _move_dict_value_to_device

    train = dataset(train_rows)
    valid = dataset(valid_rows)
    model = api.bert_model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    selection = EarlyStopping()
    for _ in range(10):
        model.train()
        for x, y in DataSetIter(batch_size=8, dataset=train, sampler=RandomSampler()):
            _move_dict_value_to_device(x, y, device=torch.device("cuda"))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model.forward(x["word_pieces"], x["word_nums"], x["seq_len"])["pred"], y["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        predicted, gold = [], []
        with torch.no_grad():
            for x, y in DataSetIter(batch_size=8, dataset=valid, sampler=None):
                _move_dict_value_to_device(x, y, device=torch.device("cuda"))
                predicted.extend(model.forward(x["word_pieces"], x["word_nums"], x["seq_len"])["pred"].argmax(-1).cpu().tolist())
                gold.extend(y["target"].cpu().tolist())
        score = sum(metric({i for i, v in enumerate(gold) if v == k}, {i for i, v in enumerate(predicted) if v == k})["f1"]
                    for k in range(3)) / 3
        if selection.update(score, lambda: torch.save(model.state_dict(), out / "best_state.pt")):
            break
    model.load_state_dict(torch.load(out / "best_state.pt"))
    model.eval()


def evaluate(api, ontology, ids, out):
    """NER and HPO-link scores of the test notes under every condition."""
    from knowledge_graph.clinical.hpo import gold_links

    docs = documents(ids)
    nlp = segmenter()
    links_gold = gold_links(ontology, ids)
    results = {}
    for condition in CONDITIONS:
        rows = {document["id"]: [row for segment in segments(nlp, document["text"], document["reference_segments"], condition)
                                 for row in parse_predictions(api.annotate_text(segment["text"]), segment, document["text"])]
                for document in docs}
        links = {(note_id, r["begin"], r["end"], hpo_id) for note_id, found in rows.items() for r in found
                 if (hpo_id := ontology.canonical_id(r.get("hpo_id")))}
        results[condition] = {"ner": score_ner(docs, rows), "el": metric(links_gold, links)}
        write_json(out / f"test_{condition}.json", {"metrics": results[condition],
                                                    "documents": [{"id": n, "predictions": p} for n, p in rows.items()]})
    write_json(out / "test_results.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=CLINICAL_RUNS / "phenobert")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate best_state.pt without training.")
    args = parser.parse_args()
    import numpy as np
    import torch

    from knowledge_graph.clinical.hpo import HpoOntology

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    folds = split()
    ontology = HpoOntology()
    api = PhenoBERTBackend()
    if args.evaluate_only:
        api.bert_model.load_state_dict(torch.load(out / "best_state.pt", map_location="cpu", weights_only=True))
        api.bert_model.eval()
    else:
        fit_matcher(api, pairs(folds["train_ids"], ontology), pairs(folds["validation_ids"], ontology), out)
    evaluate(api, ontology, folds["test_ids"], out)


if __name__ == "__main__":
    main()
