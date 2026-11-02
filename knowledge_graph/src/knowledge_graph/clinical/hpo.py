"""HPO ontology, and the comparison of three HPO linkers on the SYMPTOM mentions of the ten test notes.

python -m knowledge_graph.clinical.hpo prepare     # mentions to link and the gold links
python -m knowledge_graph.clinical.hpo fasthpocr   # FastHPOCR longest match inside each mention
python -m knowledge_graph.clinical.hpo phenobert   # PhenoBERT dictionary, then its fine-tuned phrase matcher
python -m knowledge_graph.clinical.hpo sapbert     # SapBERT nearest alias, threshold chosen on validation
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from knowledge_graph.clinical.data import CONDITIONS, CORPUS, HPO_OBO, CLINICAL_RUNS, split
from knowledge_graph.clinical.scoring import metric
from knowledge_graph.common import load_json, normalize_space, write_json

OUT = CLINICAL_RUNS / "hpo_linker_comparison"
SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def normalize_surface(value: str) -> str:
    """Lowercase, with punctuation and underscores turned into single spaces."""
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold().replace("_", " ")).split())


@dataclass
class Term:
    label: str
    synonyms: list[str]


class HpoOntology:
    """The terms of the HPO OBO file."""

    def __init__(self, path: Path = HPO_OBO):
        self.path = path
        self.version = "unknown"
        self.names = {}  # current id -> name
        self.parents = {}  # id -> is_a parent ids, including obsolete terms
        self.terms = {}  # id -> Term, current terms only
        self.alternate = {}  # alternate id -> current id
        header, *blocks = path.read_text(encoding="utf-8").split("\n[Term]\n")
        for line in header.splitlines():
            if line.startswith("data-version:"):
                self.version = line.split(":", 1)[1].strip()
        for block in blocks:
            fields = {}
            for line in block.splitlines():
                if ": " in line:
                    key, value = line.split(": ", 1)
                    fields.setdefault(key, []).append(value)
            if "id" not in fields:
                continue
            term_id = fields["id"][0]
            self.parents[term_id] = [value.split()[0] for value in fields.get("is_a", [])]
            if "name" not in fields:
                continue
            if "true" in fields.get("is_obsolete", []):
                continue
            self.names[term_id] = fields["name"][0]
            synonyms = [line.split('"')[1] for line in fields.get("synonym", []) if '"' in line]
            self.terms[term_id] = Term(fields["name"][0], synonyms)
            for alternate_id in fields.get("alt_id", []):
                self.alternate[alternate_id] = term_id

    def canonical_id(self, value: str | None) -> str | None:
        """The current term of an ID or alternate ID, or None."""
        value = self.alternate.get(value, value)
        return value if value in self.terms else None

    def ancestors(self, term_id: str) -> set[str]:
        seen, todo = set(), list(self.parents.get(term_id, []))
        while todo:
            parent = todo.pop()
            if parent not in seen:
                seen.add(parent)
                todo.extend(self.parents.get(parent, []))
        return seen


def gold_links(ontology, ids):
    """Reference HPO links of SYMPTOM mentions: (note, begin, end, HPO ID)."""
    return {(note_id, m["begin"], m["end"], ontology.canonical_id(entity["hpo_id"]))
            for note_id in ids
            for entity in load_json(CORPUS / "hpo_adjudication/review_final" / f"{note_id}.json")["symptom_entities"]
            if entity["hpo_status"] == "linked"
            for m in entity["mentions"]}


def mention_record(note_id, text, begin, end):
    return {"note_id": note_id, "begin": begin, "end": end, "text": normalize_space(text[begin:end])}


def unique_spans(mentions):
    """One mention per span (the last one wins)."""
    return list({(m["note_id"], m["begin"], m["end"]): m for m in mentions}.values())


def prepare(out):
    """Mentions to link: all reference SYMPTOM mentions, and the symptom predictions of GLiNER-Relex and HunFlair2."""
    ontology, folds = HpoOntology(), split()
    data = {"gold": {}, "inputs": {}}
    for fold in ["validation", "test"]:
        ids = folds[fold + "_ids"]
        reference = []
        for note_id in ids:
            text = load_json(CORPUS / "sources" / f"{note_id}.json")["text"]
            for entity in load_json(CORPUS / "reviews" / f"{note_id}.json")["entities"]:
                if entity["type"] == "SYMPTOM":
                    reference += [mention_record(note_id, text, m["begin"], m["end"]) for m in entity["mentions"]]
        data["gold"][fold] = sorted(gold_links(ontology, ids))
        data["inputs"][fold] = {"gt": {"reference": unique_spans(reference)}}
    for model, label in {"gliner_relex": "clinical sign or symptom", "hunflair2": "disease"}.items():
        data["inputs"]["test"][model] = {}
        for condition in CONDITIONS:
            rows = []
            for document in load_json(CLINICAL_RUNS / model / f"test_{condition}.json")["documents"]:
                text = load_json(CORPUS / "sources" / f"{document['id']}.json")["text"]
                rows += [mention_record(document["id"], text, p["begin"], p["end"])
                         for p in document["predictions"] if p["native_label"].lower() == label]
            data["inputs"]["test"][model][condition] = unique_spans(rows)
    data["phrases"] = sorted({m["text"] for fold in data["inputs"].values() for runs in fold.values()
                              for rows in runs.values() for m in rows})
    write_json(out / "inputs.json", data)


def score(out, name, cache, threshold=-1):
    """Test metrics and predicted links of every detector and input condition, from the phrase links in `cache`."""
    data = load_json(out / "inputs.json")
    gold = set(map(tuple, data["gold"]["test"]))
    results, predictions = {}, {}
    for detector, inputs in data["inputs"]["test"].items():
        results[detector], predictions[detector] = {}, {}
        for condition, mentions in inputs.items():
            found = {(m["note_id"], m["begin"], m["end"], link["hpo_id"])
                     for m in mentions for link in cache[m["text"]] if link.get("score", 1) >= threshold}
            results[detector][condition] = metric(gold, found)
            predictions[detector][condition] = sorted(found)
    write_json(out / f"{name}_results.json", {"threshold": threshold, "metrics": results})
    write_json(out / f"{name}_predictions.json", predictions)
    print(name, json.dumps(results), flush=True)


def fasthpocr(out):
    from FastHPOCR.HPOAnnotator import HPOAnnotator

    ontology = HpoOntology()
    annotator = HPOAnnotator(str(ontology.path.with_name("hp.index")))
    cache = {}
    for phrase in load_json(out / "inputs.json")["phrases"]:
        matches = annotator.annotate(phrase, longestMatch=True)
        length = lambda match: int(match.getEndOffset()) - int(match.getStartOffset())
        longest = max(map(length, matches), default=0)
        ids = {ontology.canonical_id(str(match.getHPOUri())) for match in matches if length(match) == longest}
        cache[phrase] = [{"hpo_id": hpo_id, "score": 1.0} for hpo_id in sorted(ids - {None})]
    score(out, "fasthpocr", cache)


def phenobert(out):
    """PhenoBERT's exact dictionary match first, then its fine-tuned phrase matcher."""
    import torch

    from knowledge_graph.clinical.phenobert import PhenoBERTBackend

    torch.set_num_threads(4)
    ontology = HpoOntology()
    api = PhenoBERTBackend()
    api.bert_model.load_state_dict(torch.load(CLINICAL_RUNS / "phenobert/best_state.pt", map_location="cpu"))
    api.bert_model.eval()
    phrases = load_json(out / "inputs.json")["phrases"]
    as_link = lambda hpo_id: [{"hpo_id": hpo_id, "score": 1.0}] if hpo_id else []
    cache = {}
    with torch.no_grad():
        for start in range(0, len(phrases), 32):
            queries = {phrase: " ".join(phrase.split()[:30]) for phrase in phrases[start:start + 32]}
            unmatched = []
            for phrase, query in queries.items():
                hpo_id = api.hpo_tree.matchPhrase2HPO(query)
                if hpo_id and hpo_id in api.hpo_tree.phenotypic_abnormalityNT:
                    cache[phrase] = as_link(ontology.canonical_id(hpo_id))
                else:
                    unmatched.append(phrase)
            if unmatched:
                answers = dict(api.get_most_related_HPO_term([queries[p] for p in unmatched]))
                for phrase in unmatched:
                    cache[phrase] = as_link(ontology.canonical_id(answers.get(queries[phrase])))
    score(out, "phenobert", cache)


def sapbert(out):
    """SapBERT's nearest HPO alias, with the score threshold chosen on the validation reference mentions."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    ontology = HpoOntology()
    data = load_json(out / "inputs.json")
    tokenizer = AutoTokenizer.from_pretrained(SAPBERT)
    model = AutoModel.from_pretrained(SAPBERT).eval()
    aliases = sorted({(term_id, alias) for term_id, term in ontology.terms.items()
                      for alias in [term.label, *term.synonyms] if normalize_space(alias)})

    @torch.inference_mode()
    def encode(strings):
        """Normalized [CLS] embeddings, computed in batches of strings with similar length."""
        lengths = [len(ids) for ids in tokenizer(strings, truncation=True, max_length=64)["input_ids"]]
        order = sorted(range(len(strings)), key=lambda i: lengths[i])
        output = torch.empty((len(strings), model.config.hidden_size))
        for start in range(0, len(order), 64):
            ids = order[start:start + 64]
            batch = tokenizer([strings[i] for i in ids], padding=True, truncation=True, max_length=64, return_tensors="pt")
            output[ids] = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0, :], dim=1)
        return output

    scores, best = (encode(data["phrases"]) @ encode([alias for _, alias in aliases]).T).max(dim=1)
    cache = {phrase: [{"hpo_id": aliases[i][0], "score": value}]
             for phrase, value, i in zip(data["phrases"], scores.tolist(), best.tolist())}
    gold = set(map(tuple, data["gold"]["validation"]))
    sweep = []
    for threshold in np.linspace(0, 1, 101):
        found = {(m["note_id"], m["begin"], m["end"], cache[m["text"]][0]["hpo_id"])
                 for m in data["inputs"]["validation"]["gt"]["reference"] if cache[m["text"]][0]["score"] >= threshold}
        sweep.append((metric(gold, found)["f1"], -float(threshold)))
    threshold = -max(sweep)[1]  # the best validation F1; the lower threshold wins ties
    score(out, "sapbert", cache, threshold)
    score(out, "sapbert_unthresholded", cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "fasthpocr", "phenobert", "sapbert"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    {"prepare": prepare, "fasthpocr": fasthpocr, "phenobert": phenobert, "sapbert": sapbert}[args.step](args.out)


if __name__ == "__main__":
    main()
