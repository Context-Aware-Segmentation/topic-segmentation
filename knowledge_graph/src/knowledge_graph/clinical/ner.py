"""Zero-shot clinical NER: one of five released models on all 100 notes under the six input conditions.

python -m knowledge_graph.clinical.ner --model gliner_relex   # writes zero_shot_ner/gliner_relex.json
"""

import argparse
import json
from pathlib import Path

from knowledge_graph.clinical.data import CONDITIONS, CLINICAL_RUNS, HPO_OBO, deduplicate, documents, segmenter, segments
from knowledge_graph.clinical.schema import ENTITY_PROMPTS
from knowledge_graph.clinical.scoring import score_ner
from knowledge_graph.common import write_json

BATCH_SIZE = 16  # GLiNER texts and HunFlair2 sentences per batch


class PhenoBERT:
    def __init__(self):
        from knowledge_graph.clinical.phenobert import PhenoBERTBackend

        self.api = PhenoBERTBackend()

    def predict(self, document, segments):
        from knowledge_graph.clinical.phenobert import parse_predictions

        return [row for segment in segments
                for row in parse_predictions(self.api.annotate_text(segment["text"]), segment, document["text"])]


class FastHPOCR:
    def __init__(self):
        from FastHPOCR.HPOAnnotator import HPOAnnotator

        self.annotator = HPOAnnotator(str(HPO_OBO.with_name("hp.index")))

    def predict(self, document, segments):
        rows = []
        for segment in segments:
            for annotation in self.annotator.annotate(segment["text"], longestMatch=False):
                begin = segment["begin"] + int(annotation.getStartOffset())
                end = segment["begin"] + int(annotation.getEndOffset())
                rows.append({"begin": begin, "end": end, "text": document["text"][begin:end],
                             "hpo_label": str(annotation.getHPOLabel()), "concept_id": str(annotation.getHPOUri()),
                             "segment_id": segment["segment_id"]})
        return rows


class GLiNER:
    def __init__(self):
        from gliner import GLiNER
        from knowledge_graph.clinical.gliner_relex import RELEX

        self.model = GLiNER.from_pretrained(RELEX).to("cuda")

    def predict(self, document, segments):
        from knowledge_graph.clinical.gliner_relex import THRESHOLD

        batches = self.model.batch_predict_entities([segment["text"] for segment in segments], list(ENTITY_PROMPTS),
                                                    flat_ner=False, threshold=THRESHOLD, batch_size=BATCH_SIZE,
                                                    return_relations=False)
        rows = []
        for segment, predictions in zip(segments, batches):
            for prediction in predictions:
                begin, end = segment["begin"] + int(prediction["start"]), segment["begin"] + int(prediction["end"])
                rows.append({"begin": begin, "end": end, "text": document["text"][begin:end],
                             "native_label": prediction["label"], "score": float(prediction["score"]),
                             "segment_id": segment["segment_id"]})
        return rows


def hunflair_entities(sentence, text, offset):
    """Disease and Chemical mentions with document offsets."""
    rows = []
    for label in sentence.get_labels():
        if label.value in {"Disease", "Chemical"}:
            begin = offset + int(label.data_point.start_position)
            end = offset + int(label.data_point.end_position)
            rows.append({"begin": begin, "end": end, "text": text[begin:end], "native_label": label.value,
                         "score": float(label.score)})
    return rows


class HunFlair2:
    def __init__(self):
        import flair
        import torch
        from flair.nn import Classifier

        flair.device = torch.device("cuda")
        self.model = Classifier.load("hunflair2")

    def predict(self, document, segments):
        from flair.data import Sentence

        sentences = [Sentence(segment["text"]) for segment in segments]
        self.model.predict(sentences, mini_batch_size=BATCH_SIZE, verbose=False)
        return [{**row, "segment_id": segment["segment_id"]}
                for segment, sentence in zip(segments, sentences)
                for row in hunflair_entities(sentence, document["text"], segment["begin"])]


class MedCAT:
    def __init__(self):
        from knowledge_graph.clinical.medcat import PACK, load_pack

        self.cat = load_pack(PACK)

    def predict(self, document, segments):
        from knowledge_graph.clinical.medcat import clinical_entities

        return [row for segment in segments for row in clinical_entities(self.cat, segment)]


PREDICTORS = {"phenobert": PhenoBERT, "fasthpocr": FastHPOCR, "gliner_relex": GLiNER, "hunflair2": HunFlair2,
              "medcat": MedCAT}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=PREDICTORS, required=True)
    parser.add_argument("--out-dir", type=Path, default=CLINICAL_RUNS / "zero_shot_ner")
    args = parser.parse_args()
    notes = documents()
    nlp = segmenter()
    model = PREDICTORS[args.model]()
    results = {}
    for condition in CONDITIONS:
        predictions = {}
        for document in notes:
            inputs = segments(nlp, document["text"], document["reference_segments"], condition)
            predictions[document["id"]] = deduplicate(model.predict(document, inputs))
            print(args.model, condition, document["id"], len(predictions[document["id"]]), flush=True)
        results[condition] = {"metrics": score_ner(notes, predictions),
                              "documents": [{"id": key, "predictions": rows} for key, rows in predictions.items()]}
    write_json(args.out_dir / f"{args.model}.json", {"model": args.model, "conditions": results})
    print(json.dumps({key: value["metrics"] for key, value in results.items()}))


if __name__ == "__main__":
    main()
