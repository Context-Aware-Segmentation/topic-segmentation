"""Score binary topic boundaries: 1 marks the end of a segment.

Pool precision, recall, and F1 over all units. Average Pk and WindowDiff across
documents using the convention of Yu et al. (2023). Exclude the document-final boundary.
"""
from datetime import datetime
import json

import numpy as np
import scipy.special
from segeval.window.pk import pk as segeval_pk
from segeval.window.windowdiff import window_diff as segeval_windowdiff
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from topic_segmentation.paths import REPOSITORY

PUNKT = str(REPOSITORY / "data/resources/nltk")  # the Punkt resource that split the news sentence data


def masses(labels):
    """Convert binary boundary labels to segment lengths."""
    lengths, count = [], 0
    for label in labels:
        count += 1
        if label == 1:
            lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return lengths


def window_error(measure, predictions, references):
    """Return mean window error as 1 - round(mean(1 - error), 4)."""
    agreements = []
    for prediction, reference in zip(predictions, references):
        if len(reference):
            hypothesis, gold = masses(prediction), masses(reference)
            assert sum(hypothesis) == sum(gold)
            agreements.append(1 - measure(hypothesis, gold))
    return 1 - round(float(np.mean(agreements)), 4)


def pk(predictions, references):
    return window_error(segeval_pk, predictions, references)


def windowdiff(predictions, references):
    return window_error(segeval_windowdiff, predictions, references)


def scores(predictions, references):
    """F1, precision, recall, Pk, and WindowDiff, rounded to 4 decimals."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        np.concatenate(references), np.concatenate(predictions), average="binary", zero_division=0)
    values = {"f1": f1, "precision": precision, "recall": recall,
              "pk": pk(predictions, references), "wd": windowdiff(predictions, references)}
    return {name: round(float(value), 4) for name, value in values.items()}


def trainer_metrics(boundary):
    """Build a Trainer callback for anchor-view boundary metrics and accuracy."""
    def compute(prediction):
        labels = prediction.label_ids[:, 0]
        keep = labels != -100
        predicted, gold = prediction.predictions.argmax(axis=-1)[keep], labels[keep]
        precision, recall, f1, _ = precision_recall_fscore_support(gold, predicted, average="binary",
                                                                   pos_label=boundary, zero_division=0)
        return {"overall_precision": float(precision), "overall_recall": float(recall), "overall_f1": float(f1),
                "overall_accuracy": float(accuracy_score(gold, predicted))}
    return compute


def score_documents(logits, labels, boundary=0):
    """Score document-level logits against class labels."""
    predictions = [(scipy.special.softmax(np.array(doc), axis=-1)[:, boundary] >= 0.5).astype(int) for doc in logits]
    references = [(np.array(doc) == boundary).astype(int) for doc in labels]
    return scores(predictions, references)


def score_articles(rows, predictions):
    """Score labeled articles; return None when none have internal boundaries to score."""
    pairs = [([int(label == 1) for label in row["labels"][:-1]], list(prediction)[:-1])
             for row, prediction in zip(rows, predictions) if "labels" in row and len(row["labels"]) > 1]
    if not pairs:
        return None
    references, predictions = zip(*pairs)
    return scores(predictions, references) | {"n_scored": sum(1 for reference in references if len(reference))}


def score_paragraphs(predicted, gold):
    """Score sentence predictions at paragraph ends using Punkt sentence counts.

    Inputs are prediction rows and paragraph-level gold documents.
    Missing predictions count as continuation (O).
    """
    assert len(predicted) == len(gold), f"{len(predicted)} predicted and {len(gold)} gold documents"
    import nltk
    if PUNKT not in nltk.data.path:
        nltk.data.path.insert(0, PUNKT)
    from nltk.tokenize import sent_tokenize

    predictions, references = [], []
    for row, document in zip(predicted, gold):
        paragraphs = document["sentences"][:-1]
        sentence_ends = []
        for text in paragraphs:
            count = len([s for s in sent_tokenize(text, language="english") if s.strip()])
            sentence_ends.append((sentence_ends[-1] if sentence_ends else 0) + max(1, count))
        labels = row["predictions"][:sentence_ends[-1] if sentence_ends else 0]
        predictions.append([int(end <= len(labels) and labels[end - 1] == "B-EOP") for end in sentence_ends])
        references.append(list(document["labels"][:len(paragraphs)]))
    return scores(predictions, references)


def record(path, run, settings, scores):
    """Append run settings and scores to a JSONL file."""
    row = {"time": datetime.now().isoformat(timespec="seconds"), "run": run, **settings, **scores}
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
