"""JSON and JSONL I/O, precision/recall/F1, early stopping, and text normalization helpers."""

import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj, *, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def count_metrics(tp, gold, predicted):
    """Counts, tp/fp/fn, and precision/recall/f1 from the true-positive, gold, and predicted counts."""
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"gold": gold, "predicted": predicted, "tp": tp, "fp": predicted - tp, "fn": gold - tp,
            "precision": precision, "recall": recall, "f1": f1}


def precision_recall_f1(gold, predicted):
    """Set sizes, tp/fp/fn counts, and precision/recall/f1 of two sets."""
    return count_metrics(len(gold & predicted), len(gold), len(predicted))


def normalize_space(text):
    return " ".join(str(text or "").split())


def normalize_label(text):
    return normalize_space(text).replace("_", " ").lower()


class EarlyStopping:
    """Keep the checkpoint with the best validation score; stop after `patience` epochs without improvement
    (never when patience is None)."""

    def __init__(self, patience=3):
        self.patience, self.best, self.stale = patience, -1.0, 0

    def update(self, score, save):
        """Call save() when the score improves; return True once patience is exhausted."""
        if score > self.best:
            self.best, self.stale = score, 0
            save()
        else:
            self.stale += 1
        return self.patience is not None and self.stale >= self.patience
