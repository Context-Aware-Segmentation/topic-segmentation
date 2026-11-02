"""Segment sentence JSONL files with ModernBERT or Chonkie.

    python -m topic_segmentation.inference --input data/news/test.jsonl
    python -m topic_segmentation.inference --model-type chonkie_semantic --input data/news/test.jsonl

Input rows contain sentences and optional articleId and labels fields.
Output rows contain articleId, sentences, and predictions: 1 marks a boundary
after a sentence, including the final sentence. Labeled inputs also receive
precision, recall, F1, Pk, and WindowDiff scores.
"""
import argparse
import bisect
import itertools
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

from topic_segmentation.metrics import record, score_articles
from topic_segmentation.models import ModernBert
from topic_segmentation.paths import REPOSITORY

LOGS_DIR = REPOSITORY / "runs/long_document_inference"
WEIGHTS = REPOSITORY / "runs/mind_adaptation/mind_25k_full_news_ts"
CHONKIE = {"chonkie_semantic": REPOSITORY / "checkpoints/baselines/chonkie/semantic-potion-base-32m",
           "chonkie_neural": REPOSITORY / "checkpoints/baselines/chonkie/neural-modernbert-base"}


def unit_tokens(tokenizer, units):
    """Tokenize each unit with a leading BOS; return token IDs and BOS offsets."""
    token_ids, starts = [], []
    for unit in units:
        starts.append(len(token_ids))
        token_ids.append(tokenizer.bos_token_id)
        token_ids.extend(tokenizer.encode(unit, add_special_tokens=False))
    return token_ids, starts


def windows(starts, total, window, stride, protect_window_end):
    """Yield (start, end, scored_units), with unit positions relative to the prepended CLS.

    With end protection, keep the last unit as context and score it in the next window.
    A unit longer than a window is scored from its visible prefix.
    """
    start, first = 0, 0
    while first < len(starts):
        end = min(start + window - 1, total)
        units = [(unit, starts[unit] - start + 1)
                 for unit in itertools.takewhile(lambda unit: starts[unit] < end, range(first, len(starts)))]
        deferred = units[-1][0] if protect_window_end and end < total and len(units) > 1 else None
        yield start, end, units[:-1] if deferred is not None else units
        # A stride beyond the scored tokens would skip the units starting at the window end.
        following = next((unit for unit in range(units[0][0] + 1, len(starts))
                          if starts[unit] >= min(start + stride, end)), None)
        if deferred is not None:
            following = deferred if following is None else min(following, deferred)
        if following is None:
            return
        start, first = starts[following], following


def center_weight(rank, count):
    """Weight a unit by its rank in the window, with the highest weight at the center."""
    return max(0.01, min(rank + 1, count - rank) / max(1.0, count / 2.0))


class ModernBertSegmenter:
    """Segment articles with ModernBERT using weighted sliding windows."""

    def __init__(self, model_dir, device, window, stride, batch_size, protect_window_end):
        self.model = ModernBert.from_pretrained(model_dir, ignore_mismatched_sizes=True).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device, self.window, self.stride = device, window, stride
        self.batch_size, self.protect_window_end = batch_size, protect_window_end

    @torch.inference_mode()
    def add_margins(self, batch, margins, weights):
        """Accumulate center-weighted boundary logit margins for each unit."""
        length = max(len(ids) for _, ids, _ in batch)
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([ids + [pad] * (length - len(ids)) for _, ids, _ in batch], device=self.device)
        mask = torch.tensor([[1] * len(ids) + [0] * (length - len(ids)) for _, ids, _ in batch], device=self.device)
        logits = self.model.loss_calculator.classifier(self.model.dropout(self.model.encode(ids, mask))).float()
        differences = logits[:, :, 0] - logits[:, :, 1]
        for row, (index, _, units) in enumerate(batch):
            for rank, (unit, position) in enumerate(units):
                weight = center_weight(rank, len(units))
                margins[index][unit] += differences[row, position].item() * weight
                weights[index][unit] += weight

    def segment_batch(self, articles):
        """Segment articles by batching their token windows."""
        margins = [[0.0] * len(units) for units in articles]
        weights = [[0.0] * len(units) for units in articles]
        batch = []
        for index, units in enumerate(articles):
            token_ids, starts = unit_tokens(self.tokenizer, units)
            for start, end, scored in windows(starts, len(token_ids), self.window, self.stride, self.protect_window_end):
                batch.append((index, [self.tokenizer.cls_token_id] + token_ids[start:end], scored))
                if len(batch) == self.batch_size:
                    self.add_margins(batch, margins, weights)
                    batch = []
            if (index + 1) % 1000 == 0:
                print(f"  {index + 1}/{len(articles)} articles", flush=True)
        if batch:
            self.add_margins(batch, margins, weights)
        predictions = [[int(weight > 0 and margin > 0) for margin, weight in zip(*unit_scores)]
                       for unit_scores in zip(margins, weights)]
        for labels in predictions:
            if labels:
                labels[-1] = 1  # the article end is always a boundary
        return predictions


class Chonkie:
    """Map Chonkie chunk ends to sentence boundaries."""

    def __init__(self, model_type, model_dir, device):
        from chonkie import NeuralChunker, SemanticChunker
        from chonkie.embeddings import Model2VecEmbeddings
        from transformers import AutoModelForTokenClassification

        path = str(model_dir)
        if model_type == "chonkie_semantic":
            self.chunker = SemanticChunker(embedding_model=Model2VecEmbeddings(path), threshold=0.8)
        else:
            model = AutoModelForTokenClassification.from_pretrained(path, device_map={"": device})
            self.chunker = NeuralChunker(model=model,
                                         tokenizer=AutoTokenizer.from_pretrained(path), device_map=device, stride=512)

    def segment(self, sentences):
        if not sentences:
            return []
        ends = list(itertools.accumulate(len(sentence) + 1 for sentence in sentences))  # one space between sentences
        labels = [0] * len(sentences)
        text = " ".join(sentences)
        for chunk in self.chunker(text):
            end = len(text[:chunk.end_index].rstrip())  # SemanticChunker chunks keep the trailing space
            index = bisect.bisect_left(ends, end + 1)
            if index < len(labels):
                labels[index] = 1
        labels[-1] = 1
        return labels

    def segment_batch(self, articles):
        return [self.segment(sentences) for sentences in articles]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-type", default="modernbert_ts", choices=["modernbert_ts", *CHONKIE])
    parser.add_argument("--model", help="Weights directory; defaults to the final model or the Chonkie checkpoint")
    parser.add_argument("--input", required=True, help="JSONL with one article per line")
    parser.add_argument("--output", help="Predictions JSONL (default: runs/long_document_inference/<type>_<input>_<time>.jsonl)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window", type=int, default=1024, help="Tokens per window")
    parser.add_argument("--stride", type=int, default=512, help="Tokens between window starts")
    parser.add_argument("--batch-size", type=int, default=32, help="Windows per forward pass")
    parser.add_argument("--protect-window-end", action=argparse.BooleanOptionalAction, default=True,
                        help="Defer boundaries at a window end to the next window")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = Path(args.output or LOGS_DIR / f"{args.model_type}_{Path(args.input).stem}_{timestamp}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    default = WEIGHTS if args.model_type == "modernbert_ts" else CHONKIE[args.model_type]
    args.model = str(args.model or default)
    if args.model_type == "modernbert_ts":
        segmenter = ModernBertSegmenter(args.model, args.device, args.window, args.stride, args.batch_size,
                                        args.protect_window_end)
    else:
        segmenter = Chonkie(args.model_type, args.model, args.device)

    with open(args.input, encoding="utf-8") as stream:
        articles = [json.loads(line) for line in stream]
    scored = [(index, article) for index, article in enumerate(articles) if article.get("sentences")]
    predictions = segmenter.segment_batch([article["sentences"] for _, article in scored])
    with open(output, "w", encoding="utf-8") as stream:
        for (index, article), labels in zip(scored, predictions):
            stream.write(json.dumps({"articleId": article.get("articleId", str(index)),
                                     "sentences": article["sentences"], "predictions": labels}) + "\n")
    print(f"Wrote {output}")

    metrics = score_articles([article for _, article in scored], predictions)
    if metrics is not None:
        print(f"F1 = {metrics['f1']:.4f}  P = {metrics['precision']:.4f}  R = {metrics['recall']:.4f}  "
              f"Pk = {metrics['pk']:.4f}  WD = {metrics['wd']:.4f}  (n={metrics['n_scored']})")
    settings = {"model_type": args.model_type, "model": args.model, "input": args.input, "window": args.window,
                "stride": args.stride, "protect_window_end": args.protect_window_end, "n_articles": len(scored)}
    record(output.parent / "results.jsonl", output.stem, settings, metrics or {})


if __name__ == "__main__":
    main()
