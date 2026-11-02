"""Metric sanity, a tiny random ModernBERT forward pass, and the training command line, offline on CPU."""
from dataclasses import asdict
import subprocess
import sys

import torch
from transformers import ModernBertConfig

from topic_segmentation import metrics, models
from topic_segmentation.training import ModelArguments


def test_perfect_predictions_score_perfectly():
    labels = [[0, 1, 0, 0, 1, 0, 0, 1]]
    assert metrics.scores(labels, labels) == {"f1": 1.0, "precision": 1.0, "recall": 1.0, "pk": 0.0, "wd": 0.0}


def test_tiny_modernbert_forward():
    config = ModernBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                              max_position_embeddings=64, local_attention=16, pad_token_id=0, cls_token_id=1, sep_token_id=2,
                              reference_compile=False, num_labels=2, num_tssp_labels=3, **asdict(ModelArguments(model_name_or_path="")))
    torch.manual_seed(42)
    model = models.ModernBert(config).eval()
    with torch.inference_mode():
        logits = model.loss_calculator.classifier(model.encode(torch.tensor([[1, 3, 4, 5, 2]]), torch.ones(1, 5, dtype=torch.long)))
    assert logits.shape == (1, 5, 2) and torch.isfinite(logits).all()


def test_training_command_line():
    usage = subprocess.run([sys.executable, "-m", "topic_segmentation.training", "--help"], capture_output=True, text=True, check=True)
    assert "--model_name_or_path" in usage.stdout
