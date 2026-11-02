"""Load ATLOP/DREEAM checkpoints and predict relations from DocRED inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from knowledge_graph.common import load_json
from knowledge_graph.news import atlop, dreeam
from knowledge_graph.news.docred import collate_fn, prepare_features
from knowledge_graph.news.encoding import encode_layers, encode_with_windows
from knowledge_graph.news.inputs import ENCODER, MAX_LENGTH

IMPLEMENTATIONS = {"atlop": atlop, "dreeam": dreeam}
LABELS_PER_PAIR = 4  # relation labels predicted at most for one entity pair


def build_model(
    implementation: str, tokenizer, classes: int, *, checkpoint: Path | None = None, single_pass: bool = False,
):
    """Initialize from RoBERTa for training, or restore a complete relation checkpoint on CPU."""
    if single_pass and checkpoint is None:
        raise ValueError("Single-pass inference requires a trained 2,048-subword checkpoint.")
    config = AutoConfig.from_pretrained(ENCODER, num_labels=classes)
    config.transformer_type = "roberta"
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    if single_pass:
        # Restore the learned position vectors from the checkpoint, including RoBERTa's padding offset.
        config.max_position_embeddings = 2048 + config.pad_token_id + 1
    if checkpoint is None:
        encoder = AutoModel.from_pretrained(ENCODER, config=config, attn_implementation="eager", torch_dtype=torch.float32)
    else:
        encoder = AutoModel.from_config(config, attn_implementation="eager", torch_dtype=torch.float32)
    model = IMPLEMENTATIONS[implementation].DocREModel(config, encoder, num_labels=LABELS_PER_PAIR)
    return load_checkpoint(model, checkpoint, torch.device("cpu")) if checkpoint is not None else model


def load_checkpoint(model, checkpoint: Path, device: torch.device):
    """Load FP32 weights; inference uses the same BF16 autocast as validation."""
    model.to(device=device, dtype=torch.float32)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model


def model_inputs(batch, device: torch.device) -> dict:
    """Move encoder tensors to the model device and retain entity and pair offsets."""
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "entity_pos": batch["entity_pos"],
        "hts": batch["hts"],
    }


def infer(model, features, batch_size: int, *, long_input_encoder=None) -> np.ndarray:
    device = next(model.parameters()).device
    loader = DataLoader(features, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(
                    **model_inputs(batch, device),
                    long_input_encoder=long_input_encoder,
                )
            scores = outputs["rel_pred"]
            predictions.append(scores.float().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def predicted_triples(predictions: np.ndarray, features: list[dict], id2rel: dict[int, str]) -> set[tuple]:
    output, row = set(), 0
    for feature in features:
        for head, tail in feature["hts"]:
            for relation_id in np.flatnonzero(predictions[row]):
                if relation_id:
                    output.add((feature["title"], head, id2rel[int(relation_id)], tail))
            row += 1
    return output


class DocREPredictor:
    """Predict relations from DocRED records with a trained ATLOP or DREEAM checkpoint."""

    def __init__(self, implementation: str, rel2id_path: Path, checkpoint: Path, *, single_pass: bool = False):
        rel2id = load_json(rel2id_path)
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.device = torch.device("cuda")
        self.single_pass = single_pass
        self.model = build_model(
            implementation, self.tokenizer, len(rel2id), checkpoint=checkpoint, single_pass=single_pass,
        ).to(self.device)
        self._id2rel = {value: key for key, value in rel2id.items()}

    def predict(self, inputs: list[dict], *, max_length: int = MAX_LENGTH) -> set[tuple]:
        """Return relation tuples; single-pass checkpoints also allow a 2,048-subword cap."""
        if not 2 < max_length <= (2048 if self.single_pass else MAX_LENGTH):
            raise ValueError("Input cap exceeds the selected encoding mode or leaves no room for content.")
        return self._predict(inputs, max_length, encode_layers if self.single_pass else None)

    def predict_with_windows(self, inputs: list[dict]) -> set[tuple]:
        """Predict on whole documents capped at 2,048 subwords using overlapping encoder windows."""
        if self.single_pass:
            raise ValueError("Use predict(max_length=2048) for a single-pass checkpoint.")
        return self._predict(inputs, 2048, encode_with_windows)

    def _predict(self, inputs: list[dict], max_length: int, encoder=None) -> set[tuple]:
        inputs = [document for document in inputs if len(document["vertexSet"]) >= 2]
        if not inputs:
            return set()
        features = prepare_features(inputs, self.tokenizer, max_length)
        array = infer(
            self.model, features, 1,
            long_input_encoder=encoder,
        )
        return predicted_triples(array, features, self._id2rel)
