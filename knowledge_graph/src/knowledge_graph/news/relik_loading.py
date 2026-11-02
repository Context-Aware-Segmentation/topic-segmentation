# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# PoolerEndLogits is adapted from Transformers 4.41 (Apache-2.0); see THIRD_PARTY_NOTICES.md.
"""Load the pinned ReLiK entity linker under the locked Transformers version."""

from contextlib import contextmanager

RELIK = "sapienzanlp/relik-entity-linking-large"


def patch_pooler():
    """Restore Transformers 4.41's PoolerEndLogits (same parameter names), which the pinned ReLiK reader imports."""
    import torch
    from torch import nn
    from transformers import modeling_utils

    class PoolerEndLogits(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.activation = nn.Tanh()
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dense_1 = nn.Linear(config.hidden_size, 1)

        def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
            if start_states is None and start_positions is None:
                raise AssertionError("One of start_states, start_positions should be not None")
            if start_positions is not None:
                length, hidden_size = hidden_states.shape[-2:]
                positions = start_positions[:, None, None].expand(-1, -1, hidden_size)
                start_states = hidden_states.gather(-2, positions).expand(-1, length, -1)
            logits = self.dense_1(
                self.LayerNorm(self.activation(self.dense_0(torch.cat([hidden_states, start_states], dim=-1))))
            ).squeeze(-1)
            if p_mask is not None:
                penalty = 65500 if self.dense_0.weight.dtype == torch.float16 else 1e30
                logits = logits * (1 - p_mask) - penalty * p_mask
            return logits

    modeling_utils.PoolerEndLogits = PoolerEndLogits


@contextmanager
def no_mean_resizing():
    """Load the ReLiK reader without mean-initializing new embeddings (they live on a meta tensor at that point)."""
    from transformers import PreTrainedModel

    original = PreTrainedModel.resize_token_embeddings

    def resize(self, *args, **kwargs):
        kwargs.setdefault("mean_resizing", False)
        return original(self, *args, **kwargs)

    PreTrainedModel.resize_token_embeddings = resize
    try:
        yield
    finally:
        PreTrainedModel.resize_token_embeddings = original


def load_relik(index=None):
    """ReLiK on CUDA in FP16 with its own retrieval index, or with ``index``."""
    patch_pooler()
    from relik import Relik

    kwargs = {"device": "cuda", "index_device": "cuda", "precision": "fp16", "use_faiss": False}
    if index:
        kwargs["index"] = index
    with no_mean_resizing():
        return Relik.from_pretrained(RELIK, **kwargs)
