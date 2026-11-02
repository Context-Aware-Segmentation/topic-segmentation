"""The new classification head of boundary pre-training is fixed by the seed."""
import torch
from transformers import ModernBertConfig, ModernBertModel

from topic_segmentation.boundary_pretraining import load_model


def tiny_backbone(path):
    config = ModernBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                              num_attention_heads=4, max_position_embeddings=64, local_attention=16, pad_token_id=0,
                              cls_token_id=1, sep_token_id=2, reference_compile=False)
    ModernBertModel(config).save_pretrained(path)
    return path


def head(model):
    return model.classifier.weight.detach().clone()


def test_same_seed_gives_the_same_head(tmp_path):
    base = tiny_backbone(tmp_path / "base")
    first = head(load_model(base, seed=42, attention="eager"))
    torch.manual_seed(0)  # state left behind by earlier work must not matter
    assert torch.equal(first, head(load_model(base, seed=42, attention="eager")))


def test_seed_changes_the_head(tmp_path):
    base = tiny_backbone(tmp_path / "base")
    assert not torch.equal(head(load_model(base, seed=42, attention="eager")),
                           head(load_model(base, seed=43, attention="eager")))
