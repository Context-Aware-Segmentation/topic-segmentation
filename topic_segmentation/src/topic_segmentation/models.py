"""Topic-segmentation models of Yu et al. (2023) on BERT, RoBERTa, Longformer, and ModernBERT.

Attribute names (`bert`, `roberta`, `longformer`, `model`, `dropout`, `loss_calculator`)
match the state-dict keys of the trained checkpoints.
"""
import torch
import torch.nn as nn
from transformers import (BertModel, BertPreTrainedModel, LongformerModel, LongformerPreTrainedModel,
                          ModernBertModel, ModernBertPreTrainedModel, RobertaModel, RobertaPreTrainedModel)

from topic_segmentation.objectives import CombinedObjective


class Segmentation:
    """Shared forward pass for backbones implementing encode()."""

    def heads(self, dropout):
        self.dropout = nn.Dropout(dropout)
        self.loss_calculator = CombinedObjective(self.config)
        self.post_init()

    # Trainer keeps only the dataset columns named here and returns `labels` with the predictions.
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, extract_eop_segment_ids=None,
                eop_index_for_aggregate_batch_eop_features=None, sent_pair_orders=None, sent_token_mask=None):
        """Return loss and anchor logits from inputs ordered as [anchor, augmented]."""
        anchor = self.dropout(self.encode(input_ids[:, 0], attention_mask[:, 0], token_type_ids[:, 0]))
        loss, logits = self.loss_calculator(
            anchor, labels[:, 0], extract_eop_segment_ids[:, 0], eop_index_for_aggregate_batch_eop_features[:, 0])
        if self.config.do_da_ts or self.config.do_tssp:
            augmented = self.dropout(self.encode(input_ids[:, 1], attention_mask[:, 1], token_type_ids[:, 1]))
            loss = loss + self.loss_calculator(augmented, labels[:, 1], sent_token_mask=sent_token_mask[:, 1],
                                               sent_pair_orders=sent_pair_orders[:, 1], augmented=True)[0]
        return loss, logits


class Bert(Segmentation, BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.heads(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        return self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]


class Roberta(Segmentation, RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.heads(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        return self.roberta(input_ids, attention_mask=attention_mask)[0]


class Longformer(Segmentation, LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.heads(config.hidden_dropout_prob)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        cls = torch.zeros_like(input_ids)
        cls[:, 0] = 1  # global attention on the CLS token
        return self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=cls,
                               token_type_ids=token_type_ids)[0]


class ModernBert(Segmentation, ModernBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config._attn_implementation = "eager"  # sdpa yields NaN on heavily padded sentence-level batches
        self.model = ModernBertModel(config)
        self.heads(config.classifier_dropout)

    def _init_weights(self, module):
        # ModernBERT skips plain Linear heads; initialize these with std=0.02.
        if module in (self.loss_calculator.classifier, self.loss_calculator.tssp.classifier):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        else:
            super()._init_weights(module)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
