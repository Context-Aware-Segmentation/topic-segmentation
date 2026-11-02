# Adapted from fastNLP (Apache-2.0) for PhenoBERT; see THIRD_PARTY_NOTICES.md.

__all__ = [
    "BertForSequenceClassification",
    "BertForSentenceMatching",
    "BertForMultipleChoice",
    "BertForTokenClassification",
    "BertForQuestionAnswering"
]

import warnings

import torch
from torch import nn

from .base_model import BaseModel
from ..core._logger import logger
from ..core.const import Const
from ..embeddings.My_bert_embedding import BertEmbedding


class BertForSequenceClassification(BaseModel):
    """
    BERT model for classification.

    """
    def __init__(self, embed: BertEmbedding, num_labels: int=2, dropout=0.1):
        super(BertForSequenceClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for sequence classification excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, num_labels]
        """
        hidden = self.dropout(self.bert(words))
        cls_hidden = hidden[:, 0]
        logits = self.classifier(cls_hidden)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForSentenceMatching(BaseModel):
    """
    BERT model for sentence matching.

    """
    def __init__(self, embed: BertEmbedding, num_labels: int=2, dropout=0.1):
        super(BertForSentenceMatching, self).__init__()
        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for sentence matching excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, *words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, num_labels]
        """
        hidden = self.bert(words)
        cls_hidden = self.dropout(hidden[:, 0])
        logits = self.classifier(cls_hidden)

        return {Const.OUTPUT: logits}

    def predict(self, *words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(*words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForMultipleChoice(BaseModel):
    """
    BERT model for multiple choice.

    """
    def __init__(self, embed: BertEmbedding, num_choices=2, dropout=0.1):
        super(BertForMultipleChoice, self).__init__()

        self.num_choices = num_choices
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, 1)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for multiple choice excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, num_choices, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size, num_choices]
        """
        batch_size, num_choices, seq_len = words.size()

        input_ids = words.view(batch_size * num_choices, seq_len)
        hidden = self.bert(input_ids)
        pooled_output = self.dropout(hidden[:, 0])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        return {Const.OUTPUT: reshaped_logits}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, num_choices, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForTokenClassification(BaseModel):
    """
    BERT model for token classification.

    """
    def __init__(self, embed: BertEmbedding, num_labels, dropout=0.1):
        super(BertForTokenClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = False
            warn_msg = "Bert for token classification excepts BertEmbedding `include_cls_sep` False, " \
                       "but got True. FastNLP has changed it to False."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, seq_len, num_labels]
        """
        sequence_output = self.bert(words)  # [batch_size, seq_len, embed_dim]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size, seq_len]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForQuestionAnswering(BaseModel):
    def __init__(self, embed: BertEmbedding):
        super(BertForQuestionAnswering, self).__init__()

        self.bert = embed
        self.qa_outputs = nn.Linear(self.bert.embedding_dim, 2)

    def forward(self, words):
        sequence_output = self.bert(words)
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, num_labels]

        return {'pred_start': logits[:, :, 0], 'pred_end': logits[:, :, 1]}

    def predict(self, words):
        return self.forward(words)
