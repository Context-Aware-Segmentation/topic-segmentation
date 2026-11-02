# Adapted from fastNLP (Apache-2.0) for PhenoBERT; see THIRD_PARTY_NOTICES.md.
"""
.. todo::
    doc
"""

__all__ = [
    "BertEmbedding",
    "BertWordPieceEncoder"
]

import collections
import warnings
from itertools import chain

import numpy as np
import torch
from torch import nn

from .contextual_embedding import ContextualEmbedding
from ..core import logger
from ..core.vocabulary import Vocabulary
from ..io.file_utils import PRETRAINED_BERT_MODEL_DIR
from ..modules.encoder.bert import _WordPieceBertModel, BertModel, BertTokenizer


class BertEmbedding(ContextualEmbedding):
    
    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en-base-uncased', layers: str = '-1',
                 pool_method: str = 'first', word_dropout=0, dropout=0, include_cls_sep: bool = False,
                 pooled_cls=True, requires_grad: bool = True, auto_truncate: bool = False):
        super(BertEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)

        if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
            if 'cn' in model_dir_or_name.lower() and pool_method not in ('first', 'last'):
                logger.warning("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                               " faster speed.")
                warnings.warn("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                              " faster speed.")
        
        self._word_sep_index = None
        # if '[SEP]' in vocab:
        #     self._word_sep_index = vocab['[SEP]']
        
        self.model = _WordBertModel(model_dir_or_name=model_dir_or_name, vocab=vocab, layers=layers,
                                    pool_method=pool_method, include_cls_sep=include_cls_sep,
                                    pooled_cls=pooled_cls, auto_truncate=auto_truncate, min_freq=2)
        
        self.requires_grad = requires_grad
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size
    
    def _delete_model_weights(self):
        del self.model
    
    def forward(self, words):
        words_ = self.drop_word(words[0])
        outputs = self._get_sent_reprs(words_)
        if outputs is not None:
            return self.dropout(outputs)
        outputs = self.model(words)
        outputs = torch.cat([*outputs], dim=-1)

        return self.dropout(outputs)
    
    def drop_word(self, words):
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                if self._word_sep_index:
                    sep_mask = words.eq(self._word_sep_index)
                mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)
                pad_mask = words.ne(0)
                mask = pad_mask.__and__(mask)
                words = words.masked_fill(mask, self._word_unk_index)
                if self._word_sep_index:
                    words.masked_fill_(sep_mask, self._word_sep_index)
        return words


class BertWordPieceEncoder(nn.Module):
    
    def __init__(self, model_dir_or_name: str = 'en-base-uncased', layers: str = '-1', pooled_cls: bool = False,
                 word_dropout=0, dropout=0, requires_grad: bool = True):
        super().__init__()
        
        self.model = _WordPieceBertModel(model_dir_or_name=model_dir_or_name, layers=layers, pooled_cls=pooled_cls)
        self._sep_index = self.model._sep_index
        self._wordpiece_pad_index = self.model._wordpiece_pad_index
        self._wordpiece_unk_index = self.model._wordpiece_unknown_index
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size
        self.requires_grad = requires_grad
        self.word_dropout = word_dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    @property
    def embed_size(self):
        return self._embed_size
    
    @property
    def embedding_dim(self):
        return self._embed_size
    
    @property
    def num_embedding(self):
        return self.model.encoder.config.vocab_size
    
    def index_datasets(self, *datasets, field_name, add_cls_sep=True):
        self.model.index_dataset(*datasets, field_name=field_name, add_cls_sep=add_cls_sep)
    
    def forward(self, word_pieces, token_type_ids=None):
        with torch.no_grad():
            sep_mask = word_pieces.eq(self._sep_index)  # batch_size x max_len
            if token_type_ids is None:
                sep_mask_cumsum = sep_mask.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                token_type_ids = sep_mask_cumsum.fmod(2)
                if token_type_ids[0, 0].item():
                    token_type_ids = token_type_ids.eq(0).long()
        
        word_pieces = self.drop_word(word_pieces)
        outputs = self.model(word_pieces, token_type_ids)
        outputs = torch.cat([*outputs], dim=-1)
        
        return self.dropout_layer(outputs)
    
    def drop_word(self, words):
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                if self._word_sep_index:
                    sep_mask = words.eq(self._wordpiece_unk_index)
                mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)
                pad_mask = words.ne(self._wordpiece_pad_index)
                mask = pad_mask.__and__(mask)
                words = words.masked_fill(mask, self._word_unk_index)
                if self._word_sep_index:
                    words.masked_fill_(sep_mask, self._wordpiece_unk_index)
        return words


class _WordBertModel(nn.Module):
    def __init__(self, model_dir_or_name: str, vocab: Vocabulary, layers: str = '-1', pool_method: str = 'first',
                 include_cls_sep: bool = False, pooled_cls: bool = False, auto_truncate: bool = False, min_freq=2):
        super().__init__()
        
        self.tokenzier = BertTokenizer.from_pretrained(model_dir_or_name)
        self.encoder = BertModel.from_pretrained(model_dir_or_name)
        self._max_position_embeddings = self.encoder.config.max_position_embeddings
        encoder_layer_number = len(self.encoder.encoder.layer)
        self.layers = list(map(int, layers.split(',')))
        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                       f"a bert model with {encoder_layer_number} layers."
            else:
                assert layer < encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                     f"a bert model with {encoder_layer_number} layers."
        
        assert pool_method in ('avg', 'max', 'first', 'last')
        self.pool_method = pool_method
        self.include_cls_sep = include_cls_sep
        self.pooled_cls = pooled_cls
        self.auto_truncate = auto_truncate

        self._has_sep_in_vocab = True

        self._cls_index = self.tokenzier.vocab['[CLS]']
        self._sep_index = self.tokenzier.vocab['[SEP]']
        self._wordpiece_pad_index = self.tokenzier.vocab['[PAD]']
    def forward(self, inputs):
        seq_len=inputs[2]
        batch_word_pieces_length=inputs[1]
        with torch.no_grad():
            # max_word_len=word_piece_length
            batch_size = seq_len.size(0)
            # [batch_size]
            max_word_len=seq_len.max().item()
            word_mask=batch_word_pieces_length.ne(self._wordpiece_pad_index)
            word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)
            word_piece_length=batch_word_pieces_length.sum(dim=-1).max().item()
            if word_piece_length + 2 > self._max_position_embeddings:
                if self.auto_truncate:
                    word_pieces_lengths = word_pieces_lengths.masked_fill(
                        word_pieces_lengths + 2 > self._max_position_embeddings,
                        self._max_position_embeddings - 2)
                else:
                    raise RuntimeError(
                        "After split words into word pieces, the lengths of word pieces are longer than the "
                        f"maximum allowed sequence length:{self._max_position_embeddings} of bert. You can set "
                        f"`auto_truncate=True` for BertEmbedding to automatically truncate overlong input.")

            word_pieces = batch_word_pieces_length.new_full((batch_size, min(word_piece_length + 2, self._max_position_embeddings)),
                                         fill_value=self._wordpiece_pad_index)
            attn_masks = torch.zeros_like(word_pieces)

            for i in range(batch_size):
                word_pieces_i=list(inputs[0][i][:word_pieces_lengths[i]])
                if self.auto_truncate and len(word_pieces_i) > self._max_position_embeddings - 2:
                    word_pieces_i = word_pieces_i[:self._max_position_embeddings - 2]
                word_pieces[i, 1:word_pieces_lengths[i] + 1] = torch.LongTensor(word_pieces_i)
                attn_masks[i, :word_pieces_lengths[i] + 2].fill_(1)

            word_pieces[:, 0].fill_(self._cls_index)
            batch_indexes = torch.arange(batch_size).to(batch_word_pieces_length)
            word_pieces[batch_indexes, word_pieces_lengths + 1] = self._sep_index
            if self._has_sep_in_vocab:
                sep_mask = word_pieces.eq(self._sep_index).long()  # batch_size x max_len
                sep_mask_cumsum = sep_mask.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                token_type_ids = sep_mask_cumsum.fmod(2)
                if token_type_ids[0, 0].item():
                    token_type_ids = token_type_ids.eq(0).long()
            else:
                token_type_ids = torch.zeros_like(word_pieces)

        # all_outputs: [batch_size x max_len x hidden_size, batch_size x max_len x hidden_size, ...]
        bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=token_type_ids,
                                                attention_mask=attn_masks,
                                                output_all_encoded_layers=True)
        # output_layers = [self.layers]  # len(self.layers) x batch_size x real_word_piece_length x hidden_size
        if self.include_cls_sep:
            s_shift = 1
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len + 2,
                                                 bert_outputs[-1].size(-1))

        else:
            s_shift = 0
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len,
                                                 bert_outputs[-1].size(-1))

        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(batch_size, max_word_len + 1)
        batch_word_pieces_cum_length[:, 1:] = batch_word_pieces_length.cumsum(dim=-1)  # batch_size x max_len
        if self.pool_method == 'first':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, :seq_len.max()]
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))
        elif self.pool_method == 'last':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, 1:seq_len.max() + 1] - 1
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))

        for l_index, l in enumerate(self.layers):
            output_layer = bert_outputs[l]
            real_word_piece_length = output_layer.size(1) - 2
            if word_piece_length > real_word_piece_length:
                paddings = output_layer.new_zeros(batch_size,
                                                  word_piece_length - real_word_piece_length,
                                                  output_layer.size(2))
                output_layer = torch.cat((output_layer, paddings), dim=1).contiguous()
            truncate_output_layer = output_layer[:, 1:-1]
            if self.pool_method == 'first':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(0), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1) + s_shift] = tmp

            elif self.pool_method == 'last':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(0), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1) + s_shift] = tmp
            elif self.pool_method == 'max':
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift], _ = torch.max(truncate_output_layer[i, start:end],
                                                                        dim=-2)
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift] = torch.mean(truncate_output_layer[i, start:end],
                                                                      dim=-2)
            if self.include_cls_sep:
                if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                    outputs[l_index, :, 0] = pooled_cls
                else:
                    outputs[l_index, :, 0] = output_layer[:, 0]
                outputs[l_index, batch_indexes, seq_len + s_shift] = output_layer[
                    batch_indexes, word_piece_length + s_shift]

        return outputs
