"""PLM-NR (Wu et al., 2021) with a BERT news encoder, adapted from https://github.com/wuch15/PLM4NewsRec.

Only the title encoder is kept: each input variant packs its text into the title field. Attribute names match the
state-dict keys of the trained checkpoints, and modules are built in the upstream order, so a seeded run draws the
same initial weights.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Attention pooling over the second dimension."""

    def __init__(self, dimension, hidden_size=200):
        super().__init__()
        self.att_fc1 = nn.Linear(dimension, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        alpha = torch.exp(self.att_fc2(torch.tanh(self.att_fc1(x))))
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        return torch.bmm(x.permute(0, 2, 1), alpha).reshape(x.shape[0], -1)


class MultiHeadAttention(nn.Module):
    """Unmasked multi-head self-attention with exp-normalized scores, as upstream."""

    def __init__(self, dimension, heads, head_size):
        super().__init__()
        self.heads, self.head_size = heads, head_size
        self.W_Q = nn.Linear(dimension, head_size * heads)
        self.W_K = nn.Linear(dimension, head_size * heads)
        self.W_V = nn.Linear(dimension, head_size * heads)
        for linear in (self.W_Q, self.W_K, self.W_V):
            nn.init.xavier_uniform_(linear.weight, gain=1)

    def forward(self, x):
        batch_size = x.shape[0]
        q, k, v = (linear(x).view(batch_size, -1, self.heads, self.head_size).transpose(1, 2)
                   for linear in (self.W_Q, self.W_K, self.W_V))
        scores = torch.exp(torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_size))
        attention = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.head_size)


class TextEncoder(nn.Module):
    """BERT token states, multi-head self-attention, and attention pooling of one packed text."""

    def __init__(self, bert_model, config):
        super().__init__()
        self.bert_model = bert_model
        self.dropout_rate = config["drop_rate"]
        self.multihead_attention = MultiHeadAttention(config["word_embedding_dim"], config["num_attention_heads"], 20)
        self.additive_attention = AdditiveAttention(config["num_attention_heads"] * 20, config["news_query_vector_dim"])

    def forward(self, text):
        """`text` packs input ids, token type ids, and attention mask side by side."""
        ids, types, mask = text.chunk(3, dim=1)
        words = self.bert_model(input_ids=ids, token_type_ids=types, attention_mask=mask, return_dict=True).last_hidden_state
        words = F.dropout(words, p=self.dropout_rate, training=self.training)
        words = F.dropout(self.multihead_attention(words), p=self.dropout_rate, training=self.training)
        return self.additive_attention(words)


class NewsEncoder(nn.Module):
    def __init__(self, bert_model, config):
        super().__init__()
        self.text_encoders = nn.ModuleDict({"title": TextEncoder(bert_model, config)})
        self.reduce_dim_linear = nn.Linear(config["num_attention_heads"] * 20, config["news_dim"])

    def forward(self, news):
        return self.reduce_dim_linear(self.text_encoders["title"](news))


class UserEncoder(nn.Module):
    """Attention pooling over all history slots; padded slots hold the encoded all-zero news row."""

    def __init__(self, config):
        super().__init__()
        self.news_additive_attention = AdditiveAttention(config["news_dim"], config["user_query_vector_dim"])

    def forward(self, history):
        return self.news_additive_attention(history)


class ModelBert(nn.Module):
    """Click-prediction loss over 1 + K candidates and the user's history."""

    def __init__(self, config, bert_model):
        super().__init__()
        self.npratio, self.user_log_length, self.news_dim = config["npratio"], config["user_log_length"], config["news_dim"]
        self.news_encoder = NewsEncoder(bert_model, config)
        self.user_encoder = UserEncoder(config)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, candidates, history, history_mask, targets):
        length = candidates.size(2)
        news = self.news_encoder(candidates.view(-1, length)).view(-1, 1 + self.npratio, self.news_dim)
        # Padded history slots hold the all-zero news row; encode it once and reuse it.
        history = history.view(-1, length)
        real = history_mask.reshape(-1) > 0
        vectors = self.news_encoder(history[real])
        if not real.all():
            padding = self.news_encoder(torch.zeros_like(history[:1]))
            vectors = padding.expand(len(real), -1).index_put((real.nonzero().squeeze(1),), vectors)
        user = self.user_encoder(vectors.view(-1, self.user_log_length, self.news_dim))
        score = torch.bmm(news, user.unsqueeze(-1)).squeeze(dim=-1)
        return self.criterion(score, targets)
