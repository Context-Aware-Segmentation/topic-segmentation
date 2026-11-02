"""Training objectives of Yu et al. (2023): topic segmentation (TS), CSSL, and TSSP."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine(x, y, temp):
    """Compute cosine similarity scaled by temperature."""
    return F.cosine_similarity(x, y, dim=-1) / temp


def section_ids(labels):
    """Assign topic IDs to boundary candidates, numbered across documents."""
    ids, topic = [], 0
    for example in (l[l != -100] for l in labels):
        if len(example) == 0:
            continue
        for label in example:
            ids.append(topic)
            if label == 0:
                topic += 1
        if example[-1] == 1:
            topic += 1
    return ids


def sample_pairs(ids, positives, negatives):
    """Preceding candidates of the same topic as positives and following candidates as negatives."""
    total = len(ids)
    starts = [ids.index(topic) for topic in range(ids[-1] + 1)]
    ends = [start - 1 for start in starts[1:]] + [total - 1]
    chosen = [[] for _ in range(positives)], [[] for _ in range(negatives)]
    for index, topic in enumerate(ids):
        start, end = starts[topic], ends[topic]
        choices = list(range(start, end)) or [end]
        candidate = index
        for column in chosen[0]:
            candidate -= 1
            if candidate < start:
                candidate = random.choice(choices)
            column.append(candidate)
        choices = list(range(end + 1, total)) or list(range(starts[0], starts[1]))
        candidate = end
        for column in chosen[1]:
            candidate += 1
            if candidate >= total:
                candidate = random.choice(choices)
            column.append(candidate)
    return chosen


def cssl(sequence_output, labels, segment_ids, eop_index, config):
    """Compute InfoNCE over max-pooled candidate features, using each candidate as an anchor."""
    ids = section_ids(labels)
    if len(ids) <= 2 or ids[-1] == 0:  # needs at least two topics
        return torch.tensor(0.0).to(sequence_output.device)
    batch, length = sequence_output.shape[0], sequence_output.shape[1]
    pooled = torch.zeros_like(sequence_output).scatter_reduce(
        1, segment_ids[:, :, None].expand_as(sequence_output), sequence_output, reduce="amax", include_self=False)
    offsets = eop_index + torch.arange(batch).to(sequence_output.device).unsqueeze(1).expand_as(eop_index) * length
    features = pooled.reshape(batch * length, -1)[offsets.reshape(-1)[eop_index.reshape(-1) != 0]]
    positives, negatives = sample_pairs(ids, config.cl_positive_k, config.cl_negative_k)
    similarity = torch.exp(torch.cat([cosine(features, features[column], config.cl_temp).unsqueeze(0)
                                      for column in positives + negatives]))
    count = features.shape[0]
    mask = torch.tensor([[1] * count] * len(positives) + [[0] * count] * len(negatives)).to(sequence_output.device)
    return (-1 * torch.log(torch.sum(similarity * mask, 0) / torch.sum(similarity, 0))).mean()


class TSSP(nn.Module):
    """Topic-aware sentence structure prediction head on the augmented view."""

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, config.num_tssp_labels)

    def forward(self, sequence_output, sent_token_mask, sent_pair_orders):
        logits = self.classifier(sequence_output[sent_token_mask != -100])
        targets = sent_pair_orders[sent_pair_orders != -100]
        return F.cross_entropy(logits, targets) if targets.numel() else logits.sum() * 0.0


class CombinedObjective(nn.Module):
    """Combine boundary classification, CSSL, and TSSP losses."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.tssp = TSSP(config)

    def forward(self, sequence_output, labels, segment_ids=None, eop_index=None,
                sent_token_mask=None, sent_pair_orders=None, augmented=False):
        config = self.config
        logits = self.classifier(sequence_output)
        loss = 0.0
        # Apply TS to the augmented view when document augmentation is enabled.
        if not augmented or config.do_da_ts:
            # Final singleton windows can have no internal boundary to supervise.
            ts = (F.cross_entropy(logits.reshape(-1, config.num_labels), labels.reshape(-1))
                  if (labels != -100).any() else logits.sum() * 0.0)
            loss = loss + config.ts_loss_weight * ts
        if not augmented and config.cl_loss_weight != 0:
            loss = loss + config.cl_loss_weight * cssl(sequence_output, labels, segment_ids, eop_index, config)
        if augmented and config.tssp_loss_weight != 0:
            loss = loss + config.tssp_loss_weight * self.tssp(sequence_output, sent_token_mask, sent_pair_orders)
        return loss, logits
