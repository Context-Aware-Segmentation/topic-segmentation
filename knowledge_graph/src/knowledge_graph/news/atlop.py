"""ATLOP (Zhou et al., 2021), adapted from https://github.com/wzhouad/ATLOP."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge_graph.news.encoding import encode_chunks, pool_entities


class ATLoss(nn.Module):
    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels = labels.clone()
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        return (loss1 + loss2).mean()

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits)
        mask = logits > th_logit
        if num_labels > 0:
            # Break score ties by relation ID and keep at most num_labels predictions.
            top = logits.argsort(dim=1, descending=True, stable=True)[:, :num_labels]
            selected = torch.zeros_like(mask).scatter_(1, top, True)
            mask &= selected
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.0).to(logits)
        return output


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask, long_input_encoder=None):
        encoder = long_input_encoder or encode_chunks
        return encoder(self.model, input_ids, attention_mask, layers=1)

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = pool_entities(sequence_output[i], attention[i], entity_pos[i], offset)

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = torch.einsum("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self, input_ids=None, attention_mask=None, labels=None, entity_pos=None, hts=None, long_input_encoder=None):
        sequence_output, attention = self.encode(input_ids, attention_mask, long_input_encoder)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        output = {"rel_pred": self.loss_fnt.get_label(logits, num_labels=self.num_labels)}
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}
        return output
