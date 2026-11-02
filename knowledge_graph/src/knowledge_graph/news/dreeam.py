"""DREEAM (Ma et al., 2023), adapted from https://github.com/YoumiMa/dreeam (MIT).

Uses supervised relation and sentence-evidence losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge_graph.news.atlop import ATLoss
from knowledge_graph.news.encoding import encode_chunks, pool_entities


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        """Build the relation classifier and sentence-evidence loss."""

        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.offset = 1 if config.transformer_type in ("bert", "roberta") else 0

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)

        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask, long_input_encoder=None):
        """Token states and attention from the supplied encoder or the default chunk encoder."""
        encoder = long_input_encoder or encode_chunks
        return encoder(self.model, input_ids, attention_mask, layers=3)

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        """Pool head, tail, and context representations; retain token attention for evidence scoring."""

        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)):
            entity_embs, entity_atts = pool_entities(sequence_output[i], attention[i], entity_pos[i], self.offset)

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

            rs = torch.einsum("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0)  # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch

    def forward_rel(self, hs, ts, rs):
        """Classify entity pairs with grouped bilinear interactions."""

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        return logits

    def forward_evi(self, doc_attn, sent_pos, batch_rel):
        """Sum token attention into a sentence-evidence distribution for each entity pair."""

        max_sent_num = max(len(sentences) for sentences in sent_pos)
        rel_sent_attn = []
        for attention, sentences in zip(doc_attn.split(batch_rel), sent_pos):
            columns = [attention[:, start + self.offset : end + self.offset].sum(-1) for start, end in sentences]
            sentence_attention = torch.stack(columns, dim=-1)
            rel_sent_attn.append(F.pad(sentence_attention, (0, max_sent_num - len(sentences))))
        return torch.cat(rel_sent_attn, dim=0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,  # relation labels
        entity_pos=None,
        hts=None,  # entity pairs
        sent_pos=None,
        sent_labels=None,  # evidence counts by sentence
        long_input_encoder=None,
    ):
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask, long_input_encoder)

        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(sequence_output, attention, entity_pos, hts)
        logits = self.forward_rel(hs, ts, rs)
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        if labels is None:
            return output

        loss = self.loss_fnt(logits.float(), labels.float())
        output["loss"] = {"rel_loss": loss.to(sequence_output)}

        if sent_labels is not None:  # sentence evidence derived from entity endpoints
            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel)
            # Chunk attention may not reach every retained evidence sentence.
            sent_labels = sent_labels * (s_attn > 0)
            has_evidence = (labels[:, 1:].sum(-1) > 0) & (sent_labels.sum(-1) > 0)
            idx_used = torch.nonzero(has_evidence).view(-1)

            if idx_used.numel():
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
            else:
                evi_loss = s_attn.sum() * 0
            output["loss"]["evi_loss"] = evi_loss.to(sequence_output)

        return output
