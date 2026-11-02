"""Shared DocRED features and batches for ATLOP and DREEAM.

Adapted from https://github.com/wzhouad/ATLOP and https://github.com/YoumiMa/dreeam (MIT).
"""

import torch
import torch.nn.functional as F


def tokenize_document(sample, tokenizer):
    """Tokenize sentences with entity markers and retain word and sentence offsets."""
    starts, ends = set(), set()
    for entity in sample["vertexSet"]:
        for mention in entity:
            starts.add((mention["sent_id"], mention["pos"][0]))
            ends.add((mention["sent_id"], mention["pos"][1] - 1))
    tokens, word_maps, sentence_spans = [], [], []
    for sentence_id, sentence in enumerate(sample["sents"]):
        begin = len(tokens)
        word_map = {}
        for word_id, word in enumerate(sentence):
            pieces = tokenizer.tokenize(word)
            if (sentence_id, word_id) in starts:
                pieces = ["*"] + pieces
            if (sentence_id, word_id) in ends:
                pieces = pieces + ["*"]
            word_map[word_id] = len(tokens)
            tokens.extend(pieces)
        word_map[len(sentence)] = len(tokens)
        word_maps.append(word_map)
        sentence_spans.append((begin, len(tokens)))
    return tokens, word_maps, sentence_spans


def prepare_features(documents, tokenizer, max_seq_length=1024):
    """Build token IDs, entity pairs, and offsets without training targets."""
    features = []
    for sample in documents:
        tokens, word_maps, sentence_spans = tokenize_document(sample, tokenizer)
        token_limit = max_seq_length - 2
        entity_pos = [
            [
                (start, min(word_maps[m["sent_id"]][m["pos"][1]], token_limit))
                for m in entity
                if (start := word_maps[m["sent_id"]][m["pos"][0]]) < token_limit
            ]
            for entity in sample["vertexSet"]
        ]
        # Keep the training order: labeled pairs first, then the remaining ordered pairs.
        positive_pairs = dict.fromkeys((label["h"], label["t"]) for label in sample.get("labels", []))
        pairs = list(positive_pairs)
        pairs.extend(
            (head, tail)
            for head in range(len(entity_pos))
            for tail in range(len(entity_pos))
            if head != tail and (head, tail) not in positive_pairs
        )

        # Entity markers and sentence evidence share the encoder's content-token limit.
        kept = [i for i, (start, _) in enumerate(sentence_spans) if start < token_limit]
        input_ids = tokenizer.convert_tokens_to_ids(tokens[:token_limit])
        features.append(
            {
                "input_ids": tokenizer.build_inputs_with_special_tokens(input_ids),
                "entity_pos": entity_pos,
                "hts": [list(pair) for pair in pairs],
                "sent_pos": [(sentence_spans[i][0], min(sentence_spans[i][1], token_limit)) for i in kept],
                "title": sample["title"],
            }
        )
    return features


def add_relation_targets(features, documents, rel2id):
    """Attach multi-label relation targets in each feature's entity-pair order."""
    for feature, sample in zip(features, documents, strict=True):
        targets = {tuple(pair): [0] * len(rel2id) for pair in feature["hts"]}
        for label in sample.get("labels", []):
            targets[label["h"], label["t"]][int(rel2id[label["r"]])] = 1
        for row in targets.values():
            if not any(row):
                row[0] = 1
        feature["labels"] = list(targets.values())


def add_evidence_targets(features, documents):
    """Attach sentence-evidence counts within the encoder's token limit."""
    for feature, sample in zip(features, documents, strict=True):
        sentence_count = len(feature["sent_pos"])
        targets = {tuple(pair): [0] * sentence_count for pair in feature["hts"]}
        for label in sample.get("labels", []):
            for sentence_id in label["evidence"]:
                if sentence_id < sentence_count:
                    targets[label["h"], label["t"]][sentence_id] += 1
        feature["sent_labels"] = list(targets.values())


def collate_fn(batch):
    """Pad token sequences and include the training targets present in the batch."""
    max_tokens = max(len(item["input_ids"]) for item in batch)
    input_ids = [item["input_ids"] + [0] * (max_tokens - len(item["input_ids"])) for item in batch]
    masks = [[1.0] * len(item["input_ids"]) + [0.0] * (max_tokens - len(item["input_ids"])) for item in batch]
    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.float),
        "entity_pos": [item["entity_pos"] for item in batch],
        "hts": [item["hts"] for item in batch],
    }
    if "labels" in batch[0]:
        result["labels"] = torch.tensor([row for item in batch for row in item["labels"]])
    if "sent_labels" in batch[0]:
        max_sentences = max(len(item["sent_pos"]) for item in batch)
        evidence = [
            F.pad(torch.tensor(item["sent_labels"], dtype=torch.long), (0, max_sentences - len(item["sent_pos"])))
            for item in batch
        ]
        result["sent_pos"] = [item["sent_pos"] for item in batch]
        result["sent_labels"] = torch.cat(evidence, dim=0)
    return result
