"""Build paired token windows for topic-segmentation training.

Each paragraph or sentence starts with a BOS token carrying its boundary label.
WindowMaker creates anchor and augmented views, slices both at the anchor's token
offsets, and adds padding and CSSL/TSSP features. Views are stored in [anchor, augmented] order.
"""
from dataclasses import dataclass
from pathlib import Path
import random

import datasets

from topic_segmentation.augmentation import augment

SCHEMA = datasets.Features({
    "example_id": datasets.Value("int32"),
    "sentences": datasets.Sequence(datasets.Value("string")),
    "labels": datasets.Sequence(datasets.Value("string")),
})
LABELS = {1: "B-EOP", 0: "O", "1": "B-EOP", "0": "O"}
SPLIT_FILES = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}
COLUMNS = (
    "example_id", "labels", "sentences", "input_ids", "token_type_ids", "attention_mask",
    "extract_eop_segment_ids", "eop_index_for_aggregate_batch_eop_features", "sent_pair_orders", "sent_token_mask",
)


def normalize_row(example, index):
    assert len(example["sentences"]) == len(example["labels"]), f"unaligned row {index}"
    return {
        "example_id": index,
        "sentences": example["sentences"],
        "labels": [LABELS.get(label, str(label)) for label in example["labels"]],
    }


def load_splits(data_dir=None, data_files=None, cache_dir=None):
    """Read `data_files` ({split: path}) or the train/val/test files in `data_dir`."""
    if not data_files:
        data_files = {split: Path(data_dir) / name for split, name in SPLIT_FILES.items()
                      if (Path(data_dir) / name).exists()}
    splits = {}
    for split, path in data_files.items():  # splits may carry different extra columns
        raw = datasets.load_dataset("json", data_files=str(path), split="train", cache_dir=cache_dir)
        splits[split] = raw.map(normalize_row, with_indices=True, features=SCHEMA,
                                remove_columns=[c for c in raw.column_names if c not in SCHEMA])
    return datasets.DatasetDict(splits)


@dataclass(frozen=True)
class Window:
    token_start: int
    token_end: int
    sentence_start: int
    sentence_end: int
    masked_boundary: int | None


def pack_windows(input_ids, sentence_starts, bos_token_id, max_length):
    ends = [i for i in range(1, len(input_ids)) if input_ids[i] == bos_token_id]
    ends.append(len(input_ids))
    token_start = sentence_start = current_sentence = 0
    while current_sentence < len(ends):
        token_end = ends[current_sentence]
        at_document_end = token_end == len(input_ids)
        if token_end - token_start < max_length - 1 and not at_document_end:
            current_sentence += 1
            continue
        sentence_end = current_sentence + 1
        single_sentence = current_sentence == sentence_start
        # Internal singleton windows are not repeated in the next window.
        masked_boundary = (sentence_starts[current_sentence] - token_start + 1
                           if at_document_end or not single_sentence else None)
        yield Window(token_start, token_end, sentence_start, sentence_end,
                     masked_boundary)
        token_start = token_end if single_sentence else ends[current_sentence - 1]
        if single_sentence or at_document_end:
            sentence_start = sentence_end
            current_sentence += 1
        else:
            # Adjacent windows share their final/initial sentence.
            sentence_start = sentence_end - 1


@dataclass
class WindowMaker:
    """Tokenize documents and produce paired anchor and augmented windows."""

    tokenizer: object
    label_to_id: dict
    boundary_ids: set
    max_length: int
    seed: int

    def __call__(self, examples):
        labels, contexts, example_ids = examples["labels"], examples["sentences"], examples["example_id"]
        # One generator per batch, fixed by the seed and the batch's first document, whichever worker runs it.
        rng = random.Random(self.seed * 1_000_000 + example_ids[0])
        sentences = [[self.tokenizer.bos_token + text for text in texts] for texts in contexts]
        tokenized = self.tokenizer(sentences, is_split_into_words=True, add_special_tokens=False,
                                   return_token_type_ids=True, return_attention_mask=True)
        documents, sentence_starts = self.align(tokenized, labels)
        augmented = augment(sentences, labels, tokenized, self.label_to_id, self.boundary_ids, rng)
        output = {name: [] for name in COLUMNS}
        for index, document in enumerate(documents):
            display_sentences = [f"{i}-{text}" for i, text in enumerate(contexts[index])]
            for pair in self.pairs(document, augmented[index], example_ids[index],
                                   display_sentences, sentence_starts[index]):
                for name, values in pair.items():
                    output[name].append(values)
        return output

    def align(self, tokenized, labels):
        documents, all_starts = [], []
        for index, input_ids in enumerate(tokenized["input_ids"]):
            starts = [i for i, token in enumerate(input_ids) if token in self.boundary_ids]
            token_labels = [-100] * len(input_ids)
            for sentence_index, token_index in enumerate(starts):
                token_labels[token_index] = self.label_to_id.get(labels[index][sentence_index], -100)
            documents.append({
                "input_ids": input_ids,
                "labels": token_labels,
                "token_type_ids": tokenized["token_type_ids"][index],
                "attention_mask": tokenized["attention_mask"][index],
            })
            all_starts.append(starts)
        return documents, all_starts

    def slice(self, fields, window):
        prefixes = {"input_ids": self.tokenizer.cls_token_id, "labels": -100,
                    "token_type_ids": 0, "attention_mask": 1, "sent_pair_orders": -100}
        return {
            name: ([prefixes[name]] + values[window.token_start:window.token_end])[:self.max_length]
            for name, values in fields.items()
        }

    def finish(self, fields):
        padding = {"input_ids": self.tokenizer.pad_token_id, "labels": -100,
                   "token_type_ids": 0, "attention_mask": 0, "sent_pair_orders": -100}
        pad_count = self.max_length - len(fields["input_ids"])
        for name, values in fields.items():
            values.extend([padding[name]] * pad_count)
        fields.update(self.auxiliary(fields["input_ids"], fields["labels"]))

    def auxiliary(self, input_ids, labels):
        """Build the TSSP unit mask and CSSL pooling indices."""
        sentence_mask = [-100]
        segment_ids = [0]
        segment_id = 0
        for token, label in zip(input_ids[1:], labels[1:]):
            is_boundary = token in self.boundary_ids
            # Model class 0 denotes B-EOP; ignored boundary positions retain mask 1.
            sentence_mask.append((0 if label == 0 else 1) if is_boundary else -100)
            if is_boundary and label != -100:
                segment_id += 1
                segment_ids.append(segment_id)
            else:
                segment_ids.append(0)
        boundary_count = sum(label != -100 for label in labels)
        aggregate_indices = list(range(boundary_count + 1))
        aggregate_indices.extend([0] * (self.max_length - boundary_count - 1))
        return {
            "sent_token_mask": sentence_mask,
            "extract_eop_segment_ids": segment_ids,
            "eop_index_for_aggregate_batch_eop_features": aggregate_indices,
        }

    def pairs(self, document, augmented, example_id, sentences, sentence_starts):
        augmented_fields = {
            "input_ids": augmented.input_ids,
            "labels": augmented.token_labels,
            "token_type_ids": [0] * len(augmented.input_ids),
            "attention_mask": [1] * len(augmented.input_ids),
            "sent_pair_orders": augmented.pair_orders,
        }
        for window in pack_windows(document["input_ids"], sentence_starts, self.tokenizer.bos_token_id, self.max_length):
            anchor = self.slice(document, window)
            # Slice both views at the anchor token offsets.
            other = self.slice(augmented_fields, window)
            if window.masked_boundary is not None:
                anchor["labels"][window.masked_boundary] = -100
            self.finish(anchor)
            self.finish(other)
            pair_orders = other["sent_pair_orders"]
            assert sum(v != -100 for v in pair_orders) == sum(v != -100 for v in other["sent_token_mask"])
            anchor["sent_pair_orders"] = pair_orders
            anchor["example_id"] = other["example_id"] = example_id
            start, end = window.sentence_start, window.sentence_end
            anchor["sentences"] = sentences[start:end]
            other["sentences"] = augmented.sentences[start:end]
            yield {name: [anchor[name], other[name]] for name in COLUMNS}
