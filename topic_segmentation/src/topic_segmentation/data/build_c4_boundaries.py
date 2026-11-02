"""Build C4 paragraph-boundary examples using configs/boundary_pretraining.json.

    python -m topic_segmentation.data.build_c4_boundaries

Filter paragraphs and articles using the configured length and sentence-count limits.
Label paragraph-end tokens 1, other tokens 0, and CLS/SEP tokens -100.
Split long articles at paragraph ends, then sentence ends, then the window limit.
"""
import functools
import json
import re
import tempfile
from pathlib import Path

from datasets import (Dataset, DatasetDict, DatasetInfo, Features, Sequence, Value, concatenate_datasets, load_dataset,
                      load_from_disk)
from datasets.table import MemoryMappedTable
from transformers import AutoTokenizer

from topic_segmentation.paths import REPOSITORY

CONFIG = json.loads((REPOSITORY / "configs/boundary_pretraining.json").read_text())
RAW = REPOSITORY / "data/c4/realnewslike"  # local copy; downloaded from the Hub if missing
WORK = REPOSITORY / "cache/c4"  # Temporary Arrow shards can exceed /tmp capacity.
CONTENT_LENGTH = CONFIG["max_sequence_length"] - 2  # room for CLS and SEP
SENTENCE_ENDS = [".", "!", "?", "...", '."', '!"', '?"', ".'", "!'", "?'"]
LOOKBACK = 100  # tokens searched back for a cut point


@functools.cache
def load_tokenizer():
    """Load the tokenizer and sentence-ending token IDs once per worker."""
    tokenizer = AutoTokenizer.from_pretrained(REPOSITORY / CONFIG["base_model"])
    return tokenizer, {token for end in SENTENCE_ENDS for token in tokenizer.encode(end, add_special_tokens=False)}


def paragraph_end_labels(offsets, paragraphs):
    """Label the token at each paragraph end, or the preceding token if the end falls in whitespace."""
    targets, position = [], 0
    for paragraph in paragraphs:
        position += len(paragraph)
        targets.append(position - 1)
        position += 1  # the joining space
    labels, next_target = [0] * len(offsets), 0
    for token, (start, end) in enumerate(offsets):
        if next_target >= len(targets):
            break
        if start <= targets[next_target] < end:
            labels[token] = 1
            next_target += 1
            while next_target < len(targets) and start <= targets[next_target] < end:
                next_target += 1
        elif start > targets[next_target]:
            if token > 0:
                labels[token - 1] = 1
            next_target += 1
    labels[-1] = 1
    return labels


def cut(ids, labels, sentence_end_ids):
    """Choose a cut within the last LOOKBACK tokens: the latest paragraph end,
    the earliest sentence end, or the window end, in that order.
    """
    point = len(ids) - 1
    for index in range(len(ids) - 1, max(-1, len(ids) - LOOKBACK - 1), -1):
        if labels[index] == 1:
            return index
        if ids[index] in sentence_end_ids:
            point = index
    return point


def split_article(text, tokenizer, sentence_end_ids):
    """Return labeled token windows for one article."""
    paragraphs = [paragraph.strip() for paragraph in text.split("\n")
                  if len(paragraph.split()) >= CONFIG["minimum_paragraph_words"]]
    if len(paragraphs) < CONFIG["minimum_paragraph_count"]:
        return []
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if len(sentence) > 10]
    if len(sentences) / len(paragraphs) <= CONFIG["minimum_average_sentences_per_paragraph"]:
        return []
    encoding = tokenizer(" ".join(paragraphs), add_special_tokens=False, return_offsets_mapping=True)
    if not encoding["input_ids"]:
        return []
    labels = paragraph_end_labels(encoding["offset_mapping"], paragraphs)
    separator = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    windows, ids, window_labels = [], [], []
    for token, label in zip(encoding["input_ids"], labels):
        if len(ids) >= CONTENT_LENGTH:
            point = cut(ids, window_labels, sentence_end_ids) + 1
            windows.append((ids[:point], window_labels[:point]))
            ids, window_labels = ids[point:], window_labels[point:]
        ids.append(token)
        window_labels.append(label)
    windows.append((ids, window_labels))
    return [{"input_ids": [tokenizer.cls_token_id, *ids, separator], "attention_mask": [1] * (len(ids) + 2),
             "labels": [-100, *labels, -100]} for ids, labels in windows]


def tokenize(batch):
    """Convert a batch of articles to labeled windows for Dataset.map."""
    tokenizer, sentence_end_ids = load_tokenizer()
    windows = [window for text in batch["text"] for window in split_article(text, tokenizer, sentence_end_ids)]
    return {name: [window[name] for window in windows] for name in ("input_ids", "attention_mask", "labels")}


def main():
    raw = load_from_disk(str(RAW)) if RAW.exists() else load_dataset("allenai/c4", "realnewslike")
    output = REPOSITORY / CONFIG["prepared_dataset"]
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c4-map-", dir=WORK) as temporary:
        prepared = DatasetDict({
            split: raw[split].map(tokenize, batched=True, batch_size=500, num_proc=16, remove_columns=raw[split].column_names,
                                  cache_file_name=str(Path(temporary) / f"{split}.arrow"), desc=f"Tokenizing {split}")
            for split in ("train", "validation")})
        for split, dataset in prepared.items():
            print(f"{split}: {len(raw[split]):,} articles -> {len(dataset):,} windows")
        prepared.save_to_disk(str(output))
        del prepared, dataset  # Close Arrow mappings before removing the temporary files on NFS.
    print(f"wrote {output}")


# Read Arrow shards directly to support datasets saved by newer library versions.
FEATURES = Features({
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
    "labels": Sequence(Value("int64")),
})


def load_splits(path):
    """Load train and validation splits from memory-mapped Arrow shards."""
    splits = {}
    for split in ("train", "validation"):
        shards = sorted((Path(path) / split).glob("data-*.arrow"))
        splits[split] = concatenate_datasets([
            Dataset(MemoryMappedTable.from_file(str(shard)).replace_schema_metadata(None), info=DatasetInfo(features=FEATURES))
            for shard in shards])
    return DatasetDict(splits)


if __name__ == "__main__":
    main()
