# Copyright 2020 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Adapted from the Transformers token-classification example via Yu et al. (2023).
"""Fine-tune, evaluate, and predict topic segmentation with the Hugging Face Trainer.

    python -m topic_segmentation.training --model_name_or_path MODEL --dataset_name news --output_dir OUT \\
        --do_train True --do_eval True --do_predict True ...

scripts/run_news_finetuning.py, run_mind_adaptation.py, and run_clinical_transfer.py build the full commands.
"""
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import datasets
import transformers
from transformers import (AutoConfig, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments,
                          default_data_collator, set_seed)

from topic_segmentation import metrics as scoring
from topic_segmentation import models
from topic_segmentation import prepare_inputs
from topic_segmentation.paths import REPOSITORY

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Initial checkpoint and training objectives."""

    model_name_or_path: str = field(metadata={"help": "Initial checkpoint"})
    do_da_ts: bool = field(default=False, metadata={"help": "Also train topic segmentation on the augmented view"})
    do_tssp: bool = field(default=False, metadata={"help": "Encode the augmented view for TSSP"})
    ts_loss_weight: float = field(default=1.0, metadata={"help": "Topic segmentation loss weight"})
    cl_loss_weight: float = field(default=0.0, metadata={"help": "CSSL loss weight; 0 disables CSSL"})
    cl_temp: float = field(default=1, metadata={"help": "CSSL cosine-similarity temperature"})
    cl_positive_k: int = field(default=1, metadata={"help": "CSSL positives per anchor from the same topic"})
    cl_negative_k: int = field(default=1, metadata={"help": "CSSL negatives per anchor from the following topics"})
    tssp_loss_weight: float = field(default=0.0, metadata={"help": "TSSP loss weight"})


@dataclass
class DataTrainingArguments:
    """Dataset, windowing, and evaluation schedule."""

    dataset_name: str = field(metadata={"help": "Selects data/<name> and names the prediction files"})
    data_dir: Optional[str] = field(default=None, metadata={"help": "Directory with train/val/test JSONL splits"})
    train_file: Optional[str] = field(default=None, metadata={"help": "Training JSONL instead of data_dir"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Validation JSONL instead of data_dir"})
    test_file: Optional[str] = field(default=None, metadata={"help": "Test JSONL instead of data_dir"})
    test_data_name: Optional[str] = field(default=None, metadata={"help": "Test-set name in prediction files"})
    dataset_cache_dir: Optional[str] = field(default="./cache", metadata={"help": "Dataset and tokenization cache"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Tokenization processes"})
    max_seq_length: int = field(default=4096, metadata={"help": "Tokens per window"})
    eval_cnt: int = field(default=50, metadata={"help": "Evaluations per training run"})


LABELS = ["B-EOP", "O"]  # class 0 marks a segment-final unit
LABEL_IDS = {label: i for i, label in enumerate(LABELS)}
BOUNDARY = LABEL_IDS["B-EOP"]
MODELS = {"bert": models.Bert, "roberta": models.Roberta, "longformer": models.Longformer,
          "modernbert": models.ModernBert}


def configure_logging(args):
    """Configure stdout logging at the current process log level."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    level = args.get_process_log_level()
    logger.setLevel(level)
    datasets.utils.logging.set_verbosity(level)
    transformers.utils.logging.set_verbosity(level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {args}")


def load_model(model_args, max_length):
    """Load the tokenizer and model, then configure the objectives and boundary tokens."""
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=len(LABELS))
    config.update(model_args.__dict__)
    config.num_tssp_labels = 3
    prefix = {"add_prefix_space": True} if config.model_type in ("roberta", "longformer") else {}
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True, **prefix)
    # This loading flag affects RNG consumption when initializing new weights.
    model = MODELS[config.model_type].from_pretrained(model_args.model_name_or_path, config=config,
                                                      ignore_mismatched_sizes=config.model_type != "longformer")
    if not tokenizer.bos_token:  # the BOS token opens every unit
        if config.model_type == "modernbert":
            tokenizer.bos_token = tokenizer.cls_token  # a new ModernBERT embedding row yields NaN
        else:
            tokenizer.add_special_tokens({"bos_token": "[BOS]"})
            model.resize_token_embeddings(len(tokenizer))
    if max_length > tokenizer.model_max_length:  # Persist the requested window length.
        tokenizer.model_max_length = tokenizer.init_kwargs["model_max_length"] = max_length
    model.config.label2id, model.config.id2label = LABEL_IDS, dict(enumerate(LABELS))
    return tokenizer, model


def make_windows(data_args, args, tokenizer):
    """Return windows for the requested splits and test documents when predicting."""
    data_dir = data_args.data_dir or str(REPOSITORY / "data" / data_args.dataset_name)
    files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
    # Rank 0 builds the dataset cache; other ranks reuse it.
    with args.main_process_first(desc="dataset loading"):
        splits = prepare_inputs.load_splits(data_dir, {split: path for split, path in files.items() if path},
                                            data_args.dataset_cache_dir)
    columns = next(iter(splits.values())).column_names
    window_maker = prepare_inputs.WindowMaker(tokenizer, LABEL_IDS, {tokenizer.bos_token_id}, data_args.max_seq_length,
                                              args.seed)

    def windows(documents, split):
        with args.main_process_first(desc=f"{split} dataset map pre-processing"):
            return documents.map(window_maker, batched=True, batch_size=10000, remove_columns=columns,
                                 num_proc=data_args.preprocessing_num_workers,
                                 desc=f"Running tokenizer on {split} dataset")

    output, test_documents = {}, None
    if args.do_train:
        output["train"] = windows(splits["train"], "train")
    if args.do_eval:
        output["validation"] = windows(splits["validation"], "validation")
    if args.do_predict:
        test_documents = splits["test"]
        output["test"] = windows(test_documents, "prediction")
    return output, test_documents


def build_trainer(model, tokenizer, args, windows, evaluations):
    """Build a Trainer with boundary F1 for evaluation.

    Step-based evaluation and saving use max(total_steps // evaluations, 40) steps.
    """
    if args.do_train and args.eval_strategy == "steps":
        batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        steps = len(windows["train"]) * args.num_train_epochs // batch
        if args.max_steps > 0:
            steps = args.max_steps
        args.logging_steps = args.eval_steps = args.save_steps = max(steps // evaluations, 40)
    return Trainer(model=model, args=args, tokenizer=tokenizer, data_collator=default_data_collator,
                   train_dataset=windows.get("train"), eval_dataset=windows.get("validation"),
                   compute_metrics=scoring.trainer_metrics(BOUNDARY))


def train(trainer, windows):
    result = trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)
    trainer.save_model()
    report(trainer, "train", dict(result.metrics, train_samples=len(windows)))
    trainer.save_state()


def evaluate(trainer, windows):
    report(trainer, "eval", dict(trainer.evaluate(), eval_samples=len(windows)))


def report(trainer, name, metrics):
    trainer.log_metrics(name, metrics)
    trainer.save_metrics(name, metrics)


def predict(trainer, documents, windows, data_args):
    """Save document predictions and return their scores on the main process.

    Write `predict_<test>_max_seq<length>_ts_score_lt.txt`; other ranks return None.
    """
    test_name = data_args.test_data_name or data_args.dataset_name
    name = f"predict_{test_name}_max_seq{data_args.max_seq_length}_ts_score_lt"
    logits, labels, metrics = trainer.predict(windows, metric_key_prefix="predict")
    report(trainer, name, metrics)
    if not trainer.is_world_process_zero():
        return None
    rows = [{"sentences": [], "labels": [], "int_labels": [], "predictions": [], "predict_logits": []}
            for _ in range(len(documents))]
    # Each window extends the document it came from (anchor view only).
    for window_logits, window_labels, sentences, ids in zip(logits, labels[:, 0], windows["sentences"],
                                                            windows["example_id"]):
        keep = window_labels != -100
        row = rows[ids[0]]
        row["sentences"].extend(sentences)
        row["labels"].extend(LABELS[label] for label in window_labels[keep])
        row["int_labels"].extend(int(label) for label in window_labels[keep])
        row["predictions"].extend(LABELS[p] for p in window_logits[keep].argmax(-1))
        row["predict_logits"].extend(window_logits[keep].tolist())
    with open(os.path.join(trainer.args.output_dir, name + ".txt"), "w") as writer:
        writer.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    scored = [row for row in rows if row["int_labels"]]
    scores = scoring.score_documents([row["predict_logits"] for row in scored], [row["int_labels"] for row in scored],
                                     BOUNDARY)
    metrics = {f"threshold_0.5_example_level_{key}": value for key, value in scores.items()}
    report(trainer, "example_level_" + name, metrics | {"predict_examples": len(documents)})
    return scores


def record(model_args, data_args, args, scores):
    """Append run settings and test scores to results.jsonl beside the run directory."""
    output = Path(args.output_dir)
    settings = {**vars(model_args), "test": data_args.test_data_name or data_args.dataset_name,
                "max_seq_length": data_args.max_seq_length, "seed": args.seed, "learning_rate": args.learning_rate,
                "epochs": args.num_train_epochs, "max_steps": args.max_steps}
    scoring.record(output.parent / "results.jsonl", output.name, settings, scores)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser.set_defaults(report_to="none")
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    configure_logging(args)
    set_seed(args.seed)
    tokenizer, model = load_model(model_args, data_args.max_seq_length)
    windows, test_documents = make_windows(data_args, args, tokenizer)
    trainer = build_trainer(model, tokenizer, args, windows, data_args.eval_cnt)
    if args.do_train:
        train(trainer, windows["train"])
    if args.do_eval:
        evaluate(trainer, windows["validation"])
    if args.do_predict:
        scores = predict(trainer, test_documents, windows["test"], data_args)
        if scores:
            record(model_args, data_args, args, scores)


if __name__ == "__main__":
    main()
