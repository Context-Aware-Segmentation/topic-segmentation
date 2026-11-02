"""Pre-train ModernBERT on C4 paragraph boundaries.

Label 1 marks the last token of a paragraph. Settings come from
configs/boundary_pretraining.json. Launch with scripts/run_boundary_pretraining.py;
training resumes from the last checkpoint in --output-dir.
"""
import argparse
import json
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from topic_segmentation.data import build_c4_boundaries
from topic_segmentation.paths import REPOSITORY

CONFIG = json.loads((REPOSITORY / "configs/boundary_pretraining.json").read_text())
SMOKE_EXAMPLES = 100_000
LABELS = ["O", "separator"]


class SaveEachEpoch(TrainerCallback):
    """Save the weights at the end of every epoch, independent of save_total_limit."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero:
            directory = Path(args.output_dir) / f"epoch-{round(state.epoch)}"
            model.save_pretrained(directory)
            self.tokenizer.save_pretrained(directory)


def boundary_scores(prediction):
    """Token-level precision, recall, and F1 of the paragraph-end class."""
    keep = prediction.label_ids != -100
    predicted, gold = prediction.predictions.argmax(-1)[keep] == 1, prediction.label_ids[keep] == 1
    precision, recall, f1, _ = precision_recall_fscore_support(gold, predicted, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def load_model(base, seed, attention):
    """Load `base` with a new token-classification head; seeding first fixes the head's initialization."""
    set_seed(seed)
    return AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        attn_implementation=attention, torch_dtype=torch.bfloat16)


def training_arguments(training, output_dir, run_name):
    """Build Trainer arguments from the training configuration."""
    return TrainingArguments(
        output_dir=output_dir,
        seed=training["seed"],
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=training["per_device_batch_size"],
        per_device_eval_batch_size=training["per_device_batch_size"],
        learning_rate=training["learning_rate"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        lr_scheduler_type=training["scheduler"],
        bf16=training["precision"] == "bfloat16",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        disable_tqdm=False,
        log_level="warning",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        run_name=run_name,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true", help=f"Train on the first {SMOKE_EXAMPLES:,} examples only")
    args = parser.parse_args()
    training = CONFIG["training"]
    splits = build_c4_boundaries.load_splits(REPOSITORY / CONFIG["prepared_dataset"])
    train = splits["train"]
    if args.smoke_test:
        train = train.select(range(min(SMOKE_EXAMPLES, len(train))))
    base = REPOSITORY / CONFIG["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = load_model(base, training["seed"], training["attention_implementation"])
    run_name = "modernbert_c4_smoke" if args.smoke_test else "modernbert_c4_full"
    trainer = Trainer(model=model, args=training_arguments(training, str(args.output_dir), run_name),
                      train_dataset=train, eval_dataset=splits["validation"],
                      data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
                      compute_metrics=boundary_scores, callbacks=[SaveEachEpoch(tokenizer)])
    trainer.train(resume_from_checkpoint=get_last_checkpoint(args.output_dir) if args.output_dir.exists() else None)
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
