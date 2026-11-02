"""Train ATLOP or DREEAM on the gold-entity DWIE inputs and keep the checkpoint with the best development F1.

python -m knowledge_graph.news.train --implementation atlop --inputs DOCRED --out-dir MODELS
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from knowledge_graph.common import EarlyStopping, load_json, precision_recall_f1
from knowledge_graph.news.docre import IMPLEMENTATIONS, build_model, infer, model_inputs, predicted_triples
from knowledge_graph.news.docred import add_evidence_targets, add_relation_targets, collate_fn, prepare_features
from knowledge_graph.news.inputs import ENCODER, MAX_LENGTH

SEED = 42
EPOCHS = 10
BATCH_SIZE, EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION = 2, 4, 2
ENCODER_LR, HEAD_LR = 2e-5, 1e-4
EVIDENCE_WEIGHT = 0.1  # DREEAM evidence loss


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def label_triples(documents: list[dict]) -> set[tuple]:
    return {
        (doc["title"], label["h"], label["r"], label["t"]) for doc in documents for label in doc["labels"]
    }


def training_loss(outputs) -> torch.Tensor:
    losses = outputs["loss"]
    return losses["rel_loss"] + EVIDENCE_WEIGHT * losses.get("evi_loss", 0.0)


def train_epoch(model, loader, optimizer, scheduler, scaler) -> None:
    """Run one epoch with the experiment's mixed precision and gradient accumulation."""
    model.train()
    device = next(model.parameters()).device
    for step, batch in enumerate(loader, start=1):
        group_start = ((step - 1) // GRADIENT_ACCUMULATION) * GRADIENT_ACCUMULATION
        group_size = min(GRADIENT_ACCUMULATION, len(loader) - group_start)
        inputs = model_inputs(batch, device)
        inputs["labels"] = batch["labels"].to(device)
        if "sent_labels" in batch:
            inputs.update(sent_pos=batch["sent_pos"], sent_labels=batch["sent_labels"].to(device))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
            loss = training_loss(outputs)
        scaler.scale(loss / group_size).backward()
        if step % GRADIENT_ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)


def train(args: argparse.Namespace) -> None:
    """Train from RoBERTa, keeping this model's best checkpoint."""
    inputs = args.inputs.resolve()
    run_dir = (args.out_dir / args.implementation).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda")

    rel2id = load_json(inputs / "rel2id.json")
    id2rel = {value: key for key, value in rel2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    train_documents = load_json(inputs / "train.json")
    train_features = prepare_features(train_documents, tokenizer, MAX_LENGTH)
    add_relation_targets(train_features, train_documents, rel2id)
    if args.implementation == "dreeam":
        add_evidence_targets(train_features, train_documents)
    dev_documents = load_json(inputs / "dev.json")
    dev_features = prepare_features(dev_documents, tokenizer, MAX_LENGTH)
    model = build_model(args.implementation, tokenizer, len(rel2id))
    model.to(device)

    added_names = ("extractor", "bilinear")
    encoder_params, head_params = [], []
    for name, parameter in model.named_parameters():
        (head_params if any(part in name for part in added_names) else encoder_params).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": ENCODER_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        eps=1e-6,
    )
    train_loader = DataLoader(
        train_features,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    updates_per_epoch = int(np.ceil(len(train_loader) / GRADIENT_ACCUMULATION))
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * 0.06), total_updates)
    scaler = torch.amp.GradScaler("cuda")
    selection = EarlyStopping(patience=None)  # every epoch runs; the best dev F1 is kept
    dev_gold = label_triples(dev_documents)
    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.epochs):
        train_epoch(model, train_loader, optimizer, scheduler, scaler)
        dev_array = infer(model, dev_features, EVAL_BATCH_SIZE)
        dev_pred = predicted_triples(dev_array, dev_features, id2rel)
        metrics = precision_recall_f1(dev_gold, dev_pred)
        selection.update(metrics["f1"], lambda: torch.save(model.state_dict(), run_dir / "best.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=list(IMPLEMENTATIONS), required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Gold-entity DocRED inputs (train/dev/rel2id).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Receives IMPLEMENTATION/best.pt.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs (default: 10).")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
