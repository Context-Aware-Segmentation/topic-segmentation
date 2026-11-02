"""Train PLM-NR on one article representation of the MIND-large day split, and evaluate it.

    torchrun --nproc_per_node 4 -m news_recommendation.training train title_section_summary_512 --output-dir runs/x
    python -m news_recommendation.training evaluate --output-dir runs/x

configs/plm_nr.json holds the settings shared by every representation.
`evaluate` scores the epoch with the best validation AUC on the test split.
"""
import argparse
from datetime import timedelta
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import BertConfig, BertModel

from .model import ModelBert
from .paths import REPOSITORY

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def run_config(variant, data=None):
    settings = json.loads((REPOSITORY / "configs/plm_nr.json").read_text())
    if variant not in settings["inputs"]:
        raise KeyError(variant)
    config = {key: value for key, value in settings.items() if key != "inputs"}
    return config | {"input": variant} | ({"data": data} if data else {})


def read_impressions(path):
    """(impression id, history, candidate news ids, click labels) per line of a behaviors file."""
    rows = []
    with path.open() as stream:
        for line in stream:
            key, _, _, history, candidates = line.rstrip("\n").split("\t")
            pairs = [item.rsplit("-", 1) for item in candidates.split()]
            rows.append((key, history.split(), [news for news, _ in pairs], [int(label) for _, label in pairs]))
    return rows


def news_inputs(config, rows):
    """Row index of every news id the rows use (0 is the all-zero padding row) and the packed token rows."""
    data = np.load(REPOSITORY / config["data"] / "inputs" / config["input"] / "features.npz")
    positions = {key: i for i, key in enumerate(data["keys"].tolist())}
    keys = sorted({key for _, history, candidates, _ in rows for key in history[-config["user_log_length"]:] + candidates})
    packed = torch.from_numpy(data["packed"][[positions[key] for key in keys]].astype(np.int64))
    return {key: i + 1 for i, key in enumerate(keys)}, torch.cat([torch.zeros_like(packed[:1]), packed])


def padded_history(history, index, length):
    """The last `length` clicks, left-padded with row 0, and their mask."""
    ids = [index[key] for key in history[-length:]]
    return [0] * (length - len(ids)) + ids, [0.0] * (length - len(ids)) + [1.0] * len(ids)


def build_model(config, pretrained=True):
    """PLM-NR on the first `bert_layers` BERT layers, of which only `trainable_bert_layers` are trained."""
    bert_config = BertConfig.from_pretrained(REPOSITORY / config["bert"], num_hidden_layers=config["bert_layers"])
    bert = BertModel.from_pretrained(REPOSITORY / config["bert"], config=bert_config) if pretrained else BertModel(bert_config)
    for name, parameter in bert.named_parameters():
        parameter.requires_grad = any(name.startswith(f"encoder.layer.{i}.") for i in config["trainable_bert_layers"])
    return ModelBert(config, bert)


def train_epoch(model, rows, index, packed, config, optimizer, device):
    """One epoch over every clicked candidate with `npratio` sampled negatives."""
    examples = [(row, candidate) for row in rows for candidate, label in zip(row[2], row[3]) if label]
    random.shuffle(examples)
    packed = packed.to(device)
    distributed = torch.distributed.is_initialized()
    rank, world = (torch.distributed.get_rank(), torch.distributed.get_world_size()) if distributed else (0, 1)
    model.train()
    for offset in range(0, len(examples), config["batch_size"]):
        batch = examples[offset:offset + config["batch_size"]]
        indices = []
        for row, positive in batch:
            negatives = [index[key] for key, label in zip(row[2], row[3]) if not label]
            # As upstream: too few negatives are padded by repeating the first.
            if len(negatives) < config["npratio"]:
                sampled = negatives + [negatives[0]] * (config["npratio"] - len(negatives))
            else:
                sampled = random.sample(negatives, config["npratio"])
            history, mask = padded_history(row[1], index, config["user_log_length"])
            indices.append([index[positive]] + sampled + history + mask)
        indices = torch.tensor(indices, dtype=torch.long).to(device)
        candidates, history_end = config["npratio"] + 1, config["npratio"] + 1 + config["user_log_length"]
        candidate_ids, history_ids = indices[:, :candidates], indices[:, candidates:history_end]
        mask = indices[:, history_end:].float()
        targets = torch.zeros(len(batch), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        # Each process takes a contiguous shard of the batch, in micro-batches.
        shard = -(-len(batch) // world)
        micro = config["micro_batch_size"]
        for start in range(rank * shard, (rank + 1) * shard, micro):
            end = min(start + micro, (rank + 1) * shard, len(batch))
            # Every process runs the same passes; one past the batch end reuses example 0 with zero weight.
            weight = world if end > start else 0
            start, end = (start, end) if end > start else (0, 1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = model(packed[candidate_ids[start:end]], packed[history_ids[start:end]], mask[start:end],
                             targets[start:end])
                loss = loss.mean() * ((end - start) * weight / len(batch))
            loss.backward()
        optimizer.step()


def ndcg(labels, scores, k):
    def dcg(order):
        gains = 2 ** np.take(labels, order[:k]) - 1
        return np.sum(gains / np.log2(np.arange(len(gains)) + 2))
    return dcg(np.argsort(scores)[::-1]) / dcg(np.argsort(labels)[::-1])


def mrr(labels, scores):
    ranked = np.take(labels, np.argsort(scores)[::-1])
    return np.sum(ranked / (np.arange(len(ranked)) + 1)) / np.sum(ranked)


@torch.no_grad()
def evaluate(model, rows, index, packed, config, device):
    """Mean AUC, MRR, and nDCG@5/10 over the prepared impressions, each with both click labels."""
    model.eval()
    # Row 0 is encoded too: the padded history slots use the all-zero news row.
    vectors = torch.cat([model.news_encoder(chunk.to(device)) for chunk in packed.split(config["news_encoding_batch_size"])])
    totals = np.zeros(4)
    for offset in range(0, len(rows), config["evaluation_batch_size"]):
        batch = rows[offset:offset + config["evaluation_batch_size"]]
        histories = [padded_history(history, index, config["user_log_length"])[0] for _, history, _, _ in batch]
        users = model.user_encoder(vectors[torch.tensor(histories, device=device)])
        candidates = torch.tensor([index[key] for _, _, news, _ in batch for key in news], device=device)
        owners = torch.tensor([owner for owner, (_, _, news, _) in enumerate(batch) for _ in news], device=device)
        scores = (vectors[candidates] * users[owners]).sum(-1).cpu().numpy()
        ends = np.cumsum([len(news) for _, _, news, _ in batch])
        for (_, _, _, labels), row_scores in zip(batch, np.split(scores, ends[:-1])):
            labels = np.asarray(labels)
            totals += [roc_auc_score(labels, row_scores), mrr(labels, row_scores),
                       ndcg(labels, row_scores, 5), ndcg(labels, row_scores, 10)]
    return dict(zip(METRICS, (totals / len(rows)).tolist()))


def train(config, run, device):
    if any((run / name).exists() for name in ("epochs.json", "test.json")) or any(run.glob("epoch*.pt")):
        raise FileExistsError(f"Training artifacts already exist in {run}; choose a new run directory")
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=12))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    main_process = not distributed or torch.distributed.get_rank() == 0
    data = REPOSITORY / config["data"]
    rows = read_impressions(data / "train.tsv")
    validation = read_impressions(data / "validation.tsv")
    index, packed = news_inputs(config, rows + validation)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
    trained = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if distributed else model
    epochs = []
    run.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["epochs"] + 1):
        train_epoch(trained, rows, index, packed, config, optimizer, device)
        if main_process:
            epochs.append({"epoch": epoch} | evaluate(model, validation, index, packed, config, device))
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, run / f"epoch{epoch}.pt")
            (run / "epochs.json").write_text(json.dumps(epochs, indent=2) + "\n")
            print(json.dumps(epochs[-1]), flush=True)
        if distributed:
            torch.distributed.barrier()
    if distributed:
        torch.distributed.destroy_process_group()


def test(run, device):
    """Score the epoch with the best validation AUC on the test split into test.json."""
    output = run / "test.json"
    if output.exists():
        raise FileExistsError(f"Evaluation result already exists: {output}")
    best = max(json.loads((run / "epochs.json").read_text()), key=lambda row: row["auc"])["epoch"]
    state = torch.load(run / f"epoch{best}.pt", map_location="cpu", weights_only=True)
    model = build_model(state["config"], pretrained=False).to(device)
    model.load_state_dict(state["model"])
    rows = read_impressions(REPOSITORY / state["config"]["data"] / "test.tsv")
    index, packed = news_inputs(state["config"], rows)
    report = evaluate(model, rows, index, packed, state["config"], device) | {"epoch": best}
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    training = modes.add_parser("train")
    training.add_argument("input", help="Article representation listed in `inputs` in configs/plm_nr.json")
    training.add_argument("--data", help="Prepared-data directory (default: the config's)")
    for mode in (training, modes.add_parser("evaluate")):
        mode.add_argument("--output-dir", required=True, help="Run directory, relative to recommendation/")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = REPOSITORY / args.output_dir
    if args.mode == "train":
        config = run_config(args.input, args.data)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        train(config, run, torch.device("cuda"))
    else:
        test(run, torch.device("cuda"))


if __name__ == "__main__":
    main()
