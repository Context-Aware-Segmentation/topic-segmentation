"""Summarize each retained article as a whole with Qwen (greedy decoding, frozen prompt).

    python -m news_recommendation.article_summaries run --output runs/article-summaries

The configuration uses two GPUs. Ordinary workers skip inputs over 18,000 tokens. The `long` stage splits the layers over
two GPUs for every article still missing, and `recovery` retries what remains with a larger output cap. A completion
must end with EOS; one that hits the cap is retried alone with twice the cap. Each stage appends to
<stage>/outputs.jsonl and skips the articles already there.
"""
import argparse
import csv
import json
from pathlib import Path

from .jobs import read_jsonl, run_workers, write_jsonl
from .paths import REPOSITORY

LONG_INPUT = 18000


def batches(prepared, single=False):
    """Consecutive length-sorted inputs, fewer per batch as they grow, never mixing very different lengths."""
    result, position = [], 0
    while position < len(prepared):
        length = prepared[position][0]
        size = 1 if single else 8 if length <= 2200 else 4 if length <= 6000 else 2 if length <= 12000 else 1
        batch = prepared[position:position + size]
        while len(batch) > 1 and batch[-1][0] > max(batch[0][0] * 1.35, batch[0][0] + 512):
            batch = batch[:-1]
        result.append(batch)
        position += len(batch)
    return result


def prepare(config, output):
    """Copy the config and prompt into `output` and deal the retained articles into one shard per GPU."""
    with (REPOSITORY / config["dataset"] / "news_body_metadata.csv").open() as stream:
        retained = {row["body_source_id"] for row in csv.DictReader(stream)}
    rows = [{k: row[k] for k in ("news_id", "title", "abstract", "full_body")} | {"catalog_position": position}
            for position, row in enumerate(read_jsonl(REPOSITORY / config["source"])) if row["news_id"] in retained]
    output.mkdir(parents=True)
    workers = len(config["gpu_ids"])
    for worker_id in range(workers):
        write_jsonl(output / f"shard-{worker_id}.jsonl", rows[worker_id::workers])
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "prompt.txt").write_text((REPOSITORY / config["prompt"]).read_text())


def worker(output, stage, worker_id=0):
    """Summarize one shard (stage "worker"), or every article still missing (stages "long" and "recovery")."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((output / "config.json").read_text())
    prompt = (output / "prompt.txt").read_text()
    folder = output / (f"worker-{worker_id}" if stage == "worker" else stage)
    folder.mkdir(exist_ok=True)
    if stage == "worker":
        rows = read_jsonl(output / f"shard-{worker_id}.jsonl")
        saved = folder / "outputs.jsonl"
        done = {row["news_id"] for row in read_jsonl(saved)} if saved.exists() else set()
    else:
        rows = (row for i in range(len(config["gpu_ids"])) for row in read_jsonl(output / f"shard-{i}.jsonl"))
        done = {row["news_id"] for path in output.glob("*/outputs.jsonl") for row in read_jsonl(path)}
    model_path = REPOSITORY / config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    prepared = []
    for row in rows:
        if row["news_id"] not in done:
            payload = {key: row[key] for key in ("title", "abstract", "full_body")}
            messages = [{"role": "user", "content": prompt + json.dumps(payload, ensure_ascii=False)}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            length = len(tokenizer.encode(text))
            if stage != "worker" or length <= LONG_INPUT:
                prepared.append((length, row, text))
    prepared.sort(key=lambda item: (item[0], item[1]["news_id"]))
    if not prepared:
        return
    cap = config["recovery_max_new_tokens"] if stage == "recovery" else config["max_new_tokens"]
    if stage == "long":
        # Embeddings, the first half of the layers, and the head on one GPU; the second half on the other.
        first, second = config["gpu_ids"][:2]
        layers = json.loads((model_path / "config.json").read_text())["num_hidden_layers"]
        device_map = {"model.embed_tokens": first, "model.norm": first, "lm_head": first}
        device_map |= {f"model.layers.{i}": second if i >= layers // 2 else first for i in range(layers)}
        device = f"cuda:{first}"
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map=device_map).eval()
    else:
        device = f"cuda:{config['gpu_ids'][worker_id]}"
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").to(device).eval()

    def generate(batch, token_cap):
        inputs = tokenizer([item[2] for item in batch], padding=True, return_tensors="pt").to(device)
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=token_cap, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
        except torch.OutOfMemoryError:
            # A single input that does not fit is left to the next stage.
            torch.cuda.empty_cache()
            if len(batch) > 1:
                generate(batch[:len(batch) // 2], token_cap)
                generate(batch[len(batch) // 2:], token_cap)
            return
        outputs, unfinished = [], []
        for item, tokens in zip(batch, generated[:, inputs.input_ids.shape[1]:]):
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if text and (tokens == tokenizer.eos_token_id).any():
                outputs.append({"news_id": item[1]["news_id"], "catalog_position": item[1]["catalog_position"],
                                "summary": text})
            else:
                unfinished.append(item)
        write_jsonl(folder / "outputs.jsonl", outputs, mode="a")
        if token_cap == cap:
            for item in unfinished:
                generate([item], 2 * cap)

    for batch in batches(prepared, single=stage == "long"):
        generate(batch, cap)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("run", "prepare", "worker", "long", "recovery"))
    parser.add_argument("--output", type=Path, required=True, help="Run directory, relative to recommendation/")
    parser.add_argument("--config", default="configs/article_summary.json")
    parser.add_argument("--dataset", help="Prepared-data directory whose articles are summarized")
    parser.add_argument("--source", help="Segmented article catalog")
    parser.add_argument("--gpus", type=int, nargs=2, help="Two GPUs: separate workers, then shared for the long stage")
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    output = REPOSITORY / args.output
    if args.stage in ("run", "prepare"):
        config = json.loads((REPOSITORY / args.config).read_text())
        config |= {key: value for key, value in (("dataset", args.dataset), ("source", args.source),
                                                 ("gpu_ids", args.gpus)) if value is not None}
        prepare(config, output)
    if args.stage == "run":
        run_workers("news_recommendation.article_summaries", output, len(config["gpu_ids"]))
        worker(output, "long")
        worker(output, "recovery")
    if args.stage in ("worker", "long", "recovery"):
        worker(output, args.stage, args.worker)


if __name__ == "__main__":
    main()
