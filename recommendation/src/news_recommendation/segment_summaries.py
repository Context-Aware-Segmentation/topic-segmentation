"""Summarize every predicted topic segment of the retained articles with Qwen (greedy decoding, frozen prompt).

    python -m news_recommendation.segment_summaries run --output runs/segment-summaries

The prompt asks for 15-40 words; a summary is accepted at 10-50 whitespace words. For `primary_attempts` rounds an
article's missing segments are requested together, then one by one, up to `max_attempts` rounds. Workers append every
attempt to worker-<i>/attempts.jsonl and resume from it when restarted.
"""
import argparse
import csv
import functools
import json
from pathlib import Path

from .jobs import read_jsonl, run_workers, write_jsonl
from .paths import REPOSITORY

REVISION = "Rewrite only the requested summaries using 15 to 40 words each; preserve supported meaning."
RECOVERY_REVISION = (
    "Compress the requested summary to 15 to 40 whitespace-separated words. "
    "Do not enumerate individual items from a list: describe the shared topic or central point "
    "and omit secondary details. The previous overlong answer must be shortened, not copied. "
    "Preserve supported meaning. "
    "Return only the requested segment ID."
)


def checks(text, max_words=50, min_words=10):
    words = len(text.split())
    return {"words": words, "valid": min_words <= words <= max_words}


def batch_size_for(input_tokens, segments):
    if input_tokens <= 2200 and segments <= 6:
        return 12
    if input_tokens <= 4000 and segments <= 10:
        return 8
    if input_tokens <= 7000 and segments <= 16:
        return 4
    if input_tokens <= 12000 and segments <= 28:
        return 2
    return 1


def _messages(row, targets, round_number, latest, primary_attempts, prompt):
    payload = {k: row[k] for k in ("title", "abstract", "full_body", "segments")}
    if not round_number:
        return [{"role": "user", "content": prompt + json.dumps(payload, ensure_ascii=False)}]
    revision = RECOVERY_REVISION if round_number >= primary_attempts else REVISION
    invalid = [latest[sid] for sid in targets if sid in latest and not latest[sid]["valid"]]
    payload |= {"target_segment_ids": targets, "revision_request": revision, "previous_invalid_summaries": invalid}
    return [{"role": "user", "content": prompt + json.dumps(payload, ensure_ascii=False)},
            # The repair request is repeated outside the article JSON, which is source data.
            {"role": "user", "content": revision + " Requested segment IDs: " + json.dumps(targets) +
             ". Previous invalid summaries: " + json.dumps(invalid, ensure_ascii=False) +
             ". Use all of the article context above. Return only the required JSON."}]


def parse(raw, targets, finished):
    """The summaries if `raw` is complete JSON listing exactly the requested segments in order, else the error."""
    try:
        parsed = json.loads(raw)
        assert set(parsed) == {"summaries"}
        summaries = parsed["summaries"]
        assert [s["segment_id"] for s in summaries] == targets, "Segment coverage/order mismatch"
        assert finished, "Generation cap reached"
        assert all(set(s) == {"segment_id", "summary"} and isinstance(s["summary"], str) and s["summary"].strip()
                   for s in summaries)
        return summaries, None
    except (ValueError, TypeError, KeyError, AssertionError) as exception:
        return [], f"{type(exception).__name__}: {exception}"


@functools.cache
def token_data(tokenizer):
    """lmformatenforcer's view of the vocabulary, built with its core API: its Transformers adapter imports a
    symbol removed in Transformers 5."""
    from lmformatenforcer import TokenEnforcerTokenizerData

    zero = tokenizer.encode("0", add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for token_id in range(len(tokenizer)):
        if token_id not in special:
            after_zero = tokenizer.decode([zero, token_id])[1:]
            regular.append((token_id, after_zero, len(after_zero) > len(tokenizer.decode([token_id]))))
    return TokenEnforcerTokenizerData(regular, lambda ids: tokenizer.decode(ids).rstrip("�"), tokenizer.eos_token_id)


def json_constraint(tokenizer, targets):
    """prefix_allowed_tokens_fn forcing each batch item's JSON to list exactly its target segment IDs."""
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer

    enforcers = []
    for ids in targets:
        item = {"type": "object", "additionalProperties": False, "required": ["segment_id", "summary"],
                "properties": {"segment_id": {"type": "integer", "enum": ids},
                               "summary": {"type": "string", "minLength": 1}}}
        schema = {"type": "object", "additionalProperties": False, "required": ["summaries"],
                  "properties": {"summaries": {"type": "array", "minItems": len(ids), "maxItems": len(ids),
                                               "items": item}}}
        enforcers.append(TokenEnforcer(token_data(tokenizer), JsonSchemaParser(schema)))
    return lambda batch_id, token_ids: enforcers[batch_id].get_allowed_tokens(token_ids.tolist())


def prepare(config, output):
    """Copy the config and prompt into `output` and split the retained articles into cost-balanced shards."""
    with (REPOSITORY / config["dataset"] / "news_body_metadata.csv").open() as stream:
        tokens = {row["body_source_id"]: int(row["body_bert_tokens"]) for row in csv.DictReader(stream)}
    rows = [{k: row[k] for k in ("news_id", "title", "abstract", "full_body", "segments")} | {"catalog_position": position}
            for position, row in enumerate(read_jsonl(REPOSITORY / config["source"])) if row["news_id"] in tokens]
    shards, costs = [[] for _ in config["gpu_ids"]], [0] * len(config["gpu_ids"])
    for row in sorted(rows, key=lambda r: (-(2 * tokens[r["news_id"]] + 128 * len(r["segments"])), r["news_id"])):
        worker_id = costs.index(min(costs))
        shards[worker_id].append(row)
        costs[worker_id] += 2 * tokens[row["news_id"]] + 128 * len(row["segments"])
    output.mkdir(parents=True)
    for worker_id, shard in enumerate(shards):
        write_jsonl(output / f"shard-{worker_id}.jsonl", sorted(shard, key=lambda r: r["news_id"]))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "prompt.txt").write_text((REPOSITORY / config["prompt"]).read_text())


def record(state, attempt, config):
    state["round"], state["last"] = attempt["round"] + 1, attempt
    state["seen"].add((attempt["round"], tuple(attempt["targets"])))
    for summary in attempt["summaries"]:
        state["latest"][summary["segment_id"]] = summary | checks(summary["summary"], config["max_words"], config["min_words"])
        if state["latest"][summary["segment_id"]]["valid"]:
            state["accepted"][summary["segment_id"]] = summary


def missing(state):
    return [s["segment_id"] for s in state["row"]["segments"] if s["segment_id"] not in state["accepted"]]


def load_states(output, worker_id, config):
    """Each article of the shard with the summaries accepted in the worker's recorded attempts."""
    states = {row["news_id"]: {"row": row, "accepted": {}, "latest": {}, "round": 0, "last": None, "seen": set()}
              for row in read_jsonl(output / f"shard-{worker_id}.jsonl")}
    attempts = output / f"worker-{worker_id}/attempts.jsonl"
    if attempts.exists():
        for attempt in read_jsonl(attempts):
            record(states[attempt["news_id"]], attempt, config)
    return states


def worker(output, worker_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((output / "config.json").read_text())
    prompt = (output / "prompt.txt").read_text()
    (output / f"worker-{worker_id}").mkdir(exist_ok=True)
    states = load_states(output, worker_id, config)
    device = f"cuda:{config['gpu_ids'][worker_id]}"
    tokenizer = AutoTokenizer.from_pretrained(REPOSITORY / config["model"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(REPOSITORY / config["model"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()

    def request(state, targets, round_number):
        messages = _messages(state["row"], targets, round_number, state["latest"], config["primary_attempts"], prompt)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return {"state": state, "targets": targets, "text": text, "length": len(tokenizer.encode(text))}

    def generate(batch, round_number):
        inputs = tokenizer([r["text"] for r in batch], padding=True, return_tensors="pt").to(device)
        options = {}
        # After a malformed answer, constrain the JSON structure of the retry.
        if round_number and any(r["state"]["last"]["error"] for r in batch):
            options["prefix_allowed_tokens_fn"] = json_constraint(tokenizer, [r["targets"] for r in batch])
        cap = min(4096, max(512, 128 + 96 * max(len(r["targets"]) for r in batch)))
        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=cap,
                                            pad_token_id=tokenizer.pad_token_id, **options)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            generate(batch[:len(batch) // 2], round_number)
            generate(batch[len(batch) // 2:], round_number)
            return
        attempts = []
        for item, tokens in zip(batch, output_ids[:, inputs.input_ids.shape[1]:]):
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            summaries, error = parse(raw, item["targets"], tokenizer.eos_token_id in tokens.tolist())
            attempts.append({"news_id": item["state"]["row"]["news_id"], "round": round_number,
                             "targets": item["targets"], "summaries": summaries, "error": error})
            record(item["state"], attempts[-1], config)
        write_jsonl(output / f"worker-{worker_id}/attempts.jsonl", attempts, mode="a")

    for round_number in range(config["max_attempts"]):
        recovery = round_number >= config["primary_attempts"]
        pending = []
        for state in states.values():
            if missing(state) and (recovery or state["round"] <= round_number):
                groups = [[sid] for sid in missing(state)] if recovery else [missing(state)]
                pending += [request(state, group, round_number) for group in groups
                            if (round_number, tuple(group)) not in state["seen"]]
        pending.sort(key=lambda r: (r["length"], r["state"]["row"]["news_id"], r["targets"]))
        position = 0
        while position < len(pending):
            batch = pending[position:position + batch_size_for(pending[position]["length"], len(pending[position]["targets"]))]
            while len(batch) > 1 and batch[-1]["length"] > max(1.35 * batch[0]["length"], batch[0]["length"] + 512):
                batch = batch[:-1]
            generate(batch, round_number)
            position += len(batch)
    unresolved = sum(bool(missing(state)) for state in states.values())
    if unresolved:
        raise RuntimeError(f"{unresolved} articles still have segments without a valid summary")


def finalize(output):
    """Write outputs.jsonl: every article's accepted summaries, in catalog order."""
    config = json.loads((output / "config.json").read_text())
    states = [state for worker_id in range(len(config["gpu_ids"]))
              for state in load_states(output, worker_id, config).values()]
    write_jsonl(output / "outputs.jsonl", (
        {"news_id": s["row"]["news_id"], "catalog_position": s["row"]["catalog_position"],
         "parsed": {"summaries": [s["accepted"][segment["segment_id"]] for segment in s["row"]["segments"]]}}
        for s in sorted(states, key=lambda s: s["row"]["catalog_position"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("run", "prepare", "worker", "finalize"))
    parser.add_argument("--output", type=Path, required=True, help="Run directory, relative to recommendation/")
    parser.add_argument("--config", default="configs/section_summary_40_50_full.json")
    parser.add_argument("--dataset", help="Prepared-data directory whose articles are summarized")
    parser.add_argument("--source", help="Segmented article catalog")
    parser.add_argument("--gpus", type=int, nargs="+", help="One worker per GPU")
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    output = REPOSITORY / args.output
    if args.stage in ("run", "prepare"):
        config = json.loads((REPOSITORY / args.config).read_text())
        config |= {key: value for key, value in (("dataset", args.dataset), ("source", args.source),
                                                 ("gpu_ids", args.gpus)) if value is not None}
        prepare(config, output)
    if args.stage == "run":
        run_workers("news_recommendation.segment_summaries", output, len(config["gpu_ids"]))
    if args.stage == "worker":
        worker(output, args.worker)
    if args.stage in ("run", "finalize"):
        finalize(output)


if __name__ == "__main__":
    main()
