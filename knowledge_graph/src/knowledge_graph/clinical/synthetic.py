"""Synthetic-100 note generation: local Qwen case cards from grouped MIMIC references, then two GPT-5.6-terra agent stages.

    python -m knowledge_graph.clinical.synthetic cards      # 500 private references -> 100 Qwen case cards (local GPUs)
    python -m knowledge_graph.clinical.synthetic expand     # case card -> fictional discharge note
    python -m knowledge_graph.clinical.synthetic reformat   # note -> MIMIC-style blocks and sections

Only case cards and synthetic text reach the agents; source notes and their IDs stay in the local stage.
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from knowledge_graph.clinical.agents import agent, bullet_feedback, run
from knowledge_graph.common import load_json, read_jsonl, write_json, write_jsonl
from knowledge_graph.paths import MIMIC_DATA, REPOSITORY, SYNTHETIC100

POOL = MIMIC_DATA / "finetune_sets/discharge_bhc_seed42_n101000/splits_valid1000_seed42/ids/train_100000.txt"
SOURCE_CACHES = (
    MIMIC_DATA / "finetune_sets/discharge_bhc_seed42_n5000/discharge_random5000_by_id.json",
    MIMIC_DATA / "finetune_sets/discharge_bhc_seed42_n101000/discharge_random96000_by_id.json",
)
QWEN = REPOSITORY / "checkpoints/teachers/qwen3-32b-awq"
OUT = SYNTHETIC100 / "generation"  # private MIMIC reference groups stay in OUT/private/
FORMATTED = "terra_mimic_formatted"
SAMPLES, GROUP_SIZE, SEED = 500, 5, 20260809  # 100 groups of five reference notes
WORD = re.compile(r"\b\w+\b")
BLOCK = re.compile(r"\n\s*\n\s*\n+")
HEADING = re.compile(r"(?m)^[A-Za-z][A-Za-z0-9 /&(),.'+\-]{1,100}:\s*$")
MARKDOWN = re.compile(r"(?m)^\s*(?:#{1,6}\s|```|---+\s*$)|\*\*[^*]+\*\*")

CARD_SYSTEM = """You design synthetic clinical NLP benchmark data.
The five private examples are deidentified discharge-note references. Infer only
aggregate structure, style, length, section ordering, and broad clinical-domain
patterns. Create one new fictional case. Never reproduce a sentence, unusual phrase,
exact event sequence, identifier, date, dosage, age, or rare combination from an
example. Do not summarize any individual source note. Return JSON only."""
EXPAND = """Read input.json. It contains exactly one fictional case card created
by a local Qwen model from aggregate structural patterns. You have no access to
any original clinical note.

Expand the case card into one fictional MIMIC-like discharge note. Follow its
target_words, section_plan, format_features, and clinical facts. Preserve a terse
clinical register, realistic headings, lists, and line breaks. Keep the note
internally coherent. Use ___ for all names, identifiers, and dates. Do not invent
a real name, address, phone number, record number, or exact calendar date. Do not
include a disclaimer or commentary inside the note.

Return exactly the JSON required by schema.json."""
REFORMAT = r"""Read input.json. It contains one entirely fictional case card and one
synthetic discharge-note draft. It contains no original MIMIC note or source ID.

Rewrite the draft as one long, internally consistent MIMIC-style discharge
summary while preserving the supplied clinical facts. Do not introduce a new
diagnosis, procedure, complication, patient identity, or exact calendar date.

Formatting requirements are part of the task:

- Return plain clinical text only inside the schema field. Never use Markdown,
  asterisks for headings, hash headings, horizontal rules, or code fences.
- Use ___ for Name, Unit No, dates, record numbers, clinicians, institutions,
  addresses, and telephone numbers.
- Produce at least minimum_words and no more than maximum_words. Do not stop
  early merely because every section has been mentioned.
- Create target_blocks coarse blocks. Separate consecutive blocks with exactly
  two blank lines (the text therefore contains three newline characters).
- Within each block, use several explicit section headings. Put each heading on
  its own line in the form `Heading:` and put its content on following lines.
- Aim for target_sections headings across the note.
- Use short wrapped clinical lines, numbered medication lists where appropriate,
  and terse discharge-summary prose.
- Include realistic front matter and organize the remaining blocks around:
  admission/history, background history, examination/results, hospital course
  with problem-oriented subsections, medications, condition/disposition, and
  instructions/follow-up. Merge adjacent categories when target_blocks is lower.
- Do not mention that the note is synthetic and do not write MIMIC, MIMIC-III,
  MIMIC-IV, or a hospital name.

Return exactly the JSON required by schema.json and preserve synthetic_id."""
CARD_FIELDS = [
    "service",
    "patient_profile",
    "admission_scenario",
    "primary_conditions",
    "course_events",
    "section_plan",
]
TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["synthetic_id", "text"],
    "properties": {"synthetic_id": {"type": "string"}, "text": {"type": "string"}},
}


# Stage 1: Qwen case cards from groups of five private reference notes.


def source_groups(out):
    """Sampled training-pool notes in groups of five; this private file never leaves the local stage."""
    path = out / "private" / "source_groups.private.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["groups"]
    ids = [line.strip() for line in POOL.read_text(encoding="utf-8").splitlines() if line.strip()]
    sampled = random.Random(SEED).sample(ids, SAMPLES)
    groups = [{"group_id": f"g{start // GROUP_SIZE:03d}", "source_note_ids": sampled[start:start + GROUP_SIZE]}
              for start in range(0, SAMPLES, GROUP_SIZE)]
    write_json(path, {"groups": groups})
    return groups


def source_notes(ids):
    """Texts of the requested MIMIC notes, streamed from the two large note caches."""
    import ijson

    found = {}
    for cache in SOURCE_CACHES:
        with cache.open("rb") as handle:
            for note_id, note in ijson.kvitems(handle, "notes"):
                if note_id in ids:
                    found[note_id] = note
    return found


def note_text(note):
    return str(note.get("text") or "")


def redact(source):
    """Mask bracketed placeholders and long numbers, and keep only the start and end of very long notes."""
    source = re.sub(r"\[\*\*.*?\*\*\]", "___", source, flags=re.DOTALL)
    source = re.sub(r"\b\d{6,}\b", "___", source)
    if len(source) <= 16_000:
        return source
    return source[:10_000] + "\n\n[LONG MIDDLE OMITTED FOR LOCAL STYLE INFERENCE]\n\n" + source[-6_000:]


def target_words(source):
    """Word count clipped to 150-3,500 and rounded to a multiple of 50."""
    return int(round(min(max(len(WORD.findall(source)), 150), 3_500) / 50.0) * 50)


def card_prompt(group, notes):
    examples, targets = [], []
    for index, note_id in enumerate(group["source_note_ids"], start=1):
        targets.append(target_words(note_text(notes[note_id])))
        examples.append(f'<private_example index="{index}">\n{redact(note_text(notes[note_id]))}\n</private_example>')
    return f"""Create exactly one fictional case card inspired only by the aggregate
formatting and distribution of all five private examples below.

Mandatory novelty rules:
- Do not select, preserve, combine, or retell any individual source case.
- Use new broad diagnoses, comorbidity combinations, procedures, and course events.
- Avoid rare proper nouns and all exact numeric clinical values from the examples.
- Case cards must contain facts, not prose copied from a discharge note.
- The five reference lengths are {targets}; use their median, {int(statistics.median(targets))} words,
  as the single synthetic note's target length.
- Use this synthetic ID: syn-{group["group_id"]}.

Return one JSON object with key "cases" containing exactly one case. The case must contain:
synthetic_id, target_words, service, patient_profile (age_band and sex),
admission_scenario, primary_conditions, comorbidities, procedures,
course_events, discharge_medication_classes, follow_up,
section_plan (a list of heading, target_words, and style objects), and
format_features. Use arrays for all plural fields.

{chr(10).join(examples)}

/no_think"""


def parse_card(raw, group, notes):
    """The single case of a Qwen answer, with its ID, group, and median reference length set by code."""
    raw = raw.rsplit("</think>", 1)[-1].strip()
    if raw.startswith("```"):
        raw = re.sub(r"\s*```$", "", re.sub(r"^```(?:json|text)?\s*", "", raw)).strip()
    cases = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])["cases"]
    case = cases[0]
    if len(cases) != 1 or any(field not in case for field in CARD_FIELDS):
        raise ValueError("expected exactly one case with every card field")
    case["synthetic_id"] = f"syn-{group['group_id']}"
    case["target_words"] = int(statistics.median(target_words(note_text(notes[i])) for i in group["source_note_ids"]))
    case["group_id"] = group["group_id"]
    return [case]


def cards(args):
    out = args.out.resolve()
    path = out / "qwen_case_cards.jsonl"
    done = {row["group_id"]: row for row in read_jsonl(path)} if path.exists() else {}
    pending = [group for group in source_groups(out) if group["group_id"] not in done]
    if not pending:
        return 0
    from vllm import LLM, SamplingParams

    notes = source_notes({note_id for group in pending for note_id in group["source_note_ids"]})
    parallel = args.tensor_parallel_size
    llm = LLM(
        model=str(args.model),
        quantization="awq_marlin",
        tensor_parallel_size=parallel,
        gpu_memory_utilization=0.90,
        max_model_len=32_768,
        max_num_seqs=6,
        enable_prefix_caching=True,
        disable_custom_all_reduce=parallel > 1,
        enforce_eager=parallel > 1,
    )
    params = SamplingParams(temperature=0.55, top_p=0.9, max_tokens=3_000)
    total = len(done) + len(pending)
    chat = lambda batch: llm.chat(
        [
            [{"role": "system", "content": CARD_SYSTEM}, {"role": "user", "content": card_prompt(g, notes)}]
            for g in batch
        ],
        sampling_params=params,
    )
    for start in range(0, len(pending), 2):
        batch, rows = pending[start : start + 2], []
        for group, output in zip(batch, chat(batch)):
            for attempt in range(3):  # resample invalid JSON up to twice
                try:
                    case = parse_card(output.outputs[0].text, group, notes)
                    break
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
                    if attempt == 2:
                        raise
                    output = chat([group])[0]
            rows.append({"group_id": group["group_id"], "cases": case})
        done.update({row["group_id"]: row for row in rows})
        write_jsonl(path, done.values())
        print(f"case cards: {len(done)}/{total} groups", flush=True)
    return 0


# Stages 2 and 3: agent jobs, one folder per group with input.json, schema.json, and response.json.


def check_expansion(job):
    """One note with the requested synthetic ID and at least 80 words."""
    request, response = load_json(job / "input.json"), load_json(job / "response.json")
    notes = response.get("notes")
    if not isinstance(notes, list) or len(notes) != 1:
        return ["response must contain exactly one note"]
    expected, actual = [case["synthetic_id"] for case in request["cases"]], [note.get("synthetic_id") for note in notes]
    if actual != expected:
        return [f"synthetic IDs/order mismatch: {actual} != {expected}"]
    return [f"note too short: {note.get('synthetic_id')}" for note in notes if len(note_text(note).split()) < 80]


def check_format(job):
    """Length, block and heading structure, plain text, and no dataset name; shorter notes are kept."""
    request, response = load_json(job / "input.json"), load_json(job / "response.json")
    if response.get("synthetic_id") != request["synthetic_id"]:
        return ["synthetic_id mismatch"]
    note = note_text(response).strip()
    words, blocks, headings = len(WORD.findall(note)), len(BLOCK.split(note)), len(HEADING.findall(note))
    minimum = max(14, request["target_sections"] - 6)
    if words > request["maximum_words"]:
        return [
            f"word count {words}; require no more than {request['maximum_words']} around target {request['target_words']}"
        ]
    if abs(blocks - request["target_blocks"]) > 2:
        return [f"block count {blocks} not near target {request['target_blocks']}"]
    if headings < minimum:
        return [f"only {headings} headings; require {minimum}"]
    if MARKDOWN.search(note):
        return ["Markdown formatting remains"]
    if re.search(r"(?i)\bMIMIC(?:-III|-IV)?\b", note):
        return ["dataset name appears in note text"]
    return []


def run_jobs(jobs, prompt, check, args, attempts, followup):
    """Run the agent on every job without a valid response, retrying generation and validation failures."""

    def errors(job):
        try:
            return check(job)
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as error:  # missing or broken response
            return [str(error)]

    pending = [lambda s, job=job: agent(job, prompt, lambda _: errors(job), args, s, attempts=attempts, followup=followup)
               for job in jobs if errors(job)]
    return run(pending, args)


def expand(args):
    out = args.out.resolve()
    card_rows = sorted(read_jsonl(out / "qwen_case_cards.jsonl"), key=lambda row: row["group_id"])
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["notes"],
        "properties": {"notes": {"type": "array", "minItems": 1, "maxItems": 1, "items": TEXT_SCHEMA}},
    }
    jobs = []
    for row in card_rows:
        job = out / "gpt_jobs" / row["group_id"]
        write_json(job / "input.json", {"group_id": row["group_id"], "cases": row["cases"]})
        write_json(job / "schema.json", schema)
        jobs.append(job)
    if run_jobs(jobs, EXPAND, check_expansion, args, 2, bullet_feedback("Correct every validation error:", 80)):
        return 1
    write_jsonl(out / "gpt_expanded.jsonl", [
        {"synthetic_id": note["synthetic_id"], "group_id": job.name, "text": note["text"].strip()}
        for job in sorted(jobs) for note in load_json(job / "response.json")["notes"]])
    return 0


def reformat(args):
    out = args.out.resolve()
    drafts = {row["group_id"]: row for row in read_jsonl(out / "gpt_expanded.jsonl")}
    card_rows = {row["group_id"]: row for row in read_jsonl(out / "qwen_case_cards.jsonl")}
    jobs = []
    for group_id in sorted(drafts):
        card = card_rows[group_id]["cases"][0]
        words = int(card["target_words"])
        job = out / "terra_format_jobs" / group_id
        write_json(job / "input.json", {
            "synthetic_id": drafts[group_id]["synthetic_id"],
            "target_words": words,
            "minimum_words": round(words * 0.85),
            "maximum_words": round(words * 1.15),
            "target_blocks": min(12, max(6, round(8.3 * words / 1540.7))),
            "target_sections": min(40, max(18, round(28.5 * words / 1540.7))),
            "case_card": card,
            "synthetic_draft": drafts[group_id]["text"],
        })
        write_json(job / "schema.json", TEXT_SCHEMA)
        jobs.append(job)
    followup = lambda feedback: (
        f"\n\nThe previous attempt failed mechanical validation: {feedback}. Correct this explicitly in the new response."
    )
    if run_jobs(jobs, REFORMAT, check_format, args, 3, followup):
        return 1
    write_jsonl(out / f"{FORMATTED}.jsonl", [
        {"synthetic_id": response["synthetic_id"], "text": response["text"].strip()}
        for response in (load_json(job / "response.json") for job in sorted(jobs))])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("cards", "expand", "reformat"):
        stage = stages.add_parser(name)
        stage.add_argument("--out", type=Path, default=OUT)
        if name == "cards":
            stage.add_argument("--model", default=str(QWEN), help="Local Qwen AWQ checkpoint")
            stage.add_argument("--tensor-parallel-size", type=int, default=2)
        else:
            stage.add_argument("--model", default="gpt-5.6-terra", help="Agent model")
            stage.add_argument("--reasoning-effort", default="high")
            stage.add_argument("--max-parallel", type=int, default=3)
    args = parser.parse_args()
    return {"cards": cards, "expand": expand, "reformat": reformat}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
