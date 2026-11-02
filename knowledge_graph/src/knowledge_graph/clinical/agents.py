"""`codex exec` agent jobs with schema output, validation feedback, and bounded parallelism.

Used by the Synthetic-100 generation (synthetic) and annotation (annotation, hpo_annotation) stages.
"""

import asyncio
import json

from knowledge_graph.common import load_json


def read_response(path):
    """Return a decoded response and retry feedback for missing or malformed JSON."""
    try:
        candidate = load_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as error:
        return None, [f"invalid response JSON: {error}"]
    if not isinstance(candidate, dict):
        return None, ["response must be a JSON object"]
    return candidate, []


async def agent(job, prompt, accept, args, semaphore, *, attempts, followup):
    """Run `codex exec` in the job directory until accept(response) reports no errors; errors go into the next prompt."""
    async with semaphore:
        feedback = ""
        for attempt in range(1, attempts + 1):
            response = job / "response.json"
            response.unlink(missing_ok=True)
            command = ["codex", "exec", "-C", str(job), "--skip-git-repo-check", "-m", args.model,
                       "-c", f'model_reasoning_effort="{args.reasoning_effort}"', "-s", "read-only", "--ephemeral",
                       "--output-schema", str(job / "schema.json"), "-o", str(response),
                       prompt + (followup(feedback) if feedback else "")]
            with (job / f"attempt_{attempt}.log").open("w", encoding="utf-8") as log:
                process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT)
                code = await process.wait()
            errors = [f"agent command exited with status {code}"] if code else accept(response)
            if not errors:
                return True
            feedback = "\n".join(errors)
        return False


def bullet_feedback(header, lines):
    return lambda feedback: f"\n\n{header}\n- " + "\n- ".join(feedback.splitlines()[:lines])


def run(jobs, args):
    """Run job coroutine factories with at most args.max_parallel at a time; exit status 1 when any job fails."""

    async def gather():
        semaphore = asyncio.Semaphore(args.max_parallel)
        return await asyncio.gather(*(job(semaphore) for job in jobs))

    return int(not all(asyncio.run(gather())))
