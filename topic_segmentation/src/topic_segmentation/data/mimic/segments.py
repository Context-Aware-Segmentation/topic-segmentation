"""Blocks and heading sections of MIMIC discharge notes, split into sentences.

    python -m topic_segmentation.data.mimic.segments --cache NOTES_by_id.json [--cache MORE_by_id.json] --out SEGMENTS.json

Blocks are separated by blank lines; within a block, hash, colon, underlined, and standalone headings open sections
(data.mimic.rules). Lines before the first heading form the block prelude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from topic_segmentation.data.mimic.rules import (
    UNDERLINE_RE,
    colon_header_parts,
    hash_header_parts,
    label_key,
    normalize_front_matter_doc,
    standalone_header_parts,
    split_blocks,
    split_sentences,
    underline_heading_label,
)


def start_section(
    sections: list[dict[str, Any]],
    current: dict[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    if current is not None:
        sections.append(current)
    return {
        "label": label,
        "_lines": [],
    }


def segment_block(block_lines: list[tuple[int, str]]) -> dict[str, Any]:
    prelude_lines: list[str] = []
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    i = 0
    while i < len(block_lines):
        _, line = block_lines[i]
        if i + 1 < len(block_lines):
            underline_label = underline_heading_label(line, block_lines[i + 1][1])
            if underline_label is not None:
                current = start_section(sections, current, underline_label)
                i += 2
                continue

        header, header_type = hash_header_parts(line), "hash_heading"
        if header is None:
            header = colon_header_parts(line)
            header_type = "major_inline_heading" if header is not None and header[1] else "colon_heading"
        if header is not None:
            label, value = header
            # A repeated heading right after the same heading continues that section.
            repeated = current is not None and label_key(label) == label_key(current["label"]) and not current["_lines"]
            if not repeated:
                current = start_section(sections, current, label)
            if value:
                current["_lines"].append(value)
            underlined = (not repeated and header_type != "hash_heading" and i + 1 < len(block_lines)
                          and UNDERLINE_RE.match(block_lines[i + 1][1]))
            i += 2 if underlined else 1
            continue

        standalone_header = standalone_header_parts(line)
        if standalone_header is not None:
            label, _ = standalone_header
            current = start_section(sections, current, label)
            i += 1
            continue

        if current is None:
            prelude_lines.append(line)
        else:
            current["_lines"].append(line)
        i += 1

    if current is not None:
        sections.append(current)

    sections = [{"sentences": split_sentences(section["_lines"])} for section in sections]
    prelude = split_sentences(prelude_lines)
    return {"prelude": {"sentences": prelude} if prelude else None, "sections": sections}


def build_note_segments(note):
    """Normalize front matter and split a note into blocks, preludes, and heading sections."""
    note = normalize_front_matter_doc(note)
    blocks = [segment_block(lines) for _, _, lines in split_blocks(note["text"])]
    return {"note_id": note["note_id"], "table": note["table"], "blocks": blocks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, action="append", required=True,
                        help="Note cache (*_by_id.json) of data.mimic.sample; repeat for several")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    order, notes = {}, {}
    for path in args.cache:
        cache = json.loads(path.read_text(encoding="utf-8"))
        for note_id in cache["order"]:
            if note_id in notes:
                raise ValueError(f"note {note_id} is in two caches")
            notes[note_id] = build_note_segments(cache["notes"][note_id])
            order.setdefault(notes[note_id]["table"], []).append(note_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"order": order, "notes": notes}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
