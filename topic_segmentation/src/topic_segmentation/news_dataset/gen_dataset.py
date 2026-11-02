#!/usr/bin/env python3
"""Merge parsed articles from data/outputs/*_segments.json into data/dataset.jsonl.

Add block_types (standfirst, h2, h3, image, element) and image subtypes (hero, body).
Map h4 and other heading tags to h3. Store the headline separately in title.
Normalize list formatting and remove duplicate content across outlets.

    python -m topic_segmentation.news_dataset.gen_dataset
"""

import json
import os
import re

from bs4 import BeautifulSoup

from .paths import ROOT
OUTPUTS_DIR = ROOT / "data" / "outputs"
DATASET_PATH = ROOT / "data" / "dataset.jsonl"

SEGMENTS_FILES = [
    "bbc_segments.json",
    "es_segments.json",
    "guardian_segments.json",
    "inews_segments.json",
    "ind_segments.json",
    "skynews_segments.json",
]


WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r" +([.!?,;:])")


def normalise(text: str) -> str:
    """Normalize whitespace and punctuation spacing for content comparison."""
    return SPACE_BEFORE_PUNCT_RE.sub(r"\1", WHITESPACE_RE.sub(" ", text)).strip()


def reformat_list_body(html: str, ordered: bool) -> str:
    """Convert a `<ul>`/`<ol>` HTML fragment into bullet- or number-prefixed lines."""
    soup = BeautifulSoup(html, "html.parser")
    items = [li.get_text(" ", strip=True) for li in soup.find_all("li")]
    items = [it for it in items if it]
    if not items:
        return ""
    if ordered:
        return "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items))
    return "\n".join(f"- {it}" for it in items)


def classify_record(record: dict) -> tuple[list, list]:
    block_types = []
    block_subtypes = []
    seen_figure = False

    parts = record.get("html_parts", [])
    title_labels = record.get("title_labels", [])

    for i, part in enumerate(parts):
        tag = part.get("tag", "")
        html = part.get("html", "")
        is_title = bool(title_labels[i]) if i < len(title_labels) else False
        # BBC's initial bold paragraph is a standfirst with title_labels=0.
        is_bbc_bold_lead = (
            i == 0 and tag == "p" and html.lstrip().startswith("<b>")
        )
        sub = None

        if tag == "figure":
            t = "image"
            sub = "hero" if not seen_figure else "body"
            seen_figure = True
        elif (is_title and tag == "p") or is_bbc_bold_lead:
            t = "standfirst"
        elif is_title and tag == "h2":
            t = "h2"
        elif is_title and tag in ("h3", "h4"):
            t = "h3"
        elif is_title:
            t = "h3"  # Other heading tags map to h3.
        else:
            t = "element"

        block_types.append(t)
        block_subtypes.append(sub)

    return block_types, block_subtypes


def main():
    counts: dict = {}
    total = 0
    extra_dupes = 0
    seen_content: set = set()
    with DATASET_PATH.open("w", encoding="utf-8") as out:
        for fname in SEGMENTS_FILES:
            p = OUTPUTS_DIR / fname
            if not p.exists():
                print(f"  [skip] {fname} not found")
                continue
            records = json.loads(p.read_text())
            kept = 0
            for r in records:
                r["block_types"], r["block_subtypes"] = classify_record(r)
                # Use body for the article text.
                if "sentences" in r:
                    r["body"] = r.pop("sentences")
                # Reformat list bodies as "- item" / "1. item" lines.
                for i, part in enumerate(r["html_parts"]):
                    if part.get("tag") in ("ul", "ol"):
                        r["body"][i] = reformat_list_body(
                            part.get("html", ""), ordered=part["tag"] == "ol"
                        )
                # Deduplicate across outlets after normalizing whitespace and punctuation spacing.
                content = normalise(r.get("title", "") + "".join(r["body"]))
                if content in seen_content:
                    extra_dupes += 1
                    continue
                seen_content.add(content)

                for t in r["block_types"]:
                    counts[t] = counts.get(t, 0) + 1
                for s in r["block_subtypes"]:
                    if s:
                        counts[f"image:{s}"] = counts.get(f"image:{s}", 0) + 1
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
                kept += 1
            print(f"  {fname}: {len(records)} → {kept} kept")
    if extra_dupes:
        print(f"  ({extra_dupes} duplicate(s) caught by normalised content comparison)")

    size_mb = os.path.getsize(DATASET_PATH) / 1024 / 1024
    print()
    print(f"Saved {DATASET_PATH} ({size_mb:.1f} MB) — {total} articles total")
    print()
    print("Block-type totals:")
    for k in ["standfirst", "h2", "h3", "image", "image:hero", "image:body", "element"]:
        if k in counts:
            print(f"  {k}: {counts[k]:,}")


if __name__ == "__main__":
    main()
