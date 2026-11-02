"""Shared HTML cleanup and saved-page I/O for the outlet parsers."""
import copy
import json

from .paths import ROOT

STRIP_ATTRS = {"class", "id", "style", "onclick", "onmouseover", "onerror", "onload"}


def clean_inner_html(element, origin):
    """Remove presentation attributes and resolve relative links without mutating the DOM."""
    element = copy.deepcopy(element)
    for tag in element.find_all(True):
        if tag.name == "script":
            tag.decompose()
            continue
        for attr in STRIP_ATTRS:
            tag.attrs.pop(attr, None)
        if tag.name == "a" and (tag.get("href") or "").startswith("/"):
            tag["href"] = origin + tag["href"]
            tag["target"] = "_blank"
            tag["rel"] = "noopener"
    return element.decode_contents().strip()


def is_link_only_list(element):
    """Identify lists whose items each have at least 60% linked text."""
    if element.name not in ("ul", "ol"):
        return False
    items = element.find_all("li", recursive=False)
    if not items:
        return False
    for item in items:
        text = item.get_text(" ", strip=True)
        links = "".join(link.get_text(" ", strip=True) for link in item.find_all("a"))
        if not text or not links or len(links) / len(text) < 0.6:
            return False
    return True


def read_articles(directory, parse):
    """Parse saved pages in filename order, with their editorial metadata."""
    with (ROOT / "data/articles.json").open(encoding="latin-1") as stream:
        metadata = {article["articleId"]: article for article in json.load(stream)}
    for page in sorted((ROOT / "data/pages" / directory).glob("*.html")):
        record = parse(page.stem, page.read_text(encoding="utf-8"), metadata.get(page.stem, {}))
        if record:
            yield record


def unique_articles(records):
    """Keep the first occurrence of each title and body, before section filtering."""
    seen = set()
    for record in records:
        content = record.get("title", "") + "".join(record["sentences"])
        if content not in seen:
            seen.add(content)
            yield record


def write_articles(filename, records):
    """Write articles with at least two sections."""
    destination = ROOT / "data/outputs" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump([record for record in records if record["num_sections"] >= 2],
                  stream, ensure_ascii=False, indent=2)
