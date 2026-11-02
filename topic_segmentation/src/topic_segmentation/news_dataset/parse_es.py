"""Extract editorial sections from saved Evening Standard pages."""

import ftfy
from bs4 import BeautifulSoup

from .parser_common import (
    read_articles, write_articles, unique_articles, clean_inner_html,
)


NOISE_PATTERNS = [
    "read more", "most read", "sign up", "newsletter", "trending",
    "share this", "you may also like", "recommended", "popular",
    "follow .+ on", "get in touch", "i would like to be emailed",
    "read next", "click to subscribe", "comments",
]

BLOCK_TAGS = {"h2", "h3", "h4", "p", "ul", "ol", "blockquote", "figure", "table", "picture"}

# Excluded containers
SKIP_IDS        = {"polarArticleWrapper"}
SKIP_CLASSES    = {"readMoreVariants", "teads", "mobile-only"}
SKIP_DATA_ATTRS = {"data-mpu", "data-read-more-type"}   # ad units, read-more blocks
SKIP_COMPONENTS = {"shopping_best_buys"}

# Heading texts that signal end of article
TRAILING_HEADINGS = {
    "most read",
    "read more",
}

TRAILING_TEXT_PREFIXES = (
    "follow us on",
    "get the latest",
    "by clicking register",
)

# Promotional paragraphs to remove throughout the article.
SKIP_PARA_PREFIXES = (
    "sign up",
    "i would like to be emailed",
    "create a free account",
)


def is_noise_heading(text: str) -> bool:
    import re
    t = text.lower().strip()
    if len(t) < 3:
        return True
    noise = [
        r"^read more$", r"^most read$", r"^comments?$", r"^share$",
        r"sign up", r"newsletter", r"follow .+ on",
        r"why you can trust es", r"why es best",
    ]
    return bool(re.search("|".join(noise), t, re.IGNORECASE))


def is_skip_container(el) -> bool:
    """Identify subtrees to exclude from article content."""
    el_id = el.get("id") or ""
    if el_id in SKIP_IDS:
        return True
    el_classes = set(el.get("class") or [])
    if el_classes & SKIP_CLASSES:
        return True
    if any(el.get(a) for a in SKIP_DATA_ATTRS):
        return True
    if el.get("data-component") in SKIP_COMPONENTS:
        return True
    if el.name in ("aside", "form"):
        return True
    # Related-article widgets headed "Read More".
    if el.name == "div":
        first_h = el.find(["h2", "h3", "h4"])
        if first_h and first_h.get_text(" ", strip=True).lower() == "read more":
            return True
    return False


def is_link_only_list(el) -> bool:
    if el.name not in ("ul", "ol"):
        return False
    items = el.find_all("li", recursive=False)
    if not items:
        return False
    for li in items:
        li_text = li.get_text(" ", strip=True)
        a_text  = "".join(a.get_text(" ", strip=True) for a in li.find_all("a"))
        if not li_text or li_text != a_text:
            return False
    return True


def is_trailing_element(el) -> bool:
    if el.name in ("h2", "h3", "h4"):
        t = el.get_text(" ", strip=True).lower().rstrip(":")
        if t in TRAILING_HEADINGS:
            return True
    text_lower = el.get_text(" ", strip=True).lower()
    if any(text_lower.startswith(p) for p in TRAILING_TEXT_PREFIXES):
        return True
    return False


def collect_elements(container) -> list:
    seen_ids: set = set()
    elements: list = []

    def walk(node):
        for el in node.children:
            if not hasattr(el, "name") or el.name is None:
                continue
            if is_skip_container(el):
                continue
            eid = id(el)
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            if el.name in BLOCK_TAGS:
                if not is_link_only_list(el):
                    elements.append(el)
            else:
                walk(el)

    walk(container)
    return elements


def process_article(aid: str, raw_html: str, meta: dict):
    soup = BeautifulSoup(ftfy.fix_text(raw_html), "html.parser")

    main = soup.find(id="main")
    if not main:
        return None

    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else meta.get("title", "")

    # Standfirst: sibling div immediately after h1 inside its container
    standfirst_text = None
    if h1:
        for sib in h1.next_siblings:
            if hasattr(sib, "name") and sib.name == "div":
                t = sib.get_text(" ", strip=True)
                if t:
                    standfirst_text = t
                break

    # Hero image: picture in siblings before #main.
    hero_img_src = None
    if main.parent:
        for sib in main.parent.children:
            if sib is main:
                break
            if sib.name is not None:
                pic = sib.find("picture")
                if pic:
                    img_tag = pic.find("img")
                    src = img_tag.get("src", "") if img_tag else ""
                    if src and "google_preferred" not in src:
                        hero_img_src = src
                        break

    # Shopping articles continue in sibling divs after #main.
    # Include those containers, excluding sections headed "Read More".
    containers = [main]
    if main.parent:
        passed_main = False
        for sib in main.parent.children:
            if not hasattr(sib, "name") or sib.name != "div":
                if sib is main:
                    passed_main = True
                continue
            if sib is main:
                passed_main = True
                continue
            if not passed_main:
                continue
            sib_id = sib.get("id") or ""
            if sib_id in ("article-bottom",):
                continue
            first_h = sib.find(["h2", "h3", "h4"])
            if first_h and first_h.get_text(" ", strip=True).lower().startswith("read more"):
                continue
            if len(sib.get_text(" ", strip=True)) > 500:
                containers.append(sib)

    sentences, html_parts, labels, title_labels = [], [], [], []
    if standfirst_text:
        sentences.append(standfirst_text)
        html_parts.append({"tag": "p", "html": standfirst_text})
        labels.append(0)
        title_labels.append(1)
    if hero_img_src:
        sentences.append("")
        html_parts.append({"tag": "figure", "html": f'<img src="{hero_img_src}" style="max-width:100%">'})
        labels.append(0)
        title_labels.append(0)
    first_elem = True

    for container in containers:
        for el in collect_elements(container):
            if is_trailing_element(el):
                break
            # Handle <picture> elements as inline images
            if el.name == "picture":
                img = el.find("img")
                if not img:
                    continue
                src = img.get("src", "") or img.get("data-src", "")
                if not src:
                    continue
                sentences.append("")
                html_parts.append({"tag": "figure", "html": f'<img src="{src}" style="max-width:100%">'})
                labels.append(0)
                title_labels.append(0)
                first_elem = False
                continue

            text = el.get_text(" ", strip=True)
            if not text:
                continue

            # Skip promotional headings and continue collecting content.
            if el.name in ("h2", "h3", "h4") and is_noise_heading(text):
                continue
            # Skip newsletter and subscription prompts.
            if el.name == "p" and any(text.lower().startswith(p) for p in SKIP_PARA_PREFIXES):
                continue

            is_heading = el.name in ("h2", "h3", "h4") and len(text) <= 200

            sentences.append(text)
            html_parts.append({"tag": el.name, "html": clean_inner_html(el, "https://www.standard.co.uk")})
            if is_heading:
                labels.append(0 if first_elem else 1)
                title_labels.append(1)
            else:
                labels.append(0)
                title_labels.append(0)
            first_elem = False

    # Remove trailing headings without following content.
    while sentences and title_labels[-1] == 1:
        sentences.pop()
        html_parts.pop()
        labels.pop()
        title_labels.pop()

    if not sentences:
        return None

    return {
        "articleId":   aid,
        "outlet":      "EveningStandard",
        "title":       title,
        "url":         meta.get("url", ""),
        "sentences":   sentences,
        "html_parts":  html_parts,
        "labels":      labels,
        "title_labels": title_labels,
        "num_sections": labels.count(1) + 1,
    }


def main():
    articles = read_articles("eveningstandard_subheadings", process_article)
    write_articles("es_segments.json", unique_articles(articles))


if __name__ == "__main__":
    main()
