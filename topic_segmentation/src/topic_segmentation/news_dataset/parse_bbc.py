"""Extract editorial sections from saved BBC pages."""

import ftfy
from bs4 import BeautifulSoup

from .parser_common import (
    read_articles, write_articles, is_link_only_list, clean_inner_html,
)


NOISE_PATTERNS = [
    "bookmark popover",
    "join our commenting forum",
    "most read",
    "related topics",
    "more on this story",
    "read next",
    "read more",
    "sign up",
    "newsletter",
    "get the latest",
    "stay ahead",
    "to play this video",
    "trending",
    "comments",
    "share this",
    "you may also like",
    "recommended",
    "popular",
    "unpack football",
    "acfc newsletter",
    "more about",
    "bulletin",
    "related topics:",
    "click to subscribe",
    "follow here",
    "apple podcasts",
    "spotify",
    "acast",
    "wherever you listen",
    "more on ",
    "read more from",
    "get the free",
    "morning headlines",
    "email for news",
    "related internet links",
    "removed from bookmarks",
    "i would like to be emailed",
]

# Block tags collected as article elements
BLOCK_TAGS = {
    "h2", "h3", "h4", "p", "ul", "ol", "blockquote", "figure", "table"
}

# Attributes identifying excluded subtrees
SKIP_ID_SUBSTRINGS = ("why-not-follow", )
SKIP_TEST_IDS = {"links-grid", "metadata"}
SKIP_COMPONENTS = {"byline-block"}
SKIP_BLOCKS = {"legacyHeader", "links", "promoList", "social"}

# Heading text that signals the end of article content
TRAILING_HEADINGS = {
    "related topics",
    "more on this story",
    "you might also be interested in",
    "you may also be interested in",
}

# Alt-text / plain-text substrings that signal trailing boilerplate
TRAILING_TEXT_PATTERNS = (
    "get in touch",
    "news daily banner",
)

# Promo text: exact matches after lowercasing and removing the trailing period.
SKIP_PARA_TEXT = {
    "read more here",
    "read more",
}

# Lowercase prefixes identifying promotional paragraphs.
SKIP_PARA_PREFIXES = (
    "the documentary is available",
    "read more tips",
)

# Prefixes identifying BBC series and brand promos in any element.
SKIP_ANY_PREFIXES = (
    "bbc 100 women",
    "follow bbc 100 women",
    "watch now on bbc",   # mid-article iPlayer promo CTA
    "watch on youtube",
)

# Text prefixes that signal trailing boilerplate
TRAILING_TEXT_PREFIXES = (
    "why not follow bbc",
    "additional reporting by",
    "images by ",
    "image credits",
    "video produced",
    "video production by",
    "filmed by ",
    "all photos copyright",
    "listen to the best of bbc",
    "send your story ideas",
    "correction ",  # "Correction 20 May 2025: ..."
)

# Heading text prefixes (lowercased) that signal trailing boilerplate sections
TRAILING_HEADING_PREFIXES = (
    "more stories from",
    "follow live",
)


def is_noise_heading(text: str) -> bool:
    t = text.lower().strip()
    return len(t) < 3 or any(pat in t for pat in NOISE_PATTERNS)


def is_skip_container(el) -> bool:
    """Identify subtrees to exclude from article content."""
    el_id = (el.get("id") or "").lower()
    if any(s in el_id for s in SKIP_ID_SUBSTRINGS):
        return True
    if el.get("data-testid") in SKIP_TEST_IDS:
        return True
    if el.get("data-component") in SKIP_COMPONENTS:
        return True
    if el.get("data-block") in SKIP_BLOCKS:
        return True
    # Skip video player figures (contain "To play this video..." boilerplate)
    if el.name == "figure" and el.find(
            attrs={"data-testid": "media-player-container-landscape"}):
        return True
    return False


def is_trailing_element(el) -> bool:
    """Identify the start of trailing boilerplate."""
    # Match trailing headings by text or prefix.
    if el.name in ("h2", "h3", "h4"):
        text_h = el.get_text(" ", strip=True).lower().rstrip(":")
        if text_h in TRAILING_HEADINGS:
            return True
        if any(text_h.startswith(p) for p in TRAILING_HEADING_PREFIXES):
            return True

    # Check the element and its descendants for boilerplate markers.
    for tag in [el, *el.find_all(True)]:
        if any(s in (tag.get("id") or "").lower() for s in SKIP_ID_SUBSTRINGS):
            return True
        if tag.get("data-testid") in SKIP_TEST_IDS:
            return True
        if any(pat in (tag.get("alt") or "").lower()
               for pat in TRAILING_TEXT_PATTERNS):
            return True

    # Treat italicized paragraphs as follow prompts only when they contain links.
    if el.name == "p":
        children = [c for c in el.children if hasattr(c, "name") and c.name]
        has_link = any(c.name == "a" for c in children)
        if has_link and children[0].name in ("em", "i") and all(
                c.name in ("em", "i", "a") for c in children):
            return True

    text_lower = el.get_text(" ", strip=True).lower()
    if any(pat in text_lower for pat in TRAILING_TEXT_PATTERNS):
        return True
    if any(text_lower.startswith(pat) for pat in TRAILING_TEXT_PREFIXES):
        return True

    return False


def collect_elements(container) -> list:
    """Collect content in document order, skipping excluded subtrees and stopping at trailing boilerplate."""
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

    article = soup.select_one("article")
    if not article:
        return None
    first_div = article.find("div", recursive=False)
    container = first_div if first_div else article

    h1 = soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else meta.get("title", "")

    sentences, html_parts, labels, title_labels = [], [], [], []
    first_elem = True

    for el in collect_elements(container):
        if is_trailing_element(el):
            break
        # For figures: strip "Image source,X" attribution, keep only the caption
        if el.name == "figure":
            figcap = el.find("figcaption")
            if not figcap:
                continue
            # Remove attribution spans (visually-hidden "Image source,..." nodes)
            import copy as _copy
            cap = _copy.deepcopy(figcap)
            for span in cap.find_all(True):
                if "image source" in span.get_text(" ", strip=True).lower()[:20]:
                    span.decompose()
            text = cap.get_text(" ", strip=True)
        else:
            text = el.get_text(" ", strip=True)
        if not text:
            continue

        # Skip promotional headings and continue collecting content.
        if el.name in ("h2", "h3", "h4") and is_noise_heading(text):
            continue
        # Link-only boilerplate paragraphs (e.g. "Read more here")
        if el.name == "p" and text.lower().rstrip(".") in SKIP_PARA_TEXT:
            continue
        # Mid-article promo paragraphs (e.g. "The documentary is available…")
        if el.name == "p" and any(text.lower().startswith(p)
                                  for p in SKIP_PARA_PREFIXES):
            continue
        # BBC series and brand promos
        if any(text.lower().startswith(p) for p in SKIP_ANY_PREFIXES):
            continue

        is_heading = (el.name in ("h2", "h3", "h4") and len(text) <= 200)
        sentences.append(text)
        html_parts.append({"tag": el.name, "html": clean_inner_html(el, "https://www.bbc.co.uk")})
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
        "articleId": aid,
        "outlet": "BBC",
        "title": title,
        "url": meta.get("url", ""),
        "sentences": sentences,
        "html_parts": html_parts,
        "labels": labels,
        "title_labels": title_labels,
        "num_sections": labels.count(1) + 1,
    }


def main():
    articles = read_articles("bbc_subheadings", process_article)
    write_articles("bbc_segments.json", articles)


if __name__ == "__main__":
    main()
