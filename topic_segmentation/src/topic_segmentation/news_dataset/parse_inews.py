"""Extract editorial sections from saved iNews pages."""


import ftfy
from bs4 import BeautifulSoup

from .parser_common import (
    read_articles, write_articles, unique_articles, is_link_only_list, clean_inner_html,
)


NOISE_PATTERNS = [
    "most read",
    "read more",
    "read next",
    "sign up",
    "newsletter",
    "trending",
    "share this",
    "you may also like",
    "recommended",
    "popular",
    "comments",
    "click to subscribe",
    "explore more",
    "get in touch",
    "follow ",
    "more on ",
    "apple podcasts",
    "spotify",
    "acast",
    "wherever you listen",
    "most read by subscribers",
]

BLOCK_TAGS = {"h2", "h3", "h4", "p", "ul", "ol", "blockquote", "figure", "table"}

# Classes identifying excluded subtrees
SKIP_CLASSES = {
    "inews__advert",
    "inews__shortcode-relatedarticleinline",
    "inews__shortcode-relatedarticleinline__content",
    "inews__post-share",
    "inews__post-topics",
}

TRAILING_HEADINGS = {
    "read next",
    "most read",
    "explore more on the topics in this story",
}

TRAILING_TEXT_PREFIXES = (
    "explore more on the topics",
    "follow here.",
)

SKIP_PARA_PREFIXES = (
    "sign up",
    "i would like to be emailed",
)


def is_noise_heading(text: str) -> bool:
    t = text.lower().strip()
    return len(t) < 3 or any(pat in t for pat in NOISE_PATTERNS)


def is_skip_container(el) -> bool:
    el_classes = set(el.get("class") or [])
    if el_classes & SKIP_CLASSES:
        return True
    if el.name in ("aside", "form", "nav"):
        return True
    return False


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
                # Recurse into blockquote and figure wrappers containing headings.
                if el.name in ("blockquote", "figure") and el.find(["h2", "h3", "h4"]):
                    walk(el)
                elif not is_link_only_list(el):
                    elements.append(el)
            else:
                walk(el)

    walk(container)
    return elements


def process_article(aid: str, raw_html: str, meta: dict):
    soup = BeautifulSoup(ftfy.fix_text(raw_html), "html.parser")

    article = soup.find("article")
    if not article:
        return None

    content = article.find("div", class_="article-content")
    if not content:
        return None

    h1 = article.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else meta.get("title", "")

    # Standfirst from header excerpt
    standfirst_text = None
    header = article.find("header")
    if header:
        excerpt = header.find(class_="inews__post-excerpt")
        if excerpt:
            standfirst_text = excerpt.get_text(" ", strip=True)

    # Hero image from header figure
    hero_img_src = None
    if header:
        fig = header.find("figure", class_="inews__post-header__media")
        if fig:
            img = fig.find("img")
            if img:
                hero_img_src = img.get("src", "")

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

    first_elem = not (standfirst_text or hero_img_src)

    for el in collect_elements(content):
        if is_trailing_element(el):
            break

        if el.name == "figure":
            img = el.find("img")
            if not img:
                continue
            src = img.get("src", "")
            if not src:
                continue
            figcap = el.find("figcaption")
            cap_text = figcap.get_text(" ", strip=True) if figcap else ""
            cap_html = f"<figcaption>{cap_text}</figcaption>" if cap_text else ""
            sentences.append(cap_text)
            html_parts.append({"tag": "figure", "html": f'<img src="{src}" style="max-width:100%">{cap_html}'})
            labels.append(0)
            title_labels.append(0)
            first_elem = False
            continue

        text = el.get_text(" ", strip=True)
        if not text:
            continue

        if el.name in ("h2", "h3", "h4") and is_noise_heading(text):
            continue
        if el.name == "p" and any(text.lower().startswith(p) for p in SKIP_PARA_PREFIXES):
            continue

        is_heading = el.name in ("h2", "h3", "h4") and len(text) <= 200

        sentences.append(text)
        html_parts.append({"tag": el.name, "html": clean_inner_html(el, "https://inews.co.uk")})
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
        "outlet": "INews",
        "title": title,
        "url": meta.get("url", ""),
        "sentences": sentences,
        "html_parts": html_parts,
        "labels": labels,
        "title_labels": title_labels,
        "num_sections": labels.count(1) + 1,
    }


def main():
    articles = read_articles("inews_subheadings", process_article)
    write_articles("inews_segments.json", unique_articles(articles))


if __name__ == "__main__":
    main()
