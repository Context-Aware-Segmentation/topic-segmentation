"""Extract editorial sections from saved Independent pages."""


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
    "bookmark popover",
    "join our commenting forum",
    "bulletin",
    "most popular",
    "popular videos",
    "more about",
]

TRAILING_HEADINGS = {
    "read next",
    "most read",
}

TRAILING_TEXT_PREFIXES = (
    "explore more on the topics",
    "this article was changed",
    "this article was first published",
    "this piece was first published",
    "read more of our",
)

SKIP_PARA_PREFIXES = (
    "sign up",
    "i would like to be emailed",
    "read more on ",
    "read more:",
)

# Classes identifying excluded divs
SKIP_DIV_CLASSES = {
    "hydrate-root",
    "teads",
    "taboola",
    "related",
    "sc-toncsa-0",
}


def is_noise_heading(text: str) -> bool:
    t = text.lower().strip()
    return len(t) < 3 or any(pat in t for pat in NOISE_PATTERNS)


def is_skip_container(el) -> bool:
    if el.name in ("aside", "nav", "form"):
        return True
    cls = set(el.get("class") or [])
    if cls & SKIP_DIV_CLASSES:
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
    """Collect content elements from the article body."""
    elements = []
    for el in container.children:
        if not hasattr(el, "name") or el.name is None:
            continue
        if is_skip_container(el):
            continue

        cls = set(el.get("class") or [])

        # Section subheading div or content list wrapped in sc-kk992l-0
        if el.name == "div" and "sc-kk992l-0" in cls:
            inner_list = el.find(["ul", "ol"])
            inner_head = el.find(["h2", "h3", "h4"])
            if inner_list and not inner_head:
                # Include content lists and skip link-only "Read more" lists.
                if not is_link_only_list(inner_list):
                    elements.append(inner_list)
            else:
                elements.append(el)
            continue

        # Image div (contains a figure with an actual image)
        if el.name == "div" and "image" in cls:
            fig = el.find("figure")
            if fig:
                img = fig.find("img")
                if img and (img.get("src") or "").startswith("http"):
                    elements.append(fig)
            continue

        if el.name in ("p", "ul", "ol", "blockquote"):
            if not is_link_only_list(el):
                elements.append(el)
            continue

        # Skip unrecognized divs, including ads and empty wrappers.

    return elements


def process_article(aid: str, raw_html: str, meta: dict):
    soup = BeautifulSoup(ftfy.fix_text(raw_html), "html.parser")

    article = soup.find("article")
    if not article:
        return None

    # Article header (inside article, contains title/standfirst/hero)
    art_header = article.find("header")

    h1 = art_header.find("h1") if art_header else article.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else meta.get("title", "")

    # Standfirst: h2.sc-1cmj4ml-0 inside article header
    standfirst_text = None
    if art_header:
        sf = art_header.find("h2", class_=lambda c: c and "sc-1cmj4ml-0" in c)
        if sf:
            text = sf.get_text(" ", strip=True)
            if text and not text.lower().startswith("sign up"):
                standfirst_text = text

    # Hero image from article header figure
    hero_img_src = None
    hero_img_cap = None
    if art_header:
        fig = art_header.find("figure")
        if fig:
            img = fig.find("img")
            if img and (img.get("src") or "").startswith("http"):
                hero_img_src = img["src"]
                figcap = fig.find("figcaption")
                if figcap:
                    hero_img_cap = figcap.get_text(" ", strip=True)

    main_wrapper = article.find("div", class_=lambda c: c and "main-wrapper" in c)
    if not main_wrapper:
        return None
    content = main_wrapper.find("div", class_=lambda c: c and "sc-jbiisr-0" in c)
    if not content:
        return None

    sentences, html_parts, labels, title_labels = [], [], [], []

    if standfirst_text:
        sentences.append(standfirst_text)
        html_parts.append({"tag": "p", "html": standfirst_text})
        labels.append(0)
        title_labels.append(1)

    if hero_img_src:
        cap_html = f"<figcaption>{hero_img_cap}</figcaption>" if hero_img_cap else ""
        sentences.append(hero_img_cap or "")
        html_parts.append({"tag": "figure", "html": f'<img src="{hero_img_src}" style="max-width:100%">{cap_html}'})
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

        # Section subheading div (sc-kk992l-0)
        if el.name == "div" and "sc-kk992l-0" in set(el.get("class") or []):
            text = el.get_text(" ", strip=True)
            if not text or is_noise_heading(text):
                continue
            if len(text) > 200:
                continue
            inner_head = el.find(["h2", "h3", "h4"])
            tag = inner_head.name if inner_head else "h3"
            sentences.append(text)
            html_parts.append({"tag": tag, "html": text})
            labels.append(0 if first_elem else 1)
            title_labels.append(1)
            first_elem = False
            continue

        text = el.get_text(" ", strip=True)
        if not text:
            continue

        if any(text.lower().startswith(p) for p in SKIP_PARA_PREFIXES):
            continue

        sentences.append(text)
        html_parts.append({"tag": el.name, "html": clean_inner_html(el, "https://www.independent.co.uk")})
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
        "outlet": "Independent",
        "title": title,
        "url": meta.get("url", ""),
        "sentences": sentences,
        "html_parts": html_parts,
        "labels": labels,
        "title_labels": title_labels,
        "num_sections": labels.count(1) + 1,
    }


def main():
    articles = read_articles("independent_subheadings", process_article)
    write_articles("ind_segments.json", unique_articles(articles))


if __name__ == "__main__":
    main()
