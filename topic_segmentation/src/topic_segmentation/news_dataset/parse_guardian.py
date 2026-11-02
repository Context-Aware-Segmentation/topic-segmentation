"""Extract editorial sections from saved Guardian pages."""
import copy


import ftfy
from bs4 import BeautifulSoup

from .parser_common import (
    read_articles, write_articles, unique_articles, is_link_only_list, STRIP_ATTRS,
)


NOISE_PATTERNS = [
    "most viewed",
    "more on this story",
    "comments",
    "explore more on these topics",
    "sign up",
    "newsletter",
    "get in touch",
    "read more",
    "read next",
    "trending",
    "share this",
    "you may also like",
    "recommended",
    "popular",
    "click to subscribe",
    "apple podcasts",
    "spotify",
    "wherever you listen",
    "removed from bookmarks",
    "i would like to be emailed",
    "follow ",
    "more on ",
]

BLOCK_TAGS = {"h2", "h3", "h4", "p", "ul", "ol", "blockquote", "figure", "table"}

# Heading text that signals end of article content
TRAILING_HEADINGS = {
    "most viewed",
    "more on this story",
    "explore more on these topics",
    "comments",
    "topics",
}

TRAILING_HEADING_PREFIXES = (
    "comments (",
    "comments(",
)

TRAILING_TEXT_PREFIXES = (
    "explore more on these topics",
)

SKIP_PARA_PREFIXES = (
    "sign up to",
    "i would like to be emailed",
    "after newsletter promotion",
    "this is the archive of the observer",
    "this is an extract from our",
    "send any letters to",
)


def is_noise_heading(text: str) -> bool:
    t = text.lower().strip()
    return len(t) < 3 or any(pat in t for pat in NOISE_PATTERNS)


def is_skip_container(el) -> bool:
    """Identify subtrees to exclude from article content."""
    # Rich-link figures (related article widgets)
    if el.name == "figure" and el.get("data-spacefinder-role") == "richLink":
        return True
    if el.name == "aside" and el.get("data-gu-name") != "pullquote":
        return True
    if el.name in ("form", "nav"):
        return True
    return False


def is_trailing_element(el) -> bool:
    if el.name in ("h2", "h3", "h4"):
        t = el.get_text(" ", strip=True).lower().rstrip(":")
        if t in TRAILING_HEADINGS:
            return True
        if any(t.startswith(p) for p in TRAILING_HEADING_PREFIXES):
            return True
    text_lower = el.get_text(" ", strip=True).lower()
    if any(text_lower.startswith(p) for p in TRAILING_TEXT_PREFIXES):
        return True
    return False


def clean_inner_html(el) -> str:
    el_copy = copy.deepcopy(el)
    # Replace decorative SVG quote marks with a real quote character
    for svg in el_copy.find_all("svg"):
        svg.replace_with("\u201c")
    for tag in el_copy.find_all(True):
        if tag.name == "script":
            tag.decompose()
            continue
        # Preserve bullet markers
        if tag.name == "span" and tag.get("data-dcr-style") == "bullet":
            tag.attrs = {"data-dcr-style": "bullet"}
            continue
        for attr in STRIP_ATTRS:
            tag.attrs.pop(attr, None)
        if tag.name == "a" and (tag.get("href") or "").startswith("/"):
            tag["href"] = "https://www.theguardian.com" + tag["href"]
            tag["target"] = "_blank"
            tag["rel"] = "noopener"
    return el_copy.decode_contents().strip()


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
                # Recurse into figure wrappers containing headings.
                if el.name == "figure" and el.find(["h2", "h3", "h4"]):
                    walk(el)
                elif not is_link_only_list(el):
                    elements.append(el)
            else:
                walk(el)

    walk(container)
    return elements


def process_article(aid: str, raw_html: str, meta: dict):
    soup = BeautifulSoup(ftfy.fix_text(raw_html), "html.parser")

    body = soup.find(attrs={"data-gu-name": "body"})
    if not body:
        return None

    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else meta.get("title", "")

    # Standfirst — skip if it's a pure link (newsletter CTA) or boilerplate
    standfirst_text = None
    sf_div = soup.find(attrs={"data-gu-name": "standfirst"})
    if sf_div:
        # Try <p> first, fall back to the whole standfirst div text
        p = sf_div.find("p")
        sf_el = p if p else sf_div
        text = sf_el.get_text(" ", strip=True)
        if text:
            # Skip if entire text is inside a link (e.g. "Sign up now!")
            a_text = "".join(a.get_text(" ", strip=True) for a in sf_el.find_all("a"))
            is_pure_link = a_text and a_text == text
            is_cta = text.lower().startswith("sign up") or "join the conversation" in text.lower()
            if not is_pure_link and not is_cta:
                standfirst_text = text

    # Hero image from data-gu-name="media"
    hero_img_src = None
    media_div = soup.find(attrs={"data-gu-name": "media"})
    if media_div:
        pic = media_div.find("picture")
        if pic:
            img_tag = pic.find("img")
            if img_tag:
                hero_img_src = img_tag.get("src", "")

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

    for el in collect_elements(body):
        if is_trailing_element(el):
            break

        # Include figures containing a picture.
        if el.name == "figure":
            pic = el.find("picture")
            if pic:
                img = pic.find("img")
                if img:
                    src = img.get("src", "")
                    if src:
                        figcap = el.find("figcaption")
                        cap_text = ""
                        if figcap:
                            cap_text = figcap.get_text(" ", strip=True)
                        cap_html = f'<figcaption>{cap_text}</figcaption>' if cap_text else ""
                        sentences.append(cap_text)
                        html_parts.append({"tag": "figure", "html": f'<img src="{src}" style="max-width:100%">{cap_html}'})
                        labels.append(0)
                        title_labels.append(0)
                        first_elem = False
            continue

        # Replace decorative SVGs with quote character before extracting text
        for svg in el.find_all("svg"):
            svg.replace_with("\u201c")
        text = el.get_text(" ", strip=True)
        if not text:
            continue

        if el.name in ("h2", "h3", "h4") and is_noise_heading(text):
            continue
        if any(text.lower().startswith(p) for p in SKIP_PARA_PREFIXES):
            continue

        is_heading = el.name in ("h2", "h3", "h4") and len(text) <= 200

        sentences.append(text)
        html_parts.append({"tag": el.name, "html": clean_inner_html(el)})
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
        "outlet": "GuardianInt",
        "title": title,
        "url": meta.get("url", ""),
        "sentences": sentences,
        "html_parts": html_parts,
        "labels": labels,
        "title_labels": title_labels,
        "num_sections": labels.count(1) + 1,
    }


def main():
    articles = read_articles("guardianint_subheadings", process_article)
    write_articles("guardian_segments.json", unique_articles(articles))


if __name__ == "__main__":
    main()
