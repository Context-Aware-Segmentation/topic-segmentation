"""Extract editorial sections from saved Sky News pages."""


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
    "more from sky news",
    "advertisement",
    "more on ",
]

HEADING_NOISE_PREFIXES = (
    "read more:", "latest:", "analysis:", "politics hub", "watch:",
    "follow latest:", "follow:", "live updates:", "breaking:",
    "politics latest:", "politics live:", "follow live:",
)
HEADING_NOISE_FULL = {"read more", "more from sky news", "advertisement"}

TRAILING_TEXT_PREFIXES = (
    "related topics",
    "more from sky news",
)

SKIP_PARA_PREFIXES = (
    "sign up",
    "i would like to be emailed",
    "read more",
    "politics live:",
    "politics latest:",
    "politics hub",
    "follow latest:",
    "latest:",
    "live updates:",
    "follow live:",
)

SKIP_DIV_CLASSES = {
    "sdc-site-outbrain",
    "ad--teads",
    "sdc-site-layout-sticky-region",
    "sdc-article-related-stories",
    "sdc-site-video",
    "sdc-article-strapline",
    "sdc-article-tweet",
    "sdc-article-instagram",
    "sdc-article-app-promo",
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


def is_link_only_p(el) -> bool:
    """Identify paragraphs containing only a link, optionally wrapped in strong."""
    if el.name != "p":
        return False
    direct = "".join(str(c) for c in el.children if c.name is None).strip()
    if direct:
        return False
    children = [c for c in el.children if c.name is not None]
    if len(children) != 1:
        return False
    only = children[0]
    if only.name == "a":
        return True
    if only.name == "strong":
        inner = [c for c in only.children if c.name is not None]
        if len(inner) == 1 and inner[0].name == "a":
            wrap_direct = "".join(str(c) for c in only.children if c.name is None).strip()
            if not wrap_direct:
                return True
    return False


def is_trailing_element(el) -> bool:
    text_lower = el.get_text(" ", strip=True).lower()
    if any(text_lower.startswith(p) for p in TRAILING_TEXT_PREFIXES):
        return True
    return False


def is_strong_heading(el) -> bool:
    """Recognize a `<p><strong>` heading of at most 100 characters."""
    if el.name != "p":
        return False
    # NavigableString has .name == None; Tags have .name set to the tag name.
    direct_text = "".join(str(c) for c in el.children if c.name is None).strip()
    if direct_text:
        return False
    children = [c for c in el.children if c.name is not None]
    if len(children) != 1 or children[0].name != "strong":
        return False
    txt = el.get_text(" ", strip=True)
    # Short bold leads and Q&A answers serve as headings, including those ending in punctuation.
    if len(txt) > 100:
        return False
    return True


def is_heading_noise(txt: str) -> bool:
    tl = txt.lower()
    if any(tl.startswith(n) for n in HEADING_NOISE_PREFIXES):
        return True
    if tl in HEADING_NOISE_FULL:
        return True
    return False


def collect_elements(body) -> list:
    """Collect content elements from sdc-article-body."""
    elements = []
    for el in body.children:
        if not hasattr(el, "name") or el.name is None:
            continue
        if is_skip_container(el):
            continue

        cls = set(el.get("class") or [])

        if el.name == "div" and "sdc-article-image" in cls:
            fig = el.find("figure")
            if fig:
                img = fig.find("img")
                if img and (img.get("src") or "").startswith("http"):
                    elements.append(fig)
            continue

        if el.name == "div" and "sdc-article-factbox" in cls:
            elements.append(("factbox", el))
            continue

        if el.name in ("p", "ul", "ol", "blockquote"):
            if not is_link_only_list(el):
                elements.append(el)
            continue


    return elements


def process_article(aid: str, raw_html: str, meta: dict):
    soup = BeautifulSoup(ftfy.fix_text(raw_html), "html.parser")

    h1 = soup.find("h1", class_="sdc-article-header__title")
    if not h1:
        h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else meta.get("title", "")

    standfirst_text = None
    sf = soup.find("p", class_="sdc-article-header__sub-title")
    if sf:
        text = sf.get_text(" ", strip=True)
        if text and not text.lower().startswith("sign up"):
            standfirst_text = text

    body = soup.find("div", class_="sdc-article-body")
    if not body:
        return None

    # Hero: first sdc-article-image div in body
    hero_img_src = None
    hero_img_cap = None
    first_img_div = body.find("div", class_="sdc-article-image")
    if first_img_div:
        fig = first_img_div.find("figure")
        if fig:
            img = fig.find("img")
            if img and (img.get("src") or "").startswith("http"):
                hero_img_src = img["src"]
                figcap = fig.find("figcaption")
                if figcap:
                    hero_img_cap = figcap.get_text(" ", strip=True)

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

    hero_consumed = False  # skip first sdc-article-image (already used as hero)
    first_elem = not (standfirst_text or hero_img_src)

    # data-highlight-intro marks the first paragraph as an extended standfirst.
    highlight_intro = body.get("data-highlight-intro") == "true"
    intro_p_consumed = not highlight_intro

    for el in collect_elements(body):
        if is_trailing_element(el if not isinstance(el, tuple) else el[1]):
            break

        if isinstance(el, tuple) and el[0] == "factbox":
            fb = el[1]
            fb_title = fb.find("h4")
            fb_text = fb.get_text(" ", strip=True)
            if not fb_text:
                continue
            title_html = f"<strong>{fb_title.get_text(' ', strip=True)}</strong><br>" if fb_title else ""
            sentences.append(fb_text)
            html_parts.append({"tag": "blockquote", "html": title_html + fb_text})
            labels.append(0)
            title_labels.append(0)
            first_elem = False
            continue

        # Figure (body image) — skip hero on first encounter
        if el.name == "figure":
            img = el.find("img")
            if not img:
                continue
            src = img.get("src", "")
            if not src:
                continue
            if not hero_consumed and hero_img_src and src == hero_img_src:
                hero_consumed = True
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

        # Remove link-only promotional paragraphs.
        if is_link_only_p(el):
            continue

        if is_strong_heading(el):
            text = el.get_text(" ", strip=True)
            if not text or is_noise_heading(text) or is_heading_noise(text):
                continue
            sentences.append(text)
            html_parts.append({"tag": "h3", "html": text})
            labels.append(0 if first_elem else 1)
            title_labels.append(1)
            first_elem = False
            continue

        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if any(text.lower().startswith(p) for p in SKIP_PARA_PREFIXES):
            continue

        # First body <p> in a data-highlight-intro="true" article = styled lede
        is_intro = el.name == "p" and not intro_p_consumed
        if is_intro:
            intro_p_consumed = True

        sentences.append(text)
        html_parts.append({"tag": el.name, "html": clean_inner_html(el, "https://news.sky.com")})
        labels.append(0)
        title_labels.append(1 if is_intro else 0)
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
        "outlet": "SkyNews",
        "title": title,
        "url": meta.get("url", ""),
        "sentences": sentences,
        "html_parts": html_parts,
        "labels": labels,
        "title_labels": title_labels,
        "num_sections": labels.count(1) + 1,
    }


def main():
    articles = read_articles("skynews_subheadings", process_article)
    write_articles("skynews_segments.json", unique_articles(articles))


if __name__ == "__main__":
    main()
