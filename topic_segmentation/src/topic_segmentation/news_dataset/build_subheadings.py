#!/usr/bin/env python3
"""Select saved articles with at least two content subheadings.

Copy matching pages from `data/pages/raw_html/` to `data/pages/<outlet>_subheadings/`,
using outlet-specific HTML selectors and promo filters.

    python -m topic_segmentation.news_dataset.build_subheadings BBC
    python -m topic_segmentation.news_dataset.build_subheadings all
"""
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
import json
import re
import shutil

import ftfy
from bs4 import BeautifulSoup

from .paths import ROOT
RAW_HTML_DIR = ROOT / "data" / "pages" / "raw_html"
ARTICLES_JSON = ROOT / "data" / "articles.json"

# Noise heading patterns

COMMON_NOISE = [
    r"related topics", r"more on this story", r"related internet links",
    r"you may also like", r"you may also be interested in",
    r"you might also be interested in", r"more on ", r"read next", r"read more",
    r"sign up", r"newsletter", r"to play this video", r"trending",
    r"^comments?$", r"share this article", r"recommended", r"popular",
    r"click to subscribe", r"follow .+ on", r"most read$",
    r"get in touch", r"how is the .+ affecting you",
    r"the bbc is not responsible", r"explore more on the topics",
]

ES_NOISE = [
    r"^read more$", r"^most read$", r"^comments?$", r"^share$",
    r"sign up", r"newsletter", r"follow .+ on", r"get in touch",
    r"why you can trust es", r"why es best",
]

COMMON_RE = re.compile("|".join(COMMON_NOISE), re.IGNORECASE)
ES_RE = re.compile("|".join(ES_NOISE), re.IGNORECASE)


def is_noise(text: str, noise_re: re.Pattern) -> bool:
    t = text.strip()
    return len(t) < 3 or bool(noise_re.search(t))


# Subheading detection

def has_subheadings_default(soup) -> bool:
    """Find at least two content h2/h3 headings inside article."""
    article = soup.select_one("article")
    if not article:
        return False
    real = sum(
        1 for h in article.find_all(["h2", "h3"])
        if not is_noise(h.get_text(strip=True), COMMON_RE)
    )
    return real >= 2


def has_subheadings_es(soup) -> bool:
    """Find Evening Standard headings in #main, excluding empty Opta articles."""
    widgets = soup.find_all("opta-widget")
    if widgets and all(len(w.get_text(strip=True)) == 0 for w in widgets):
        return False
    main = soup.find(id="main")
    if not main:
        return False
    real = sum(
        1 for h in main.find_all(["h2", "h3", "h4"])
        if not is_noise(h.get_text(strip=True), ES_RE)
    )
    return real >= 2


def has_subheadings_independent(soup) -> bool:
    """Find Independent section headings in div.sc-kk992l-0."""
    article = soup.find("article")
    if not article:
        return False
    main_wrapper = article.find("div", class_=lambda c: c and "main-wrapper" in c)
    if not main_wrapper:
        return False
    content = main_wrapper.find("div", class_=lambda c: c and "sc-jbiisr-0" in c)
    if not content:
        return False
    real = sum(
        1 for d in content.find_all("div", class_=lambda c: c and "sc-kk992l-0" in c)
        if d.find(["h2", "h3", "h4"])
        and not is_noise(d.get_text(strip=True), COMMON_RE)
        and len(d.get_text(strip=True)) <= 200
    )
    return real >= 2


def has_subheadings_inews(soup) -> bool:
    """Find iNews headings in div.article-content, excluding Read Next widgets."""
    content = soup.find("div", class_="article-content")
    if not content:
        return False
    real = 0
    for h in content.find_all(["h2", "h3"]):
        if h.find_parent(class_="inews__shortcode-relatedarticleinline__content"):
            continue
        if h.find_parent(class_="inews__shortcode-relatedarticleinline"):
            continue
        if is_noise(h.get_text(strip=True), COMMON_RE):
            continue
        real += 1
    return real >= 2


SKY_NEWS_NOISE_PREFIXES = (
    "read more:", "latest:", "analysis:", "politics hub", "watch:",
    "follow latest:", "follow:", "live updates:", "breaking:", "politics latest:",
)
SKY_NEWS_NOISE_FULL = {"read more", "more from sky news", "advertisement"}


def is_sky_news_strong_heading(el) -> bool:
    """Recognize a `<p><strong>` heading of at most 100 characters."""
    if el.name != "p":
        return False
    direct_text = "".join(str(c) for c in el.children if c.name is None).strip()
    if direct_text:
        return False
    children = [c for c in el.children if c.name is not None]
    if len(children) != 1 or children[0].name != "strong":
        return False
    txt = el.get_text(" ", strip=True)
    if len(txt) > 100:
        return False
    return True


def has_subheadings_sky_news(soup) -> bool:
    """Find Sky News bold paragraph headings in div.sdc-article-body."""
    body = soup.find("div", class_="sdc-article-body")
    if not body:
        return False
    real = 0
    for p in body.find_all("p"):
        if not is_sky_news_strong_heading(p):
            continue
        txt = p.get_text(" ", strip=True)
        tl = txt.lower()
        if any(tl.startswith(n) for n in SKY_NEWS_NOISE_PREFIXES):
            continue
        if tl in SKY_NEWS_NOISE_FULL:
            continue
        real += 1
    return real >= 2


def has_subheadings(html: str, outlet: str) -> bool:
    soup = BeautifulSoup(ftfy.fix_text(html), "html.parser")
    if outlet == "EveningStandard":
        return has_subheadings_es(soup)
    if outlet == "Independent":
        return has_subheadings_independent(soup)
    if outlet == "INews":
        return has_subheadings_inews(soup)
    if outlet == "SkyNews":
        return has_subheadings_sky_news(soup)
    return has_subheadings_default(soup)


# Main

# Map outlet names to output directory names
OUTLET_DIR_NAMES = {
    "BBC": "bbc_subheadings",
    "EveningStandard": "eveningstandard_subheadings",
    "GuardianInt": "guardianint_subheadings",
    "INews": "inews_subheadings",
    "Independent": "independent_subheadings",
    "SkyNews": "skynews_subheadings",
}


def inspect_article(aid, outlet):
    source = RAW_HTML_DIR / f"{aid}.html"
    if not source.exists():
        return aid, False
    return aid, has_subheadings(source.read_text(encoding="utf-8"), outlet)


def build_outlet(outlet: str, articles: list) -> None:
    dir_name = OUTLET_DIR_NAMES.get(outlet, f"{outlet.lower()}_subheadings")
    output_dir = ROOT / "data" / "pages" / dir_name

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Deduplicate by URL, keep earliest articleId
    seen_urls: set = set()
    ids: set = set()
    for a in sorted(articles, key=lambda x: x["articleId"]):
        if a.get("outlet") != outlet:
            continue
        url = a.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            ids.add(a["articleId"])
    print(f"{outlet} articles (deduplicated): {len(ids)}", flush=True)

    found = 0
    workers = max(1, int(os.environ.get("TS_NEWS_WORKERS", "1")))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for aid, keep in executor.map(partial(inspect_article, outlet=outlet), sorted(ids)):
            if keep:
                shutil.copy2(RAW_HTML_DIR / f"{aid}.html", output_dir / f"{aid}.html")
                found += 1

    print(f"Copied {found} articles with subheadings -> {output_dir}/", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build <outlet>_subheadings/ directory")
    parser.add_argument(
        "outlet",
        help="Outlet name as in articles.json (e.g. BBC, EveningStandard, GuardianInt, "
             "INews, Independent, SkyNews) — or 'all' to run every outlet",
    )
    args = parser.parse_args()

    with open(ARTICLES_JSON, encoding="latin-1") as f:
        articles = json.load(f)

    outlets = list(OUTLET_DIR_NAMES) if args.outlet == "all" else [args.outlet]
    for i, outlet in enumerate(outlets):
        if i > 0:
            print()
        build_outlet(outlet, articles)


if __name__ == "__main__":
    main()
