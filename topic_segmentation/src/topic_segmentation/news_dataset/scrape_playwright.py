#!/usr/bin/env python3
"""Download JavaScript-rendered article HTML with Playwright.

Existing pages are overwritten unless --skip-existing is set. Navigation failures
are retried once; redirect loops are resolved with requests before retrying.

    python -m topic_segmentation.news_dataset.scrape_playwright --outlet EveningStandard
    python -m topic_segmentation.news_dataset.scrape_playwright --ids ids.txt --skip-existing

After activating the environment, install Chromium with playwright install chromium.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests as _requests
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

from .paths import ROOT



def resolve_final_url(url: str, timeout: int = 15) -> str:
    """Follow up to 60 HTTP redirects and return the resolved URL."""
    session = _requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    visited: set = set()
    current = url
    for _ in range(60):
        if current in visited:
            break
        visited.add(current)
        try:
            resp = session.get(current, allow_redirects=False, timeout=timeout)
        except Exception:
            break
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location", "")
            if not loc:
                break
            current = urljoin(current, loc)
        else:
            break
    return current



async def _accept_consent(page) -> int:
    """Accept page and widget consent dialogs; return the number of buttons clicked."""
    clicks = 0
    # Global CMP consent (standard.co.uk uses cmpv2.standard.co.uk iframe)
    for frame in page.frames:
        if "cmpv2" in frame.url or "consent" in frame.url:
            try:
                btn = await frame.query_selector("button:has-text(\"Accept\")")
                if btn and await btn.is_visible():
                    await btn.click()
                    clicks += 1
                    await page.wait_for_timeout(1500)
                    break
            except Exception:
                pass
    # Per-widget "Allow and Continue" buttons (e.g. Opta)
    for btn in await page.query_selector_all("button:has-text(\"Allow and Continue\")"):
        try:
            if await btn.is_visible():
                await btn.click(timeout=2000)
                clicks += 1
        except Exception:
            pass
    if clicks:
        await page.wait_for_timeout(2000)
    return clicks


async def scrape_one(page, article: dict, output_dir: Path, timeout_ms: int):
    aid = article["articleId"]
    url = article.get("url", "")
    out_path = output_dir / f"{aid}.html"

    if not url:
        return (aid, "failed", "no URL")

    async def fetch(target_url: str, t_ms: int):
        await page.goto(target_url, wait_until="networkidle", timeout=t_ms)
        await _accept_consent(page)
        html = await page.content()
        # Reject pages whose Opta widgets are all empty.
        if "<opta-widget" in html:
            rendered = await page.evaluate(
                "() => Array.from(document.querySelectorAll('opta-widget'))"
                ".every(w => w.innerHTML.trim() === '')"
            )
            if rendered:
                raise ValueError("opta_widget_unrendered")
        out_path.write_text(html, encoding="utf-8")
        return html

    try:
        html = await fetch(url, timeout_ms)
        return (aid, "ok", len(html))
    except PlaywrightTimeout:
        # Save partial HTML if enough content has rendered.
        try:
            html = await page.content()
            if len(html) > 5000:
                out_path.write_text(html, encoding="utf-8")
                return (aid, "ok_partial", len(html))
        except Exception:
            pass
        # Retry once with 2× timeout
        try:
            html = await fetch(url, timeout_ms * 2)
            return (aid, "ok_retry", len(html))
        except Exception as e:
            return (aid, "failed", f"timeout after retry: {str(e)[:60]}")
    except Exception as e:
        err = str(e)
        # Reject pages with empty Opta widgets.
        if "opta_widget_unrendered" in err:
            return (aid, "failed", "opta_widget_blocked")
        # ERR_TOO_MANY_REDIRECTS — resolve the chain and retry
        if "ERR_TOO_MANY_REDIRECTS" in err:
            try:
                final_url = await asyncio.get_event_loop().run_in_executor(
                    None, resolve_final_url, url
                )
                html = await fetch(final_url, timeout_ms)
                return (aid, "ok_redirected", len(html))
            except Exception as e2:
                return (aid, "failed", f"redirect+retry failed: {str(e2)[:60]}")
        # Retry other navigation errors once.
        try:
            html = await fetch(url, timeout_ms * 2)
            return (aid, "ok_retry", len(html))
        except Exception as e2:
            return (aid, "failed", f"{type(e2).__name__}: {str(e2)[:60]}")


# Worker (one browser context / tab)

async def worker(browser, articles, output_dir, timeout_ms, results):
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 800},
    )
    page = await context.new_page()
    await page.route(
        "**/*",
        lambda route: route.abort()
        if route.request.resource_type in ("media", "font")
        else route.continue_(),
    )
    for article in articles:
        result = await scrape_one(page, article, output_dir, timeout_ms)
        results.append(result)
    await context.close()



async def main_async(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="latin-1") as f:
        all_articles = json.load(f)

    if args.outlet:
        all_articles = [a for a in all_articles if a.get("outlet") == args.outlet]
        print(f"Filtered to outlet '{args.outlet}': {len(all_articles)} articles")

    if args.ids:
        id_set = set(Path(args.ids).read_text().split())
        all_articles = [a for a in all_articles if a["articleId"] in id_set]
        print(f"Filtered to {len(all_articles)} articles from ID list")

    # Deduplicate by URL (keep earliest articleId)
    seen_urls: set = set()
    deduped = []
    for a in sorted(all_articles, key=lambda x: x["articleId"]):
        url = a.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(a)
    print(f"After URL dedup: {len(deduped)} articles")

    if args.skip_existing:
        existing = {p.stem for p in output_dir.glob("*.html")}
        deduped = [a for a in deduped if a["articleId"] not in existing]
        print(f"Skipping existing, to scrape: {len(deduped)}")

    if not deduped:
        print("Nothing to do!")
        return []

    n = args.workers
    chunks = [deduped[i::n] for i in range(n)]
    timeout_ms = args.timeout * 1000
    results = []
    start = time.time()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        await asyncio.gather(*[
            worker(browser, chunk, output_dir, timeout_ms, results)
            for chunk in chunks if chunk
        ])
        await browser.close()

    elapsed = time.time() - start
    ok = sum(1 for _, s, _ in results if s.startswith("ok"))
    failed = [(aid, d) for aid, s, d in results if s == "failed"]

    print(f"\nDone in {elapsed:.0f}s ({len(deduped)/elapsed:.1f} articles/s)")
    print(f"  OK: {ok}  Failed: {len(failed)}")

    from collections import Counter
    statuses = Counter(s for _, s, _ in results)
    for status, count in statuses.most_common():
        print(f"    {status}: {count}")

    return [{"articleId": a, "reason": d} for a, d in failed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         default=str(ROOT / "data/articles.json"))
    parser.add_argument("--output-dir",    default=str(ROOT / "data/pages/raw_html"))
    parser.add_argument("--outlet",        default=None,  help="Filter by outlet name")
    parser.add_argument("--ids",           default=None,  help="File with article IDs to scrape")
    parser.add_argument("--workers",       type=int, default=3,  help="Parallel browser tabs")
    parser.add_argument("--timeout",       type=int, default=30, help="Per-page timeout (seconds)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip articles already in output-dir")
    args = parser.parse_args()
    failed = asyncio.run(main_async(args))
    if failed:
        fail_path = ROOT / "data/scrape_failures_playwright.json"
        fail_path.write_text(
            json.dumps(failed, indent=2), encoding="utf-8"
        )
        print(f"  Failures → {fail_path}")


if __name__ == "__main__":
    main()
