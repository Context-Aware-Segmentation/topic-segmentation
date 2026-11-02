#!/usr/bin/env python3
"""Download article HTML using requests and Playwright.

Read data/articles.json and save pages to data/pages/raw_html/.
Use Playwright for Evening Standard and requests for other outlets.
Existing pages are skipped by default.

    python -m topic_segmentation.news_dataset.scrape_all
    python -m topic_segmentation.news_dataset.scrape_all --outlet BBC --workers 10 --timeout 20
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .paths import ROOT

# Outlets requiring JavaScript rendering
PLAYWRIGHT_OUTLETS = {"EveningStandard"}


# Phase 1: requests-based scraper

def scrape_article(session, article, output_dir, timeout):
    """Save article HTML and return (articleId, status, detail)."""
    aid = article["articleId"]
    url = article.get("url", "")
    out_path = output_dir / f"{aid}.html"

    if not url:
        return (aid, "failed", "no URL")

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        out_path.write_text(resp.text, encoding="utf-8")
        return (aid, "ok", resp.status_code)
    except requests.exceptions.HTTPError:
        return (aid, "failed", f"HTTP {resp.status_code}")
    except requests.exceptions.Timeout:
        return (aid, "failed", "timeout")
    except requests.exceptions.ConnectionError as e:
        return (aid, "failed", f"connection error: {str(e)[:80]}")
    except Exception as e:
        return (aid, "failed", f"{type(e).__name__}: {str(e)[:80]}")


# Main

def main():
    parser = argparse.ArgumentParser(description="Scrape all article HTML (requests + Playwright)")
    parser.add_argument("--input",               default=str(ROOT / "data/articles.json"))
    parser.add_argument("--output-dir",          default=str(ROOT / "data/pages/raw_html"))
    parser.add_argument("--workers",             type=int, default=8,
                        help="Concurrent workers for requests phase")
    parser.add_argument("--timeout",             type=int, default=15,
                        help="Per-request timeout (seconds)")
    parser.add_argument("--playwright-workers",  type=int, default=3,
                        help="Playwright browser tabs for EveningStandard phase")
    parser.add_argument("--playwright-timeout",  type=int, default=30,
                        help="Playwright per-page timeout (seconds)")
    parser.add_argument("--force",               action="store_true",
                        help="Re-scrape articles even if HTML already exists")
    parser.add_argument("--outlet",              default=None,
                        help="Restrict to one outlet (both phases filter independently)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="latin-1") as f:
        all_articles = json.load(f)

    if args.outlet:
        all_articles = [a for a in all_articles if a.get("outlet") == args.outlet]
        print(f"Filtered to outlet '{args.outlet}': {len(all_articles)} articles")

    static_articles   = [a for a in all_articles if a.get("outlet") not in PLAYWRIGHT_OUTLETS]
    playwright_articles = [a for a in all_articles if a.get("outlet") in PLAYWRIGHT_OUTLETS]

    print(f"Total articles : {len(all_articles)}")
    print(f"  requests     : {len(static_articles)}")
    print(f"  Playwright   : {len(playwright_articles)}")

    total_start = time.time()

    # Phase 1: requests
    existing = {p.stem for p in output_dir.glob("*.html")}
    if args.force:
        to_scrape = static_articles
    else:
        to_scrape = [a for a in static_articles if a["articleId"] not in existing]

    print(f"\nPhase 1 (requests): {len(to_scrape)} articles to scrape")

    ok_count = 0
    failures = []
    phase1_start = time.time()

    if to_scrape:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(scrape_article, session, a, output_dir, args.timeout): a
                for a in to_scrape
            }
            for i, future in enumerate(as_completed(futures), 1):
                aid, status, detail = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "failed":
                    article = futures[future]
                    failures.append({
                        "articleId": aid,
                        "url": article.get("url", ""),
                        "outlet": article.get("outlet", ""),
                        "reason": detail,
                    })
                if i % 100 == 0 or i == len(to_scrape):
                    elapsed = time.time() - phase1_start
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"  [{i}/{len(to_scrape)}] ok={ok_count} fail={len(failures)} "
                          f"({rate:.1f}/s, {elapsed:.0f}s)")

    print(f"Phase 1 done in {time.time() - phase1_start:.0f}s  "
          f"ok={ok_count}  failed={len(failures)}")

    if failures:
        from collections import Counter
        for reason, count in Counter(f["reason"].split(":")[0] for f in failures).most_common():
            print(f"    {reason}: {count}")

    # Phase 2: Playwright (EveningStandard)
    if playwright_articles:
        print(f"\nPhase 2 (Playwright / EveningStandard): {len(playwright_articles)} articles")
        try:
            from .scrape_playwright import main_async as playwright_main
        except ImportError:
            print("  scrape_playwright.py not found — skipping Playwright phase.")
            print("  Run: pip install playwright && playwright install chromium")
        else:
            import argparse as _ap
            pw_args = _ap.Namespace(
                input         = args.input,
                output_dir    = str(output_dir),
                outlet        = "EveningStandard",
                ids           = None,
                workers       = args.playwright_workers,
                timeout       = args.playwright_timeout,
                skip_existing = not args.force,
            )
            pw_start = time.time()
            pw_failures = asyncio.run(playwright_main(pw_args))
            failures.extend(pw_failures)
            print(f"Phase 2 done in {time.time() - pw_start:.0f}s")

    failure_path = ROOT / "data/scrape_failures.json"
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    if failures:
        print(f"  Total failures: {len(failures)} → {failure_path}")

    print(f"\nTotal time: {time.time() - total_start:.0f}s")


if __name__ == "__main__":
    main()
