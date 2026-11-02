"""Stage 1 (news): rebuild IDEA-seg from saved article pages.

    python scripts/build_news.py

Reads the article list and pages from data/news/raw/ using configs/news_construction.json.
Writes the editorial dataset (data/dataset.jsonl), paragraph splits (prepared/news),
and sentence splits (prepared/news_sent) under runs/news_construction/<time>/.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPOSITORY / "configs/news_construction.json").read_text())


def main():
    source = REPOSITORY / CONFIG["source_root"]
    run = REPOSITORY / CONFIG["run_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (run / "data/pages").mkdir(parents=True)
    (run / "logs").mkdir()
    for saved, link in ((source / "articles.json", run / "data/articles.json"),
                        (source / "pages/raw_html", run / "data/pages/raw_html")):
        link.symlink_to(os.path.relpath(saved, link.parent), target_is_directory=saved.is_dir())
    environment = dict(os.environ, NLTK_DATA=str(REPOSITORY / "data/resources/nltk"), TS_NEWS_WORKDIR=str(run),
                       TS_NEWS_WORKERS=str(CONFIG["html_workers"]))
    for step in CONFIG["steps"]:
        command = [sys.executable, "-m", f"topic_segmentation.news_dataset.{step}", *(["all"] if step == "build_subheadings" else [])]
        print(f"running {step}", flush=True)
        with (run / "logs" / f"{step}.log").open("w") as log:
            subprocess.run(command, cwd=REPOSITORY, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"wrote {run / 'data/dataset.jsonl'} and {run / 'prepared'}")


if __name__ == "__main__":
    main()
