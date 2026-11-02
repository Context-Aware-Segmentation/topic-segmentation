"""News construction paths relative to TS_NEWS_WORKDIR (default: runs/news_construction/manual)."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("TS_NEWS_WORKDIR", str(REPOSITORY / "runs/news_construction/manual"))).resolve()
