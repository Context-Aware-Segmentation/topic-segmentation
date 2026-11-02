"""Build the three documentation sites into one Cloudflare Pages directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "topic_segmentation": "topic-segmentation",
    "knowledge_graph": "knowledge-graph",
    "recommendation": "recommendation",
}
PAGES = ("index.html", "404.html", "_redirects")


def build(output):
    output = output.resolve()
    if output.exists():
        unexpected = {p.name for p in output.iterdir()} - set(MODULES.values()) - set(PAGES)
        if unexpected:
            raise ValueError(f"Output contains unrelated files: {sorted(unexpected)}")
    output.mkdir(parents=True, exist_ok=True)
    for folder, slug in MODULES.items():
        subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(output / slug)],
            cwd=ROOT / folder,
            check=True,
        )
    for name in PAGES:
        shutil.copyfile(ROOT / "docs" / name, output / name)
    print(f"Documentation ready: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "tmp/mt-docs-site")
    build(parser.parse_args().output)
