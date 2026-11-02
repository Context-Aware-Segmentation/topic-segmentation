"""Generate recommendation guides and API pages without importing model dependencies."""
import ast
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src/news_recommendation"

for source, destination in (("README.md", "index.md"), ("SETUP.md", "SETUP.md")):
    with mkdocs_gen_files.open(destination, "w") as output:
        text = (ROOT / source).read_text(encoding="utf-8")
        text = text.replace("](docs/", "](").replace("](README.md", "](index.md")
        text = text.replace("../topic_segmentation/SETUP.md", "/topic-segmentation/SETUP.html")
        output.write(text)

for name in ("plm_nr.json", "data_preparation.json", "segmentation.json"):
    with mkdocs_gen_files.open(Path("configs") / name, "w") as output:
        output.write((ROOT / "configs" / name).read_text(encoding="utf-8"))

groups = {
    "Workflow": ("pipeline", "paths"),
    "Data preparation": ("preparation",),
    "Segmentation and summaries": ("segmentation", "segment_summaries", "article_summaries", "jobs"),
    "Training and evaluation": ("model", "training"),
}
nav = mkdocs_gen_files.Nav()
nav["Overview"] = "index.md"
index = ["# API reference\n", "Generated from the Python source and docstrings.\n"]

for group, modules in groups.items():
    index.append(f"\n## {group}\n")
    for name in modules:
        tree = ast.parse((SOURCE / f"{name}.py").read_text(encoding="utf-8"))
        members = [node.name for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and not node.name.startswith("_")]
        nav[group, name] = f"{name}.md"
        index.append(f"- [{name}]({name}.md)\n")
        with mkdocs_gen_files.open(f"api/{name}.md", "w") as output:
            output.write(f"# {name}\n\n`news_recommendation.{name}`\n\n::: news_recommendation.{name}\n")
            output.write(f"    options:\n      members: {members!r}\n")

with mkdocs_gen_files.open("api/index.md", "w") as output:
    output.write("\n".join(index))
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as output:
    output.writelines(nav.build_literate_nav())
