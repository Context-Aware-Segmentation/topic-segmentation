"""Build guide pages and API navigation from the repository files."""
import ast
import re
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src/knowledge_graph"

for source, destination in (
    ("README.md", "index.md"),
    ("SETUP.md", "SETUP.md"),
    ("docs/ARCHITECTURE.md", "ARCHITECTURE.md"),
):
    with mkdocs_gen_files.open(destination, "w") as output:
        text = (ROOT / source).read_text(encoding="utf-8")
        text = text.replace("](docs/", "](").replace("](README.md", "](index.md")
        text = text.replace("](../SETUP.md", "](SETUP.md").replace("](../README.md", "](index.md")
        text = re.sub(r"src/knowledge_graph/([\w/]+)\.py", r"api/\1.md", text)
        text = text.replace("src/knowledge_graph/clinical/vendor/phenobert/", "https://github.com/EclipseCN/PhenoBERT")
        text = text.replace("../topic_segmentation/README.md", "/topic-segmentation/index.html")
        text = text.replace("../topic_segmentation/SETUP.md", "/topic-segmentation/SETUP.html")
        output.write(text)


nav = mkdocs_gen_files.Nav()
nav["Overview"] = "index.md"
groups = {(): "Core", ("news",): "News", ("clinical",): "Clinical"}
index = ["# API reference\n", "Generated from the Python source and docstrings.\n"]

for directory, group in groups.items():
    index.append(f"\n## {group}\n")
    for path in sorted(SOURCE.joinpath(*directory).glob("*.py")):
        if path.stem.startswith("_"):
            continue
        relative = path.relative_to(SOURCE)
        module = ".".join(("knowledge_graph", *relative.with_suffix("").parts))
        page = relative.with_suffix(".md")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        members = [node.name for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and not node.name.startswith("_")]
        nav[group, path.stem] = page.as_posix()
        index.append(f"- [{path.stem}]({page.as_posix()})\n")
        with mkdocs_gen_files.open(Path("api") / page, "w") as output:
            output.write(f"# {path.stem}\n\n`{module}`\n\n::: {module}\n")
            output.write(f"    options:\n      members: {members!r}\n")

with mkdocs_gen_files.open("api/index.md", "w") as output:
    output.write("\n".join(index))
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as output:
    output.writelines(nav.build_literate_nav())
