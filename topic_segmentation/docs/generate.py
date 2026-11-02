"""Build guide pages and API navigation from the repository files."""
import ast
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src/topic_segmentation"

for source, destination in (
    ("README.md", "index.md"),
    ("SETUP.md", "SETUP.md"),
    ("data/sample_input.jsonl", "data/sample_input.jsonl"),
):
    with mkdocs_gen_files.open(destination, "w") as output:
        # The site root is docs/, so repository links into docs/ lose that prefix.
        output.write((ROOT / source).read_text(encoding="utf-8").replace("](docs/", "]("))

nav = mkdocs_gen_files.Nav()
nav["Overview"] = "index.md"
groups = {
    (): "Core",
    ("data",): "Data preparation",
    ("data", "mimic"): "Clinical data",
    ("news_dataset",): "News construction",
}
index = ["# API reference\n", "Generated from the Python source and docstrings.\n"]

for directory, group in groups.items():
    index.append(f"\n## {group}\n")
    for path in sorted(SOURCE.joinpath(*directory).glob("*.py")):
        if path.stem.startswith("_"):
            continue
        relative = path.relative_to(SOURCE)
        module = ".".join(("topic_segmentation", *relative.with_suffix("").parts))
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
