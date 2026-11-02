# Build documentation

Published site: [Knowledge Graph Construction documentation](https://mt-docs.pages.dev/knowledge-graph/).

Run from the repository's `knowledge_graph/` directory:

```bash
uvx --with-requirements docs/requirements.txt mkdocs build --strict
```

Open `../../tmp/knowledge-graph-docs/index.html` in a browser. To preview with live updates:

```bash
uvx --with-requirements docs/requirements.txt mkdocs serve
```

The preview runs at `http://127.0.0.1:8000`.

The user guide and setup pages come from `README.md` and `SETUP.md`. API pages
come from the source and docstrings in `src/knowledge_graph/`. Rebuild after
editing those files.

Documentation tools run in a separate uv tool environment. Building the site
requires neither the training environment nor model weights.

To preview or publish all three modules together, follow the repository's `docs/README.md`.
