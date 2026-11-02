# Build documentation

Published site: [Topic Segmentation documentation](https://mt-docs.pages.dev/topic-segmentation/).

Run from the repository's `topic_segmentation/` directory:

```bash
uvx --with-requirements docs/requirements.txt mkdocs build --strict
```

Open `../../tmp/topic-segmentation-docs/index.html` in a browser. To preview with live updates:

```bash
uvx --with-requirements docs/requirements.txt mkdocs serve
```

The preview runs at `http://127.0.0.1:8000`.

The user guide and setup pages come from `README.md` and `SETUP.md`. API pages
come from the source and docstrings in `src/topic_segmentation/`. Rebuild after
editing those files.

Documentation tools run in a separate uv tool environment. Building the site
requires neither the training environment nor model weights.

To preview or publish all three modules together, follow the repository's `docs/README.md`.
