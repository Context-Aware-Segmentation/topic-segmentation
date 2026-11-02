# Build and publish documentation

The [documentation site](https://mt-docs.pages.dev/) hosts three independently generated MkDocs sites:

| Module | URL path |
|---|---|
| Topic segmentation | `/topic-segmentation/` |
| Knowledge graph construction | `/knowledge-graph/` |
| News recommendation | `/recommendation/` |

From the repository root:

```bash
uv run --no-project --with-requirements docs/requirements.txt python docs/build.py
python -m http.server 8000 --directory ../tmp/mt-docs-site
```

Open `http://localhost:8000/`. Each module has its own search index. README and SETUP supply the guides; API pages are generated from source without importing the training packages or loading model assets.

The build uses strict link checks. `--output PATH` selects another output directory. Keep that directory dedicated to documentation: MkDocs replaces each module's generated files when rebuilding.

## Cloudflare Pages

Publish the complete output directory to the existing `mt-docs` project. With Wrangler installed and authenticated:

```bash
# Preview branch; use a name different from the project's production branch.
npx wrangler pages deploy ../tmp/mt-docs-site --project-name mt-docs --branch docs-preview

# Production branch of the existing mt-docs project.
npx wrangler pages deploy ../tmp/mt-docs-site --project-name mt-docs --branch main
```

The root page links to all modules. `_redirects` preserves the former Seg links at the site root; `/topic-segmentation/` keeps its existing path. The root `404.html` prevents unknown paths from falling back to the home page.
