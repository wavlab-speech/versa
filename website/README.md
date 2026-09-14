# VERSA website

A static project website and documentation for GitHub Pages. The build uses
the pinned Python Markdown package in `requirements.txt`; visitors need no
JavaScript to read the documentation. Only generated `website/dist` is uploaded
by `.github/workflows/pages.yml`.
The scoring toolkit, caches, and internal development documents are not deployed.

## Preview

From the repository root:

```sh
python3 -m venv /tmp/versa-website-env
/tmp/versa-website-env/bin/pip install -r website/requirements.txt
/tmp/versa-website-env/bin/python website/build.py
python3 -m http.server 8000 --directory website/dist
```

Open http://localhost:8000. Documentation lives under `/docs/`, with individual
guides at `/docs/installation/`, `/docs/metrics/`, and so on.

## Publish

Enable GitHub Pages in the destination repository under **Settings → Pages →
Source → GitHub Actions**. Merge the website and Pages workflow into `main`.
Changes to the website, source docs, README, license, or visualization images
on `main` rebuild and deploy automatically. The workflow can also be started
manually from the Actions tab.

For `wavlab-speech/versa`, the default URL is https://wavlab-speech.github.io/versa/.
For `ftshijt/versa`, it is https://ftshijt.github.io/versa/.
The URL becomes available after the first successful Pages deployment.

Update `index.html` for the home page, `styles.css` for its appearance, and
`script.js` for the optional clipboard control. Documentation uses the shared
`templates/docs.html` layout and `docs.css` stylesheet. `build.py` renders an
explicit list of published Markdown files and selected README sections; add
guides to its `PAGES` list. It rewrites hosted documentation links and copies
the visualization images locally. External papers and source repositories
remain external. Unlisted planning documents are never included.

Generated output is ignored by Git. Edit the original Markdown documentation
to keep the website and repository synchronized. Relative assets support
either project URL.

## Project statistics

The build reads canonical metric names through VERSA's source-based discovery
registry and fetches stars and forks from the official GitHub repository API.
No model backends are imported. Counts, category totals, the source revision,
and collection timestamp are published in `dist/stats.json` and the statistics
guide. Aliases are excluded; named prompt metrics count separately.

GitHub Actions refreshes the site daily at 08:23 UTC and on changes to `versa/`
or the existing documentation and website sources. Scheduled runs may be delayed.
The build uses its read-only `GITHUB_TOKEN` for API requests; no credential is
included in the published files. Local builds need network access and may use
an optional `GITHUB_TOKEN` to avoid unauthenticated API rate limits.

Unavailable or invalid GitHub data fails the build before replacing the output,
leaving the last successful deployment and its dated statistics online.
