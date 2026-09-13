# VERSA website

A static project website for GitHub Pages, with no build dependencies.
Only this `website` directory is uploaded by `.github/workflows/pages.yml`.
The scoring toolkit, caches, and internal development documents are not deployed.

## Preview

From the repository root, run `python3 -m http.server 8000 --directory website`
and open http://localhost:8000.

## Publish

Enable GitHub Pages in the destination repository under **Settings → Pages →
Source → GitHub Actions**. Merge the website and Pages workflow into `main`.
Future changes to `website/` on `main` deploy automatically. The workflow can
also be started manually from the Actions tab.

For `wavlab-speech/versa`, the default URL is https://wavlab-speech.github.io/versa/.
For `ftshijt/versa`, it is https://ftshijt.github.io/versa/.
The URL becomes available after the first successful Pages deployment.

Content is based on the project README and metric documentation. Update
`index.html` for content, `styles.css` for appearance, and `script.js` for the
optional clipboard control. Relative assets support either project URL.
