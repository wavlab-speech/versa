"""Build an allowlisted set of VERSA guides as portable GitHub Pages HTML."""

import html
import json
import posixpath
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
from site_stats import collect_statistics, statistics_markdown

SITE = Path(__file__).resolve().parent
REPO = SITE.parent
OUT = SITE / "dist"
GITHUB = "https://github.com/wavlab-speech/versa"
# Deliberately exclude development plans and other non-reader documentation.
PAGES = [
    (
        "",
        "Overview",
        "Find the right guide for your next evaluation.",
        "website/content/overview.md",
        None,
    ),
    (
        "installation",
        "Installation",
        "Set up VERSA and the metric backends you need.",
        "README.md",
        "Installation",
    ),
    (
        "usage",
        "Usage guide",
        "Run evaluations locally or across a compute cluster.",
        "README.md",
        "Usage Examples",
    ),
    (
        "metrics",
        "Metric catalog",
        "Configuration keys, outputs, implementations, and references.",
        "docs/supported_metrics.md",
        None,
    ),
    (
        "visualization",
        "Visualization",
        "Turn evaluation results into reports and charts.",
        "docs/visualization.md",
        None,
    ),
    (
        "community",
        "Community",
        "Explore projects and research using VERSA.",
        "docs/users.md",
        None,
    ),
    (
        "citation",
        "Research & citation",
        "Cite the research behind VERSA.",
        "README.md",
        "Citation",
    ),
    (
        "contributing",
        "Contributing",
        "Add metrics and improve the toolkit.",
        "docs/contributing.md",
        None,
    ),
    (
        "testing",
        "Testing & CI",
        "Run the project's development checks.",
        "docs/ci.md",
        None,
    ),
    (
        "docstring-coverage",
        "Docstring coverage",
        "Coverage requirements and the documented audit.",
        "docs/docstring_coverage.md",
        None,
    ),
    (
        "statistics",
        "Project statistics",
        "Current metric collection and GitHub activity, with sources and counting methodology.",
        "website/content/statistics.md",
        None,
    ),
    ("license", "License", "Terms for using and distributing VERSA.", "LICENSE", None),
]
DOC_ROUTES = {source: slug for slug, _, _, source, section in PAGES if section is None}
README_ROUTES = {
    "installation": "installation",
    "usage-examples": "usage",
    "citation": "citation",
    "supported-metrics": "metrics",
    "contributing": "contributing",
    "license": "license",
    "quick-testing": "testing",
}


def route(slug):
    """Return the generated document path relative to the website root."""
    return "docs/" + (slug + "/" if slug else "")


def local_url(target, current):
    """Resolve a site-root path relative to the current HTML directory."""
    relative = posixpath.relpath(target.rstrip("/") or ".", current)
    return relative + "/" if target.endswith("/") else relative


def rewrite_url(url, source, current):
    """Keep hosted docs and images on-site while retaining external references."""
    parsed = urlsplit(html.unescape(url))
    path = unquote(parsed.path)
    fragment = parsed.fragment
    if parsed.netloc == "github.com" and path.startswith("/wavlab-speech/versa"):
        path = re.sub(r"^/wavlab-speech/versa/(?:blob|tree)/main/?", "", path)
        if path in ("/wavlab-speech/versa", "/wavlab-speech/versa/"):
            if not fragment:
                return url
            path = "README.md"
        elif path.startswith("/wavlab-speech/versa"):
            return url
    elif parsed.scheme or parsed.netloc or not path:
        return url
    elif source.startswith("website/content/"):
        return url
    else:
        path = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    if path == "docs":
        return local_url("docs/", current)
    if path == "README.md":
        slug = README_ROUTES.get(fragment.lstrip("-"), "")
        return local_url(route(slug), current)
    if path in DOC_ROUTES:
        return local_url(route(DOC_ROUTES[path]), current) + (
            "#" + fragment if fragment else ""
        )
    if path.startswith("scripts/visualization/") and path.endswith(".png"):
        return local_url("assets/" + posixpath.basename(path), current)
    if parsed.netloc:
        return url
    return GITHUB + "/blob/main/" + path + ("#" + fragment if fragment else "")


class HostedLinks(HTMLParser):
    """Rewrite rendered links without touching commands or other prose."""

    def __init__(self, source, current):
        """Retain the source path used to resolve relative Markdown links."""
        super().__init__(convert_charrefs=False)
        self.source = source
        self.current = current
        self.parts = []

    def handle_starttag(self, tag, attrs):
        """Rewrite href and src attributes and make wide tables scrollable."""
        if tag == "table":
            self.parts.append(
                '<div class="table-scroll" tabindex="0" role="region" aria-label="Documentation table; scroll horizontally for more columns">'
            )
        rendered = []
        for key, value in attrs:
            if key in ("href", "src") and value:
                value = rewrite_url(value, self.source, self.current)
            rendered.append(
                key if value is None else f'{key}="{html.escape(value, quote=True)}"'
            )
        self.parts.append(
            "<" + tag + (" " + " ".join(rendered) if rendered else "") + ">"
        )

    def handle_endtag(self, tag):
        """Close an element and its optional scrolling container."""
        self.parts.append("</" + tag + ">")
        if tag == "table":
            self.parts.append("</div>")

    def handle_data(self, data):
        """Preserve rendered text exactly."""
        self.parts.append(data)

    def handle_entityref(self, name):
        """Preserve named HTML entities."""
        self.parts.append("&" + name + ";")

    def handle_charref(self, name):
        """Preserve numeric HTML entities."""
        self.parts.append("&#" + name + ";")


def content_for(source, section):
    """Read public source text and optionally extract one README section."""
    text = (REPO / source).read_text()
    if source == "LICENSE":
        return "```text\n" + text + "\n```"
    if section:
        chunks = re.split(r"(?m)^## ", text)
        text = next(
            chunk.split("\n", 1)[1]
            for chunk in chunks[1:]
            if chunk.split("\n", 1)[0].endswith(section)
        )
        if section == "Citation":
            text = (
                "[Read the VERSA paper](https://arxiv.org/abs/2412.17667).\n\n" + text
            )
    elif source != "website/content/overview.md":
        # The layout supplies the page title; retain every actual guide section.
        text = re.sub(r"\A#{1,2} [^\n]+\n", "", text, count=1)
    if source == "docs/supported_metrics.md":
        text = re.sub(r"(?m)^### ", "## ", text)
    return text


def build():
    """Create only public assets and allowlisted documentation in dist."""
    stats = collect_statistics()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for filename in ("index.html", "styles.css", "script.js", "docs.css"):
        shutil.copy2(SITE / filename, OUT / filename)
    homepage = (OUT / "index.html").read_text()
    values = {
        "metric_count": str(stats["metric_count"]),
        "stars": f'{stats["stars"]:,}',
        "forks": f'{stats["forks"]:,}',
        "category_count": str(len(stats["categories"])),
        "updated_at": stats["updated_at"],
        "updated_label": stats["updated_at"].replace("T", " ").replace("Z", " UTC"),
    }
    for key, value in values.items():
        homepage = homepage.replace("{{" + key + "}}", html.escape(value))
    (OUT / "index.html").write_text(homepage)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    shutil.copytree(SITE / "assets", OUT / "assets")
    for filename in ("radar_chart.png", "sample_sunburstchart.png"):
        shutil.copy2(
            REPO / "scripts/visualization" / filename, OUT / "assets" / filename
        )
    template = (SITE / "templates/docs.html").read_text()
    for index, (slug, title, description, source, section) in enumerate(PAGES):
        current = route(slug).rstrip("/")
        root = "../" if not slug else "../../"
        renderer = markdown.Markdown(
            extensions=["tables", "fenced_code", "toc"],
            extension_configs={"toc": {"permalink": True, "toc_depth": "2-3"}},
        )
        content = (
            statistics_markdown(stats)
            if slug == "statistics"
            else content_for(source, section)
        )
        rendered = renderer.convert(content)
        links = HostedLinks(source, current)
        links.feed(rendered)
        navigation = "".join(
            f'<a href="{local_url(route(other), current)}"'
            + (' aria-current="page"' if other == slug else "")
            + f">{html.escape(label)}</a>"
            for other, label, *_ in PAGES
        )
        pagination = []
        for offset, label in ((-1, "Previous"), (1, "Next")):
            adjacent = index + offset
            if 0 <= adjacent < len(PAGES):
                other, name, *_ = PAGES[adjacent]
                pagination.append(
                    f'<a href="{local_url(route(other), current)}"><span>{label}</span>{html.escape(name)}</a>'
                )
        values = {
            "title": html.escape(title),
            "description": html.escape(description),
            "root": root,
            "navigation": navigation,
            "toc": renderer.toc,
            "content": "".join(links.parts),
            "pagination": "".join(pagination),
        }
        page = re.sub(r"\{\{(\w+)\}\}", lambda match: values[match[1]], template)
        target = OUT / current / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page)
    print(f"Built home page and {len(PAGES)} documentation pages in {OUT}")


if __name__ == "__main__":
    build()
