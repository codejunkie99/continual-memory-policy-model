"""Load the console's static assets (HTML/CSS/JS) from the package.

Assets are read once at import time from the ``assets/`` package directory and
served from memory. No request-time filesystem reads happen, so there is no
path-traversal surface in the static serving path.
"""

from __future__ import annotations

from importlib import resources


def _read(name: str) -> str:
    return resources.files("mpm.console").joinpath("assets", name).read_text(encoding="utf-8")


INDEX_HTML = _read("index.html")
STYLES_CSS = _read("console.css")
CLIENT_JS = _read("console.js")
