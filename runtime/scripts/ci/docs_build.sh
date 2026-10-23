#!/usr/bin/env bash
# Build the site using the locked image environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/ci/docs_check.py
python - <<'PY'
import html
import json
import logging
import os
from pathlib import Path
from urllib.parse import urljoin

from mkdocs.commands.build import build
from mkdocs.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
base_path = os.environ.get("DOCS_BASE_PATH", "/")
if base_path not in {"/", "/debug/"}:
    raise ValueError("DOCS_BASE_PATH must be / or /debug/")
config = load_config(config_file="mkdocs-next.yml", strict=True)
config.site_url = urljoin(config.site_url, base_path)
config.plugins.on_startup(command="build", dirty=False)
try:
    build(config)
finally:
    config.plugins.on_shutdown()
site = Path(config.site_dir)
pages = [page.relative_to(site) for page in site.rglob("*.html") if page.name != "404.html"]


def page_url(page):
    return page.as_posix().removesuffix("index.html")


def redirect(output, target):
    output.parent.mkdir(parents=True, exist_ok=True)
    escaped = html.escape(target, quote=True)
    output.write_text(
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">'
        '<title>页面已迁移</title>'
        f'<link rel="canonical" href="{escaped}">'
        f'<script>location.replace({json.dumps(target)}'
        ' + location.search + location.hash);</script>'
        f'<meta http-equiv="refresh" content="0; url={escaped}"></head>'
        f'<body><a href="{escaped}">前往文档</a></body></html>\n',
        encoding="utf-8",
    )


# Preserve existing /next/ links, including query strings and anchors.
for page in pages:
    redirect(site / "next" / page, base_path + page_url(page))
PY
test -f site/index.html
test -f site/next/index.html
