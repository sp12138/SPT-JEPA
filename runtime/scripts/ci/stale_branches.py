"""Read-only Gongfeng branch inventory; never sends notices or deletes refs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

PROJECT_ID = 1828042
API_ROOT = f"https://git.woa.com/api/v3/projects/{PROJECT_ID}"
KEEP_PATTERNS = ("master", "main", "develop", "release", "release/*", "production", "production/*")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the job token to a redirect destination.
        return None


class Gongfeng:
    def __init__(self, token):
        self.token = token
        self.opener = build_opener(NoRedirect)

    def get(self, path="", **params):
        url = API_ROOT + path
        if params:
            url += "?" + urlencode(params)
        request = Request(url, headers={"PRIVATE-TOKEN": self.token, "Accept": "application/json"})
        try:
            with self.opener.open(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Gongfeng GET {path or '/'} failed: HTTP {exc.code}") from None
        except (URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Gongfeng GET {path or '/'} failed: {type(exc).__name__}") from None

    def pages(self, path, **params):
        result = []
        previous = None
        for page in range(1, 501):
            batch = self.get(path, page=page, per_page=100, **params)
            if not isinstance(batch, list):
                raise RuntimeError(f"Expected a list from {path}; refusing an incomplete report")
            if batch and batch == previous:
                raise RuntimeError(f"Repeated page from {path}; refusing an incomplete report")
            result.extend(batch)
            if len(batch) < 100:
                return result
            previous = batch
        raise RuntimeError(f"Too many pages from {path}; refusing an incomplete report")


def timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else None


def mr_branches(merge_requests):
    result = set()
    for mr in merge_requests:
        if mr.get("state") not in {"opened", "reopened"}:
            continue
        for side in ("source", "target"):
            if str(mr.get(f"{side}_project_id")) == str(PROJECT_ID):
                branch = mr.get(f"{side}_branch")
                if not isinstance(branch, str) or not branch:
                    raise RuntimeError("Open MR has incomplete branch metadata")
                result.add(branch)
            elif mr.get(f"{side}_project_id") is None:
                raise RuntimeError("Open MR has incomplete project metadata")
    return result


def inventory(branches, merge_requests, default_branch, now, days, keep_patterns=KEEP_PATTERNS):
    if not default_branch:
        raise RuntimeError("Missing default branch; refusing an incomplete report")
    active_mrs = mr_branches(merge_requests)
    rows = []
    for branch in branches:
        name = branch.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Missing branch name")
        commit = branch.get("commit") or {}
        row = {
            "branch": name,
            "sha": commit.get("id"),
            "creator": (branch.get("author") or {}).get("username", "unknown"),
            "status": "recent",
            "last_known_activity": None,
        }
        if name == default_branch:
            row["status"] = "default_branch"
        elif branch.get("protected") is not False:
            row["status"] = "protected_or_unknown"
        elif any(fnmatchcase(name, pattern) for pattern in keep_patterns):
            row["status"] = "keep_pattern"
        elif name in active_mrs:
            row["status"] = "open_mr"
        else:
            # An old commit on a newly created branch is not an old branch.
            created = timestamp(branch.get("created_at"))
            committed = timestamp(commit.get("committed_date"))
            updated = timestamp(branch.get("updated_at"))
            if (
                not created
                or not committed
                or not row["sha"]
                or (branch.get("updated_at") is not None and updated is None)
            ):
                row["status"] = "incomplete_metadata"
            else:
                activity = max(date for date in (created, committed, updated) if date)
                row["last_known_activity"] = activity.isoformat()
                if now - activity >= timedelta(days=days):
                    row["status"] = "candidate"
        rows.append(row)
    return {
        "mode": "preview_only",
        "project_id": PROJECT_ID,
        "generated_at": now.isoformat(),
        "threshold_days": days,
        "counts": dict(Counter(row["status"] for row in rows)),
        "branches": sorted(rows, key=lambda row: row["branch"]),
    }


def markdown(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = [
        "# vforge stale branch preview",
        "",
        f"Threshold: {report['threshold_days']} days; mode: preview only.",
        "No branches are deleted and no notifications are sent.",
        "Age uses the newest available branch creation/update or head commit date.",
        "These dates do not guarantee the last push time. Recheck before any future cleanup.",
        "",
        "Counts: " + json.dumps(report["counts"], sort_keys=True),
        "",
        "| Branch | Creator | Last known activity (UTC) | HEAD |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["branches"]:
        if row["status"] == "candidate":
            lines.append(
                "| "
                + " | ".join(
                    cell(row[key]) for key in ("branch", "creator", "last_known_activity", "sha")
                )
                + " |"
            )
    if not report["counts"].get("candidate"):
        lines.append("| No candidate branches | | | |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path(".ci-output/stale-branches"))
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    token = os.environ.get("GONGFENG_TOKEN")
    if not token:
        parser.error("GONGFENG_TOKEN is required (Stream: ci.token)")
    try:
        api = Gongfeng(token)
        project = api.get()
        if project.get("id") != PROJECT_ID:
            raise RuntimeError("Unexpected project returned by Gongfeng")
        branches = api.pages("/repository/branches")
        merge_requests = [
            mr
            for state in ("opened", "reopened")
            for mr in api.pages("/merge_requests", state=state)
        ]
        report = inventory(
            branches,
            merge_requests,
            project.get("default_branch"),
            datetime.now(UTC),
            args.days,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rendered = markdown(report)
        (args.output_dir / "report.md").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"Branch scan failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
