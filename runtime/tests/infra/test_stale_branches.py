"""Branch lifecycle exclusions and incomplete API responses fail conservatively."""

import importlib.util
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stale_branches", Path(__file__).resolve().parents[2] / "scripts/ci/stale_branches.py"
)
scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scan)
NOW = datetime(2026, 9, 15, tzinfo=UTC)


def branch(name="feature/old", **kwargs):
    return {
        "name": name,
        "protected": False,
        "created_at": "2026-01-01T00:00:00Z",
        "commit": {"id": "abc123", "committed_date": "2026-01-01T00:00:00Z"},
        **kwargs,
    }


def test_exclusions_and_new_branch_from_old_commit():
    branches = [
        branch(),
        branch("master"),
        branch("release/v1"),
        branch("protected", protected=True),
        branch("unknown", protected=None),
        branch("mr-source"),
        branch("mr-target"),
        branch("recent", created_at="2026-09-14T00:00:00Z"),
        branch("updated", updated_at="2026-09-14T00:00:00Z"),
        branch("missing", created_at=None),
    ]
    mrs = [
        {
            "state": "reopened",
            "source_project_id": scan.PROJECT_ID,
            "target_project_id": scan.PROJECT_ID,
            "source_branch": "mr-source",
            "target_branch": "mr-target",
        }
    ]
    report = scan.inventory(branches, mrs, "master", NOW, 60)
    status = {row["branch"]: row["status"] for row in report["branches"]}
    assert status == {
        "feature/old": "candidate",
        "master": "default_branch",
        "release/v1": "keep_pattern",
        "protected": "protected_or_unknown",
        "unknown": "protected_or_unknown",
        "mr-source": "open_mr",
        "mr-target": "open_mr",
        "recent": "recent",
        "updated": "recent",
        "missing": "incomplete_metadata",
    }


def test_fork_source_does_not_hide_same_name_in_parent():
    mrs = [
        {
            "state": "opened",
            "source_project_id": 123,
            "target_project_id": scan.PROJECT_ID,
            "source_branch": "feature/old",
            "target_branch": "master",
        }
    ]
    assert scan.inventory([branch()], mrs, "master", NOW, 60)["counts"] == {"candidate": 1}


def test_exact_threshold_and_missing_timezone():
    item = branch(created_at="2026-07-17T00:00:00Z")
    assert scan.inventory([item], [], "master", NOW, 60)["counts"] == {"candidate": 1}
    item["created_at"] = "2026-07-17T00:00:01Z"
    assert scan.inventory([item], [], "master", NOW, 60)["counts"] == {"recent": 1}
    item["created_at"] = "2026-01-01T00:00:00"
    assert scan.inventory([item], [], "master", NOW, 60)["counts"] == {"incomplete_metadata": 1}


def test_pages_and_incomplete_mr_fail_closed():
    api = scan.Gongfeng("test-token")
    api.get = Mock(side_effect=[[branch(str(i)) for i in range(100)], [branch("last")]])
    assert len(api.pages("/repository/branches")) == 101
    api.get = Mock(return_value={"message": "forbidden"})
    with pytest.raises(RuntimeError, match="Expected a list"):
        api.pages("/repository/branches")
    api.get = Mock(return_value=[branch(str(i)) for i in range(100)])
    with pytest.raises(RuntimeError, match="Repeated page"):
        api.pages("/repository/branches")
    with pytest.raises(RuntimeError, match="project metadata"):
        scan.inventory([branch()], [{"state": "opened"}], "master", NOW, 60)
    with pytest.raises(RuntimeError, match="default branch"):
        scan.inventory([branch()], [], None, NOW, 60)


def test_command_writes_report_using_only_get_requests(monkeypatch, tmp_path, capsys):
    requests = []

    def respond(request, timeout):
        requests.append(request)
        assert request.get_method() == "GET"
        assert request.get_header("Private-token") == "test-token"
        assert "test-token" not in request.full_url
        if request.full_url == scan.API_ROOT:
            payload = {"id": scan.PROJECT_ID, "default_branch": "master"}
        elif "/repository/branches?" in request.full_url:
            payload = [branch()]
        else:
            assert "/merge_requests?" in request.full_url
            payload = []
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(scan, "build_opener", lambda *_: Mock(open=respond))
    monkeypatch.setenv("GONGFENG_TOKEN", "test-token")
    monkeypatch.setattr("sys.argv", ["stale_branches.py", "--output-dir", str(tmp_path)])
    assert scan.main() == 0
    assert len(requests) == 4
    assert json.loads((tmp_path / "report.json").read_text())["mode"] == "preview_only"
    assert "preview only" in (tmp_path / "report.md").read_text()
    assert "test-token" not in capsys.readouterr().out
