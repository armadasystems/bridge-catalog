#!/usr/bin/env python3
"""Export bridge-catalog models and apps as a snapshot for the Pulse dashboard."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


class CatalogError(Exception):
    """A catalog file or the checkout is not in a state we can export."""


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        raise CatalogError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return result.stdout.strip()


def ensure_full_history(repo: Path) -> None:
    if _git(repo, "rev-parse", "--is-shallow-repository") == "true":
        raise CatalogError(
            "shallow clone: git dates would be wrong; check out with fetch-depth: 0"
        )


# on_branch=True: only commits on the checked-out branch's first-parent line (merges into
# staging, not feature-branch commits), dated when they landed there (committer date).
# on_branch=False: every commit, dated when it was written (author date).
def _history_args(on_branch: bool) -> list[str]:
    if on_branch:
        return ["--first-parent", "--diff-merges=first-parent", "--format=@%H %cd"]
    return ["--format=@%H %ad"]


def first_added_date(repo: Path, rel_path: str, follow: bool, on_branch: bool = False) -> str | None:
    """Date of the oldest commit that added rel_path (or any file under it)."""
    # --name-only: --diff-merges would otherwise turn on full patch output.
    args = ["log", "--diff-filter=A", *_history_args(on_branch), "--date=short", "--name-only"]
    if follow:
        args.append("--follow")
    log = _git(repo, *args, "--", rel_path)
    dates = [line.split(" ", 1)[1] for line in log.splitlines() if line.startswith("@")]
    return dates[-1] if dates else None


def last_updated_date(repo: Path, rel_path: str) -> str | None:
    return _git(repo, "log", "-1", "--format=%ad", "--date=short", "--", rel_path) or None


STATUSES = ("in-progress", "live", "deprecated")
TRACKING_KEYS = ("status", "liveDate", "note", "source", "partner", "type", "appName")


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iso_date(value, where: str, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        # Blank is a common placeholder for an unknown date, same as _optional_str.
        if not value.strip():
            return None
        try:
            return dt.date.fromisoformat(value.strip()).isoformat()
        except ValueError:
            pass
    raise CatalogError(f"{where}: {field} must be a YYYY-MM-DD date, got {value!r}")


def parse_tracking(doc: dict, where: str) -> dict:
    """Validate the optional `tracking:` block and return it with normalised values."""
    raw = doc.get("tracking")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise CatalogError(f"{where}: tracking must be a mapping")
    unknown = sorted(str(key) for key in raw if key not in TRACKING_KEYS)
    if unknown:
        raise CatalogError(
            f"{where}: unknown tracking keys {unknown}; allowed: {list(TRACKING_KEYS)}"
        )
    status = _optional_str(raw.get("status"))
    if status is not None and status not in STATUSES:
        raise CatalogError(
            f"{where}: tracking.status must be one of {list(STATUSES)}, got {status!r}"
        )
    return {
        "status": status,
        "live_date": _iso_date(raw.get("liveDate"), where, "tracking.liveDate"),
        "note": _optional_str(raw.get("note")),
        "source": _optional_str(raw.get("source")),
        "partner": _optional_str(raw.get("partner")),
        "type": _optional_str(raw.get("type")),
        "app_name": _optional_str(raw.get("appName")),
    }


PROVIDER_SOURCES = {"azureml": "Azure ML", "nim": "Nvidia NIM", "primalabs": "PrimaLabs"}
MODEL_PROVIDER_SOURCES = {
    "Mistral": "Mistral AI",
    "NVIDIA": "NVIDIA (HuggingFace)",
    "Multiverse": "Multiverse Computing",
}
DEFAULT_SOURCE = "HuggingFace (direct)"


def load_yaml(path: Path, where: str) -> dict:
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    # ValueError: PyYAML builds dates eagerly, so `2026-02-30` fails here, not as YAMLError.
    except (yaml.YAMLError, ValueError, UnicodeDecodeError, OSError) as exc:
        raise CatalogError(f"{where}: invalid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise CatalogError(f"{where}: expected a mapping at the top level")
    return doc


def derive_source(doc: dict, tracking: dict) -> str:
    """The Excel "Source" column: delivery channel, or the partner for partner models."""
    if tracking["source"]:
        return tracking["source"]
    provider = (_optional_str(doc.get("provider")) or "").lower()
    if provider in PROVIDER_SOURCES:
        return PROVIDER_SOURCES[provider]
    model_provider = _optional_str(doc.get("modelProvider")) or ""
    return MODEL_PROVIDER_SOURCES.get(model_provider, DEFAULT_SOURCE)


def _rel(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _identity(doc) -> str | None:
    if not isinstance(doc, dict):
        return None
    return _optional_str(doc.get("modelId")) or _optional_str(doc.get("name"))


def _identity_at(repo: Path, rev: str, rel_path: str) -> str | None:
    try:
        return _identity(yaml.safe_load(_git(repo, "show", f"{rev}:{rel_path}")))
    except (CatalogError, yaml.YAMLError, ValueError):
        return None


def model_start_date(
    repo: Path, rel_path: str, identity: str | None, on_branch: bool = False
) -> str | None:
    """First-added date, following renames only while the model stayed the same.

    git's rename detection is a similarity guess: deleting one model and adding a
    near-identical variant in one commit looks like a rename. A rename whose old file
    had a different modelId (or name) counts as the new model being added.
    """
    log = _git(
        repo, "log", "--follow", *_history_args(on_branch), "--date=short", "--name-status",
        "--", rel_path,
    )
    sha = date = start = None
    for line in log.splitlines():
        if line.startswith("@"):
            sha, date = line[1:].split(" ", 1)
            continue
        status, *paths = line.split("\t")
        if status.startswith("R") and _identity_at(repo, f"{sha}^", paths[0]) != identity:
            return date
        if status == "A":
            start = date
    return start


def model_row(repo: Path, path: Path) -> dict:
    rel = _rel(repo, path)
    doc = load_yaml(path, rel)
    name = _optional_str(doc.get("name"))
    if not name:
        raise CatalogError(f"{rel}: missing name")
    tracking = parse_tracking(doc, rel)
    return {
        "id": path.stem,
        "source": derive_source(doc, tracking),
        "type": "Model",
        "name": name,
        "start_date": model_start_date(repo, rel, _identity(doc)),
        "merged_date": model_start_date(repo, rel, _identity(doc), on_branch=True),
        "live_date": tracking["live_date"],
        "status": tracking["status"],
        "status_note": tracking["note"],
        "model_provider": _optional_str(doc.get("modelProvider")),
        "provider": _optional_str(doc.get("provider")),
        "model_id": _optional_str(doc.get("modelId")),
        "file_path": rel,
        "last_updated": last_updated_date(repo, rel),
    }


def app_row(repo: Path, app_dir: Path) -> dict:
    rel_dir = _rel(repo, app_dir)
    rel = f"{rel_dir}/app.yaml"
    app_yaml = app_dir / "app.yaml"
    if not app_yaml.is_file():
        raise CatalogError(f"{rel}: missing")
    doc = load_yaml(app_yaml, rel)
    name = _optional_str(doc.get("displayName")) or _optional_str(doc.get("name"))
    if not name:
        raise CatalogError(f"{rel}: missing displayName")
    tracking = parse_tracking(doc, rel)
    versions_dir = app_dir / "versions"
    versions = (
        sorted(p.name for p in versions_dir.iterdir() if p.is_dir())
        if versions_dir.is_dir()
        else []
    )
    return {
        "id": app_dir.name,
        "partner": tracking["partner"] or name,
        "type": tracking["type"] or "App",
        "name": tracking["app_name"] or name,
        "start_date": first_added_date(repo, rel_dir, follow=False),
        "merged_date": first_added_date(repo, rel_dir, follow=False, on_branch=True),
        "live_date": tracking["live_date"],
        "status": tracking["status"],
        "status_note": tracking["note"],
        "category": _optional_str(doc.get("category")),
        "versions": versions,
        "file_path": rel,
        "last_updated": last_updated_date(repo, rel_dir),
        "is_partner": False,
        # Raw tracking values: partner data may only fill what these leave empty.
        "_tracking": tracking,
    }


MODELS_DIR = "models/v1"
APPS_DIR = "apps/v1"
SCHEMA_VERSION = 2


def _ensure_unique_ids(rows: list[dict]) -> None:
    """`id` is the consumer's key; two files with one id would silently overwrite each other."""
    seen: dict[str, str] = {}
    for row in rows:
        if row["id"] in seen:
            raise CatalogError(
                f"duplicate id {row['id']!r}: {seen[row['id']]} and {row['file_path']}"
            )
        seen[row["id"]] = row["file_path"]


def build_snapshot(repo: Path) -> tuple[list[dict], list[dict]]:
    """Every model and app row, sorted by id. Raises CatalogError on any bad file."""
    ensure_full_history(repo)
    models_dir = repo / MODELS_DIR
    if not models_dir.is_dir():
        raise CatalogError(f"{MODELS_DIR}: directory not found in {repo}")
    # Recursive, .yaml and .yml — same set model-catalog-service loads.
    model_files = sorted(
        p for p in models_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".yaml", ".yml")
    )
    models = sorted((model_row(repo, p) for p in model_files), key=lambda row: row["id"])
    _ensure_unique_ids(models)
    apps_dir = repo / APPS_DIR
    app_dirs = sorted(p for p in apps_dir.iterdir() if p.is_dir()) if apps_dir.is_dir() else []
    apps = sorted((app_row(repo, d) for d in app_dirs), key=lambda row: row["id"])
    return models, apps


def build_manifest(
    repo: Path, models: list[dict], apps: list[dict], env: Mapping[str, str], now: dt.datetime
) -> dict:
    sha = _git(repo, "rev-parse", "HEAD")
    run_id = env.get("GITHUB_RUN_ID")
    if run_id:
        snapshot_id = f"{run_id}-{env.get('GITHUB_RUN_ATTEMPT', '1')}"
    else:
        snapshot_id = f"local-{sha[:12]}"
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "commit_sha": sha,
        "branch": env.get("GITHUB_REF_NAME") or _git(repo, "branch", "--show-current") or None,
        "generated_at": now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_count": len(models),
        "app_count": len(apps),
    }


PARTNERS_FILE = "src/features/partners/partners.tracking.json"
PARTNERS_CONST = "src/features/partners/partners.const.ts"
PARTNER_KEYS = (
    "id", "name", "onDashboard", "appName", "type", "status", "startDate", "liveDate", "note",
)
PARTNER_TYPES = ("App", "API")
_PARTNER_ID = re.compile(r"[a-z0-9-]+")
_CONST_ID = re.compile(r"\bid:\s*'([a-z0-9-]+)'")


def load_partners(ui_repo: Path) -> list[dict]:
    """Validated entries of bridge-ui's partners.tracking.json, in file order."""
    where = f"bridge-ui:{PARTNERS_FILE}"
    try:
        data = json.loads((ui_repo / PARTNERS_FILE).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise CatalogError(f"{where}: invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise CatalogError(f"{where}: expected a JSON array")
    partners, seen = [], set()
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise CatalogError(f"{where}: entry {index} must be an object")
        pid = _optional_str(raw.get("id"))
        at = f"{where} ({pid or f'entry {index}'})"
        unknown = sorted(str(key) for key in raw if key not in PARTNER_KEYS)
        if unknown:
            raise CatalogError(f"{at}: unknown keys {unknown}; allowed: {list(PARTNER_KEYS)}")
        for key in ("id", "name"):
            if not _optional_str(raw.get(key)):
                raise CatalogError(f"{at}: missing {key}")
        # Ids go into a git -G (POSIX) regex unescaped, so keep them to this safe set.
        if not _PARTNER_ID.fullmatch(pid):
            raise CatalogError(f"{at}: id must use only lowercase letters, digits and '-'")
        if not isinstance(raw.get("onDashboard"), bool):
            raise CatalogError(f"{at}: onDashboard must be true or false")
        if pid in seen:
            raise CatalogError(f"{where}: duplicate id {pid!r}")
        seen.add(pid)
        status = _optional_str(raw.get("status"))
        if status is not None and status not in STATUSES:
            raise CatalogError(f"{at}: status must be one of {list(STATUSES)}, got {status!r}")
        kind = _optional_str(raw.get("type"))
        if kind is not None and kind not in PARTNER_TYPES:
            raise CatalogError(f"{at}: type must be one of {list(PARTNER_TYPES)}, got {kind!r}")
        partners.append({
            "id": pid,
            "name": _optional_str(raw.get("name")),
            "on_dashboard": raw["onDashboard"],
            "app_name": _optional_str(raw.get("appName")),
            "type": kind,
            "status": status,
            "start_date": _iso_date(raw.get("startDate"), at, "startDate"),
            "live_date": _iso_date(raw.get("liveDate"), at, "liveDate"),
            "note": _optional_str(raw.get("note")),
        })
    return partners


def const_partner_ids(ui_repo: Path) -> list[str] | None:
    """Partner ids on the Partners page, or None when the const file is missing."""
    path = ui_repo / PARTNERS_CONST
    if not path.is_file():
        return None
    return _CONST_ID.findall(path.read_text(encoding="utf-8"))


def drift_warnings(partners: list[dict], const_ids: list[str]) -> list[str]:
    json_ids = [p["id"] for p in partners]
    missing = [
        f"partner {pid!r} is in {PARTNERS_CONST} but not in {PARTNERS_FILE}; "
        "it won't reach the dashboard"
        for pid in const_ids if pid not in json_ids
    ]
    stale = [
        f"partner {pid!r} is in {PARTNERS_FILE} but no longer in {PARTNERS_CONST}"
        for pid in json_ids if pid not in const_ids
    ]
    return missing + stale


def _entry_added_date(
    ui_repo: Path, rel_path: str, pickaxe: str, on_branch: bool
) -> str | None:
    log = _git(
        ui_repo, "log", pickaxe, *_history_args(on_branch), "--date=short",
        "--name-only", "--", rel_path,
    )
    dates = [line.split(" ", 1)[1] for line in log.splitlines() if line.startswith("@")]
    return dates[-1] if dates else None


def _earliest_added_date(ui_repo: Path, partner: dict, on_branch: bool) -> str | None:
    # The Partners page entry (partners.const.ts) dates from when the partner was added;
    # partners.tracking.json only appeared later (bridge-ui#1080). Const entries were keyed
    # by name until ids were added, so match the name literally (-S) or the id.
    const_dates = [
        _entry_added_date(ui_repo, PARTNERS_CONST, pickaxe, on_branch)
        for pickaxe in (f"-Sname: '{partner['name']}'", f"-Gid:[[:space:]]*'{partner['id']}'")
    ]
    found = [date for date in const_dates if date]
    if found:
        return min(found)
    # -G (regex on changed lines) rather than -S: survives reformatting of the JSON.
    pattern = f'-G"id"[[:space:]]*:[[:space:]]*"{partner["id"]}"'
    return _entry_added_date(ui_repo, PARTNERS_FILE, pattern, on_branch)


def partner_dates(ui_repo: Path, partner: dict) -> tuple[str | None, str | None]:
    """(first commit of the partner's entry, merge of it into the checked-out branch)."""
    return (
        _earliest_added_date(ui_repo, partner, on_branch=False),
        _earliest_added_date(ui_repo, partner, on_branch=True),
    )


def partner_row(ui_repo: Path, partner: dict) -> dict:
    first_commit, merged = partner_dates(ui_repo, partner)
    return {
        "id": partner["id"],
        "partner": partner["name"],
        "type": partner["type"] or "App",
        "name": partner["app_name"] or partner["name"],
        # Existing partners predate the tracking file, so their real start is given explicitly.
        "start_date": partner["start_date"] or first_commit,
        "merged_date": merged,
        "live_date": partner["live_date"],
        "status": partner["status"],
        "status_note": partner["note"],
        "category": None,
        "versions": None,
        "file_path": f"bridge-ui/{PARTNERS_FILE}",
        "last_updated": None,
        "is_partner": True,
    }


def _with_partner(app: dict, partner: dict) -> dict:
    tracking = app["_tracking"]
    return {
        **app,
        "partner": tracking["partner"] or partner["name"],
        "name": tracking["app_name"] or partner["app_name"] or app["name"],
        "type": tracking["type"] or partner["type"] or "App",
        "status": tracking["status"] or partner["status"],
        "live_date": tracking["live_date"] or partner["live_date"],
        "status_note": tracking["note"] or partner["note"],
        "is_partner": True,
    }


def merge_partner_apps(apps: list[dict], partners: list[dict], ui_repo: Path) -> list[dict]:
    """Catalog apps (always kept) plus dashboard partners; one row per id."""
    by_id = {p["id"]: p for p in partners}
    rows = [_with_partner(app, by_id[app["id"]]) if app["id"] in by_id else app for app in apps]
    app_ids = {app["id"] for app in apps}
    rows += [
        partner_row(ui_repo, p) for p in partners if p["on_dashboard"] and p["id"] not in app_ids
    ]
    return sorted(rows, key=lambda row: row["id"])


def add_partner_apps(ui_repo: Path, apps: list[dict]) -> tuple[list[dict], list[str]]:
    """Apps merged with bridge-ui partners, plus warnings for the CI log."""
    ensure_full_history(ui_repo)
    if not (ui_repo / PARTNERS_FILE).is_file():
        return apps, [f"{PARTNERS_FILE} not found in {ui_repo}; partner apps are not included"]
    partners = load_partners(ui_repo)
    const_ids = const_partner_ids(ui_repo)
    if const_ids is None:
        warnings = [f"{PARTNERS_CONST} not found in {ui_repo}; drift not checked"]
    else:
        warnings = drift_warnings(partners, const_ids)
    return merge_partner_apps(apps, partners, ui_repo), warnings


SNAPSHOT_FILE = "bridge_catalog_snapshot.parquet"

# One row per model/app. Column order and types are the contract with the data team.
SNAPSHOT_SCHEMA = pa.schema([
    ("entity_type", pa.string()), ("id", pa.string()), ("name", pa.string()),
    ("type", pa.string()), ("source", pa.string()), ("partner", pa.string()),
    ("is_partner", pa.bool_()),
    ("start_date", pa.date32()), ("merged_date", pa.date32()), ("live_date", pa.date32()),
    ("status", pa.string()), ("status_note", pa.string()),
    ("model_provider", pa.string()), ("provider", pa.string()), ("model_id", pa.string()),
    ("category", pa.string()), ("versions", pa.list_(pa.string())),
    ("file_path", pa.string()), ("last_updated", pa.date32()),
    ("snapshot_id", pa.string()), ("commit_sha", pa.string()), ("branch", pa.string()),
    ("generated_at", pa.timestamp("ms", tz="UTC")), ("schema_version", pa.int32()),
])
DATE_COLUMNS = ("start_date", "merged_date", "live_date", "last_updated")
SNAPSHOT_COLUMNS = ("snapshot_id", "commit_sha", "branch", "schema_version")


def build_table(models: list[dict], apps: list[dict], manifest: dict) -> pa.Table:
    """Models then apps, each row carrying the snapshot info; columns a row lacks are null."""
    snapshot = {key: manifest[key] for key in SNAPSHOT_COLUMNS}
    snapshot["generated_at"] = dt.datetime.fromisoformat(manifest["generated_at"])
    rows = []
    for entity_type, items in (("Model", models), ("App", apps)):
        for item in items:
            row = {name: item.get(name) for name in SNAPSHOT_SCHEMA.names}
            row.update(snapshot, entity_type=entity_type)
            for column in DATE_COLUMNS:
                if row[column] is not None:
                    row[column] = dt.date.fromisoformat(row[column])
            rows.append(row)
    return pa.Table.from_pylist(rows, schema=SNAPSHOT_SCHEMA)


def write_snapshot(out_dir: Path, models: list[dict], apps: list[dict], manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(build_table(models, apps, manifest), out_dir / SNAPSHOT_FILE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the catalog snapshot for Pulse.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="bridge-catalog checkout")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    parser.add_argument(
        "--partners-repo", type=Path, default=None, help="bridge-ui checkout with partner data"
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        # Build everything before writing so a bad file never leaves partial output.
        models, apps = build_snapshot(repo)
        if args.partners_repo:
            apps, warnings = add_partner_apps(args.partners_repo.resolve(), apps)
        else:
            warnings = ["--partners-repo not given; partner apps are not included"]
        manifest = build_manifest(repo, models, apps, os.environ, dt.datetime.now(dt.timezone.utc))
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"::warning::{warning}")
    write_snapshot(args.out, models, apps, manifest)
    print(f"wrote {len(models)} models and {len(apps)} apps to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
