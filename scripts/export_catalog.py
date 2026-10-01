#!/usr/bin/env python3
"""Export bridge-catalog models and apps as a snapshot for the Pulse dashboard."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

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


def first_added_date(repo: Path, rel_path: str, follow: bool) -> str | None:
    """Date of the oldest commit that added rel_path (or any file under it)."""
    args = ["log", "--diff-filter=A", "--format=%ad", "--date=short"]
    if follow:
        args.append("--follow")
    lines = _git(repo, *args, "--", rel_path).splitlines()
    return lines[-1] if lines else None


def last_updated_date(repo: Path, rel_path: str) -> str | None:
    return _git(repo, "log", "-1", "--format=%ad", "--date=short", "--", rel_path) or None


STATUSES = ("in-progress", "live", "deprecated")
TRACKING_KEYS = ("status", "endDate", "note", "source", "partner", "type")


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iso_date(value, where: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip()).isoformat()
        except ValueError:
            pass
    raise CatalogError(f"{where}: tracking.endDate must be a YYYY-MM-DD date, got {value!r}")


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
        "end_date": _iso_date(raw.get("endDate"), where),
        "note": _optional_str(raw.get("note")),
        "source": _optional_str(raw.get("source")),
        "partner": _optional_str(raw.get("partner")),
        "type": _optional_str(raw.get("type")),
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
    except yaml.YAMLError as exc:
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
        "start_date": first_added_date(repo, rel, follow=True),
        "end_date": tracking["end_date"],
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
        "name": name,
        "start_date": first_added_date(repo, rel_dir, follow=False),
        "end_date": tracking["end_date"],
        "status": tracking["status"],
        "status_note": tracking["note"],
        "category": _optional_str(doc.get("category")),
        "versions": versions,
        "file_path": rel,
        "last_updated": last_updated_date(repo, rel_dir),
    }


MODELS_DIR = "models/v1"
APPS_DIR = "apps/v1"
SCHEMA_VERSION = 1


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_snapshot(out_dir: Path, models: list[dict], apps: list[dict], manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "models.jsonl", models)
    _write_jsonl(out_dir / "apps.jsonl", apps)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the catalog snapshot for Pulse.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="bridge-catalog checkout")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        # Build everything before writing so a bad file never leaves partial output.
        models, apps = build_snapshot(repo)
        manifest = build_manifest(repo, models, apps, os.environ, dt.datetime.now(dt.timezone.utc))
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_snapshot(args.out, models, apps, manifest)
    print(f"wrote {len(models)} models and {len(apps)} apps to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
