#!/usr/bin/env python3
"""Export bridge-catalog models and apps as a snapshot for the Pulse dashboard."""
from __future__ import annotations

import datetime as dt
import subprocess
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
