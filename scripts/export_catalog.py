#!/usr/bin/env python3
"""Export bridge-catalog models and apps as a snapshot for the Pulse dashboard."""
from __future__ import annotations

import subprocess
from pathlib import Path


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
