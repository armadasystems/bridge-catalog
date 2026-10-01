import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class GitRepo:
    def __init__(self, root: Path):
        self.root = root

    def run(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def commit(self, message: str, date: str) -> None:
        stamp = f"{date}T12:00:00+00:00"
        env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", message], cwd=self.root, check=True, env=env
        )


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    root = tmp_path / "catalog"
    root.mkdir()
    git = GitRepo(root)
    git.run("init", "-q", "-b", "staging")
    git.run("config", "user.email", "test@example.com")
    git.run("config", "user.name", "Test")
    git.run("config", "commit.gpgsign", "false")
    return git
