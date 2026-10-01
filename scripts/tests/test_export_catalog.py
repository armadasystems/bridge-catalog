import subprocess

import pytest

import export_catalog as ec

MODEL_TEXT = "name: Model A\nprovider: huggingface\nmodelProvider: Qwen\nmodelId: org/model-a\n"


def test_first_and_last_dates(repo):
    repo.write("models/v1/a.yaml", MODEL_TEXT)
    repo.commit("add a", "2025-07-29")
    repo.write("models/v1/a.yaml", MODEL_TEXT + "version: 1.0.0\n")
    repo.commit("edit a", "2026-08-06")

    assert ec.first_added_date(repo.root, "models/v1/a.yaml", follow=True) == "2025-07-29"
    assert ec.last_updated_date(repo.root, "models/v1/a.yaml") == "2026-08-06"


def test_first_added_date_follows_renames(repo):
    repo.write("models/v1/old.yaml", MODEL_TEXT)
    repo.commit("add old", "2025-07-29")
    repo.run("mv", "models/v1/old.yaml", "models/v1/new.yaml")
    repo.commit("rename", "2026-01-09")

    assert ec.first_added_date(repo.root, "models/v1/new.yaml", follow=True) == "2025-07-29"


def test_directory_first_added_uses_earliest_file(repo):
    repo.write("apps/v1/mlflow/app.yaml", "name: mlflow\n")
    repo.commit("add app", "2025-09-02")
    repo.write("apps/v1/mlflow/versions/3.3.2/values.yaml", "a: 1\n")
    repo.commit("add version", "2026-01-01")

    assert ec.first_added_date(repo.root, "apps/v1/mlflow", follow=False) == "2025-09-02"
    assert ec.last_updated_date(repo.root, "apps/v1/mlflow") == "2026-01-01"


def test_uncommitted_path_has_no_dates(repo):
    repo.write("models/v1/a.yaml", MODEL_TEXT)
    repo.commit("add a", "2025-07-29")
    repo.write("models/v1/b.yaml", MODEL_TEXT)

    assert ec.first_added_date(repo.root, "models/v1/b.yaml", follow=True) is None
    assert ec.last_updated_date(repo.root, "models/v1/b.yaml") is None


def test_shallow_clone_is_rejected(repo, tmp_path):
    repo.write("models/v1/a.yaml", MODEL_TEXT)
    repo.commit("add a", "2025-07-29")
    repo.write("models/v1/a.yaml", MODEL_TEXT + "version: 2\n")
    repo.commit("edit a", "2026-08-06")
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{repo.root}", str(shallow)],
        check=True,
    )

    ec.ensure_full_history(repo.root)  # full repo: no error
    with pytest.raises(ec.CatalogError, match="fetch-depth: 0"):
        ec.ensure_full_history(shallow)
