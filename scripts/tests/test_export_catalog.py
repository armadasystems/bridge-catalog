import datetime as dt
import json
import subprocess

import pytest
import yaml

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


EMPTY_TRACKING = {
    "status": None, "end_date": None, "note": None,
    "source": None, "partner": None, "type": None,
}


def test_tracking_missing_or_null_gives_defaults():
    assert ec.parse_tracking({"name": "x"}, "f.yaml") == EMPTY_TRACKING
    assert ec.parse_tracking(yaml.safe_load("name: x\ntracking:\n"), "f.yaml") == EMPTY_TRACKING


def test_tracking_full_block():
    doc = yaml.safe_load(
        "tracking:\n"
        "  status: in-progress\n"
        "  endDate: 2026-09-10\n"
        "  note: ' needs ray-llm '\n"
        "  source: Cohere\n"
        "  partner: SecurIn\n"
        "  type: API\n"
    )
    assert ec.parse_tracking(doc, "f.yaml") == {
        "status": "in-progress", "end_date": "2026-09-10", "note": "needs ray-llm",
        "source": "Cohere", "partner": "SecurIn", "type": "API",
    }


@pytest.mark.parametrize("raw", ["endDate: 2026-09-10", "endDate: '2026-09-10'"])
def test_end_date_accepts_yaml_date_and_string(raw):
    doc = yaml.safe_load(f"tracking:\n  {raw}\n")
    assert ec.parse_tracking(doc, "f.yaml")["end_date"] == "2026-09-10"


def test_end_date_accepts_datetime():
    doc = {"tracking": {"endDate": dt.datetime(2026, 9, 10, 8, 30)}}
    assert ec.parse_tracking(doc, "f.yaml")["end_date"] == "2026-09-10"


@pytest.mark.parametrize(
    "block, message",
    [
        ("tracking:\n  status: done\n", "tracking.status"),
        ("tracking:\n  endDate: 10/09/2026\n", "tracking.endDate"),
        ("tracking:\n  satus: live\n", "unknown tracking keys"),
        ("tracking: live\n", "must be a mapping"),
    ],
)
def test_tracking_rejects_bad_values(block, message):
    with pytest.raises(ec.CatalogError, match=message) as info:
        ec.parse_tracking(yaml.safe_load(block), "models/v1/x.yaml")
    assert "models/v1/x.yaml" in str(info.value)


@pytest.mark.parametrize(
    "provider, model_provider, expected",
    [
        ("azureml", "Qwen", "Azure ML"),
        ("nim", "Mistral", "Nvidia NIM"),
        ("primalabs", "OpenAI", "PrimaLabs"),
        ("huggingface", "Mistral", "Mistral AI"),
        ("huggingface", "NVIDIA", "NVIDIA (HuggingFace)"),
        ("huggingface", "Multiverse", "Multiverse Computing"),
        ("huggingface", "Qwen", "HuggingFace (direct)"),
        (None, None, "HuggingFace (direct)"),
    ],
)
def test_derive_source(provider, model_provider, expected):
    doc = {"provider": provider, "modelProvider": model_provider}
    assert ec.derive_source(doc, EMPTY_TRACKING) == expected


def test_tracking_source_overrides_rules():
    tracking = {**EMPTY_TRACKING, "source": "Cohere"}
    assert ec.derive_source({"provider": "nim"}, tracking) == "Cohere"


def test_model_row(repo):
    path = repo.write(
        "models/v1/ministral-3-8b.yaml",
        "name: Ministral 3 8B Instruct (FP8)\n"
        "provider: huggingface\n"
        "modelProvider: Mistral\n"
        "modelId: mistralai/Ministral-3-8B\n"
        "tracking:\n  status: live\n  endDate: 2026-09-10\n",
    )
    repo.commit("add", "2026-08-19")

    assert ec.model_row(repo.root, path) == {
        "id": "ministral-3-8b",
        "source": "Mistral AI",
        "type": "Model",
        "name": "Ministral 3 8B Instruct (FP8)",
        "start_date": "2026-08-19",
        "first_commit_date": "2026-08-19",
        "end_date": "2026-09-10",
        "status": "live",
        "status_note": None,
        "model_provider": "Mistral",
        "provider": "huggingface",
        "model_id": "mistralai/Ministral-3-8B",
        "file_path": "models/v1/ministral-3-8b.yaml",
        "last_updated": "2026-08-19",
    }


def test_model_row_requires_name(repo):
    path = repo.write("models/v1/noname.yaml", "provider: nim\n")
    repo.commit("add", "2026-01-01")
    with pytest.raises(ec.CatalogError, match="models/v1/noname.yaml: missing name"):
        ec.model_row(repo.root, path)


@pytest.mark.parametrize("text", ["name: [unclosed\n", "- just\n- a list\n"])
def test_model_row_rejects_invalid_yaml(repo, text):
    path = repo.write("models/v1/bad.yaml", text)
    repo.commit("add", "2026-01-01")
    with pytest.raises(ec.CatalogError, match="models/v1/bad.yaml"):
        ec.model_row(repo.root, path)


APP_TEXT = 'name: "openwebui"\ndisplayName: "Open WebUI"\ncategory: "AI"\n'


def test_app_row_defaults(repo):
    repo.write("apps/v1/openwebui/app.yaml", APP_TEXT)
    repo.write("apps/v1/openwebui/versions/0.6.18/values.yaml", "a: 1\n")
    repo.commit("add app", "2025-08-06")
    repo.write("apps/v1/openwebui/versions/0.3.8/values.yaml", "a: 1\n")
    repo.commit("add version", "2026-06-30")

    assert ec.app_row(repo.root, repo.root / "apps/v1/openwebui") == {
        "id": "openwebui",
        "partner": "Open WebUI",
        "type": "App",
        "name": "Open WebUI",
        "start_date": "2025-08-06",
        "first_commit_date": "2025-08-06",
        "end_date": None,
        "status": None,
        "status_note": None,
        "category": "AI",
        "versions": ["0.3.8", "0.6.18"],
        "file_path": "apps/v1/openwebui/app.yaml",
        "last_updated": "2026-06-30",
    }


def test_app_row_tracking_overrides(repo):
    repo.write(
        "apps/v1/securin/app.yaml",
        'displayName: "Securin Model Security Scan"\n'
        "tracking:\n  partner: SecurIn\n  type: API\n  status: live\n",
    )
    repo.commit("add", "2026-06-16")
    row = ec.app_row(repo.root, repo.root / "apps/v1/securin")
    assert (row["partner"], row["type"], row["status"], row["versions"]) == (
        "SecurIn", "API", "live", []
    )


def test_app_row_falls_back_to_name(repo):
    repo.write("apps/v1/nginx/app.yaml", 'name: "nginx"\n')
    repo.commit("add", "2025-07-29")
    assert ec.app_row(repo.root, repo.root / "apps/v1/nginx")["name"] == "nginx"


def test_app_row_requires_app_yaml(repo):
    repo.write("apps/v1/empty/versions/1/values.yaml", "a: 1\n")
    repo.commit("add", "2025-07-29")
    with pytest.raises(ec.CatalogError, match="apps/v1/empty/app.yaml: missing"):
        ec.app_row(repo.root, repo.root / "apps/v1/empty")


NOW = dt.datetime(2026, 10, 1, 10, 0, tzinfo=dt.timezone.utc)


def seed_catalog(repo):
    repo.write("models/v1/mistral-7B.yaml", "name: Mistral 7B Instruct\nprovider: huggingface\nmodelProvider: Mistral\n")
    repo.write("models/v1/mistral-7b-instruct-v03-nim.yaml", "name: Mistral 7B Instruct\nprovider: nim\nmodelProvider: Mistral\n")
    repo.write("models/v1/a-qwen.yaml", "name: Qwen2.5 1.5B Instruct – ünïcode\nprovider: huggingface\nmodelProvider: Qwen\n")
    repo.write("apps/v1/mlflow/app.yaml", 'displayName: "MLflow"\ncategory: "MLOps"\n')
    repo.commit("seed", "2025-07-29")


def test_build_snapshot_sorted_by_id(repo):
    seed_catalog(repo)
    models, apps = ec.build_snapshot(repo.root)
    assert [m["id"] for m in models] == ["a-qwen", "mistral-7B", "mistral-7b-instruct-v03-nim"]
    assert [a["id"] for a in apps] == ["mlflow"]


def test_duplicate_names_are_both_exported(repo):
    seed_catalog(repo)
    models, _ = ec.build_snapshot(repo.root)
    mistral = {m["id"]: m["source"] for m in models if m["name"] == "Mistral 7B Instruct"}
    assert mistral == {"mistral-7B": "Mistral AI", "mistral-7b-instruct-v03-nim": "Nvidia NIM"}


def test_build_snapshot_requires_models_dir(repo):
    repo.write("README.md", "x\n")
    repo.commit("init", "2025-01-01")
    with pytest.raises(ec.CatalogError, match="models/v1"):
        ec.build_snapshot(repo.root)


def test_manifest_from_github_env(repo):
    seed_catalog(repo)
    models, apps = ec.build_snapshot(repo.root)
    env = {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_REF_NAME": "staging"}
    manifest = ec.build_manifest(repo.root, models, apps, env, NOW)
    sha = subprocess.run(["git", "-C", str(repo.root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert manifest == {
        "schema_version": 1,
        "snapshot_id": "123-2",
        "commit_sha": sha,
        "branch": "staging",
        "generated_at": "2026-10-01T10:00:00Z",
        "model_count": 3,
        "app_count": 1,
    }


def test_manifest_local_run(repo):
    seed_catalog(repo)
    manifest = ec.build_manifest(repo.root, [], [], {}, NOW)
    assert manifest["snapshot_id"] == f"local-{manifest['commit_sha'][:12]}"
    assert manifest["branch"] == "staging"


def test_main_writes_snapshot(repo, tmp_path, monkeypatch):
    seed_catalog(repo)
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)  # set when CI runs the tests
    out = tmp_path / "out"

    assert ec.main(["--repo", str(repo.root), "--out", str(out)]) == 0

    lines = (out / "models.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "ünïcode" in lines[0]  # written as UTF-8, not \u escapes
    assert json.loads(lines[0])["start_date"] == "2025-07-29"
    assert len((out / "apps.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_id"] == "99-1"
    assert (manifest["model_count"], manifest["app_count"]) == (3, 1)


def test_main_bad_yaml_fails_without_writing(repo, tmp_path, capsys):
    seed_catalog(repo)
    repo.write("models/v1/broken.yaml", "name: [unclosed\n")
    repo.commit("break", "2026-01-01")
    out = tmp_path / "out"
    out.mkdir()
    (out / "models.jsonl").write_text("previous\n", encoding="utf-8")

    assert ec.main(["--repo", str(repo.root), "--out", str(out)]) == 1

    assert "models/v1/broken.yaml" in capsys.readouterr().err
    assert (out / "models.jsonl").read_text(encoding="utf-8") == "previous\n"
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize(
    "content",
    [
        b"name: x\ntracking:\n  endDate: 2026-02-30\n",  # YAML date that cannot exist
        b"name: \xff\xfe broken\n",  # not UTF-8
    ],
)
def test_model_row_unreadable_file_names_the_file(repo, content):
    path = repo.root / "models/v1/odd.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    repo.commit("add", "2026-01-01")
    with pytest.raises(ec.CatalogError, match="models/v1/odd.yaml"):
        ec.model_row(repo.root, path)


@pytest.mark.parametrize("other", ["models/v1/a.yml", "models/v1/sub/a.yaml"])
def test_build_snapshot_rejects_duplicate_ids(repo, other):
    repo.write("models/v1/a.yaml", MODEL_TEXT)
    repo.write(other, MODEL_TEXT)
    repo.commit("add", "2026-01-01")
    with pytest.raises(ec.CatalogError, match="duplicate id 'a'") as info:
        ec.build_snapshot(repo.root)
    assert "models/v1/a.yaml" in str(info.value) and other in str(info.value)


VARIANT_BODY = "".join(f"config{i}: value-{i}\n" for i in range(30))


def test_start_date_ignores_rename_from_a_different_model(repo):
    repo.write("models/v1/llama3-8b.yaml", "name: Llama 3 8B\nmodelId: meta/llama-3-8b\n" + VARIANT_BODY)
    repo.commit("add llama3", "2024-01-01")
    repo.run("rm", "-q", "models/v1/llama3-8b.yaml")
    path = repo.write("models/v1/llama3_1-8b.yaml", "name: Llama 3.1 8B\nmodelId: meta/llama-3.1-8b\n" + VARIANT_BODY)
    repo.commit("replace with llama3.1", "2026-06-01")
    # git's similarity guess calls this a rename...
    assert ec.first_added_date(repo.root, "models/v1/llama3_1-8b.yaml", follow=True) == "2024-01-01"
    # ...but it is a different model, so it starts when it was added.
    assert ec.model_row(repo.root, path)["start_date"] == "2026-06-01"


def test_start_date_keeps_rename_of_the_same_model(repo):
    repo.write("models/v1/qwen2-5B.yaml", "name: Qwen2.5 1.5B\nmodelId: Qwen/Qwen2.5-1.5B\n" + VARIANT_BODY)
    repo.commit("add qwen", "2025-07-29")
    repo.run("mv", "models/v1/qwen2-5B.yaml", "models/v1/qwen2.5-1.5B.yaml")
    repo.commit("fix file name", "2025-11-28")
    path = repo.root / "models/v1/qwen2.5-1.5B.yaml"
    assert ec.model_row(repo.root, path)["start_date"] == "2025-07-29"


def _merge_feature(repo, rel: str, text: str, authored: str, merged: str) -> None:
    """Commit `rel` on a feature branch on `authored`, merge it into staging on `merged`."""
    repo.write("README.md", "base\n")
    repo.commit("base", "2025-01-01")
    repo.run("checkout", "-q", "-b", "feat")
    repo.write(rel, text)
    repo.commit("add on feature branch", authored)
    repo.run("checkout", "-q", "staging")
    repo.merge("feat", merged)


def test_model_start_date_is_merge_into_staging(repo):
    _merge_feature(repo, "models/v1/m.yaml", MODEL_TEXT, "2026-08-19", "2026-09-17")
    row = ec.model_row(repo.root, repo.root / "models/v1/m.yaml")
    assert (row["start_date"], row["first_commit_date"]) == ("2026-09-17", "2026-08-19")


def test_app_start_date_is_merge_into_staging(repo):
    _merge_feature(repo, "apps/v1/x/app.yaml", APP_TEXT, "2026-02-06", "2026-07-08")
    row = ec.app_row(repo.root, repo.root / "apps/v1/x")
    assert (row["start_date"], row["first_commit_date"]) == ("2026-07-08", "2026-02-06")


def test_start_date_survives_rename_after_merge(repo):
    _merge_feature(repo, "models/v1/llama2-7b.yaml", MODEL_TEXT + VARIANT_BODY, "2025-07-20", "2025-07-29")
    repo.run("mv", "models/v1/llama2-7b.yaml", "models/v1/llama3-8b.yaml")
    repo.commit("fix file name", "2026-06-30")
    row = ec.model_row(repo.root, repo.root / "models/v1/llama3-8b.yaml")
    assert (row["start_date"], row["first_commit_date"]) == ("2025-07-29", "2025-07-20")


def test_start_date_uses_landing_date_for_rebased_commits(repo):
    repo.write("models/v1/m.yaml", MODEL_TEXT)
    repo.commit("rebase-merged", "2026-08-19", landed="2026-09-17")
    row = ec.model_row(repo.root, repo.root / "models/v1/m.yaml")
    assert (row["start_date"], row["first_commit_date"]) == ("2026-09-17", "2026-08-19")
