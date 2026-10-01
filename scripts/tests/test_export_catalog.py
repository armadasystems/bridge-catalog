import datetime as dt
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
