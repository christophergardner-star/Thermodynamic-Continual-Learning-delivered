import json
import tempfile
from pathlib import Path

from deepseek_asc_finetune import MODEL_PRESETS, derive_run_name, resolve_model_id
from types import SimpleNamespace

from serve_local import build_continue_config, build_endpoint_manifest, build_status_payload
from stack_data_prep import DEFAULT_TOKENIZER_ID


def test_qwen_presets_present():
    assert MODEL_PRESETS["qwen-14b"] == "Qwen/Qwen2.5-Coder-14B"
    assert MODEL_PRESETS["qwen-32b-instruct"] == "Qwen/Qwen2.5-Coder-32B-Instruct"


def test_resolve_model_id_prefers_registry():
    assert resolve_model_id("6.7b") == "deepseek-ai/deepseek-coder-6.7b-base"


def test_derive_run_name_slugs_model_id():
    assert derive_run_name("Qwen/Qwen2.5-Coder-14B") == "qwen2.5-coder-14b"


def test_continue_config_has_researcher_profile():
    config = build_continue_config()
    titles = [model["title"] for model in config["models"]]
    assert "Local Coding AI" in titles
    assert "Cruxy Researcher" in titles


def test_build_status_payload_reads_known_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "status.json").write_text(json.dumps({"phase": "complete"}), encoding="utf-8")
        payload = build_status_payload(tmp)
        assert payload["status"]["phase"] == "complete"
        assert payload["training_manifest"] is None


def test_build_endpoint_manifest_includes_adapter_path():
    args = SimpleNamespace(
        backend="transformers",
        model="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="C:/tmp/final_adapter",
        host="127.0.0.1",
        port=8000,
        role="assistant",
        served_model_name="tar-operator-ws27r2",
        workspace=".",
        trust_remote_code=False,
    )
    manifest = build_endpoint_manifest(args)
    assert manifest["adapter_path"] == "C:/tmp/final_adapter"
    assert manifest["model"] == "Qwen/Qwen2.5-7B-Instruct"


def test_default_tokenizer_is_qwen():
    assert DEFAULT_TOKENIZER_ID == "Qwen/Qwen2.5-Coder-7B"
