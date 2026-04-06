import json
import tempfile
from pathlib import Path

from deepseek_asc_finetune import MODEL_PRESETS, derive_run_name, resolve_model_id
from serve_local import build_continue_config, build_status_payload
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


def test_default_tokenizer_is_qwen():
    assert DEFAULT_TOKENIZER_ID == "Qwen/Qwen2.5-Coder-7B"
