import json
import sys
import types
from pathlib import Path

from tar_lab.eval_harness import GoldPredictor, HeuristicPredictor, build_eval_pack, evaluate_eval_pack


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _seed_example(
    *,
    example_id: str,
    task_family: str,
    task_name: str,
    lineage_key: str,
    input_context: dict,
    target: dict,
) -> dict:
    return {
        "example_id": example_id,
        "dataset_version": "tar-master-ws23-v1",
        "lineage_key": lineage_key,
        "task_family": task_family,
        "task_name": task_name,
        "source_kind": task_family,
        "tags": [],
        "input_context": input_context,
        "target": target,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "provenance": {
            "state_file": "seed.jsonl",
            "source_id": example_id,
            "state_root": "tar_state",
            "observed": True,
            "content_hash": example_id,
        },
    }


def _seed_dataset(root: Path) -> Path:
    dataset_dir = root / "dataset_artifacts" / "tar_master_dataset_ws23_v1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_version": "tar-master-ws23-v1",
        "records": 4,
        "splits": {"train": 0, "validation": 0, "test": 4},
        "task_families": {
            "benchmark_honesty": 1,
            "project_resume": 1,
            "reproducibility_refusal": 1,
            "tcl_trace_analysis": 1,
        },
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    test_records = [
        _seed_example(
            example_id="benchmark-honesty-1",
            task_family="benchmark_honesty",
            task_name="study_truth_assessment",
            lineage_key="project:bench-1",
            input_context={
                "requested_benchmark_tier": "canonical",
                "canonical_comparable": False,
                "benchmark_truth_statuses": ["unsupported", "unsupported"],
            },
            target={
                "benchmark_alignment": "refused",
                "canonical_comparable": False,
                "recommended_operator_language": "validation_or_refused",
                "truthful_statuses": ["unsupported", "unsupported"],
            },
        ),
        _seed_example(
            example_id="project-resume-1",
            task_family="project_resume",
            task_name="project_state_to_resume_snapshot",
            lineage_key="project:resume-1",
            input_context={
                "budget_pressure_level": "medium",
                "resume_snapshot": {
                    "active_thread_id": "thread-1",
                    "current_question_id": "question-1",
                    "next_action_id": "action-1",
                },
                "next_action": {"action_kind": "run_problem_study", "status": "planned"},
            },
            target={
                "budget_pressure_level": "medium",
                "resume_snapshot": {
                    "active_thread_id": "thread-1",
                    "current_question_id": "question-1",
                    "next_action_id": "action-1",
                },
                "next_action": {"action_kind": "run_problem_study", "status": "planned"},
            },
        ),
        _seed_example(
            example_id="repro-refusal-1",
            task_family="reproducibility_refusal",
            task_name="run_manifest_to_lock_refusal",
            lineage_key="project:repro-1",
            input_context={
                "reproducibility_complete": False,
                "unresolved_packages": ["evaluate"],
            },
            target={
                "next_action": "pin_dependencies_and_rebuild_manifest",
                "operator_language": "manifest_lock_incomplete_refuse_or_downgrade",
                "should_refuse_promotion": True,
            },
        ),
        _seed_example(
            example_id="tcl-trace-1",
            task_family="tcl_trace_analysis",
            task_name="thermo_trace_to_dynamics_summary",
            lineage_key="trial:trace-1",
            input_context={
                "final_regime": "searching",
                "d_pr_trend": "stable",
                "equilibrium_trend": "stable",
                "drift_trend": "stable",
            },
            target={
                "d_pr_trend": "stable",
                "drift_trend": "stable",
                "equilibrium_trend": "stable",
                "final_regime": "searching",
                "warning": None,
            },
        ),
    ]
    _write_jsonl(dataset_dir / "tar_master_dataset_test.jsonl", test_records)
    return dataset_dir


def test_build_eval_pack_writes_manifest_and_suite_files(tmp_path: Path):
    dataset_dir = _seed_dataset(tmp_path)
    eval_pack_dir = tmp_path / "eval_artifacts" / "ws24"
    manifest = build_eval_pack(
        dataset_dir=dataset_dir,
        eval_pack_dir=eval_pack_dir,
        eval_version="tar-operator-eval-ws24-v1",
    )
    assert manifest["items"] == 4
    assert manifest["task_families"]["benchmark_honesty"] == 1
    assert manifest["task_families"]["tcl_trace_analysis"] == 1
    assert manifest["files"]["core"]["records"] == 4
    assert len(manifest["source_dataset"]["test_sha256"]) == 64
    assert (eval_pack_dir / "eval_honesty.jsonl").exists()
    assert (eval_pack_dir / "eval_tcl.jsonl").exists()
    first_item = json.loads((eval_pack_dir / "eval_core.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [message["role"] for message in first_item["messages"]] == ["system", "user"]


def test_evaluate_eval_pack_gold_predictor_scores_perfectly(tmp_path: Path):
    dataset_dir = _seed_dataset(tmp_path)
    eval_pack_dir = tmp_path / "eval_artifacts" / "ws24"
    output_dir = tmp_path / "eval_artifacts" / "runs" / "gold"
    build_eval_pack(
        dataset_dir=dataset_dir,
        eval_pack_dir=eval_pack_dir,
        eval_version="tar-operator-eval-ws24-v1",
    )
    summary = evaluate_eval_pack(
        eval_pack_dir=eval_pack_dir,
        output_dir=output_dir,
        predictor=GoldPredictor(),
    )
    assert summary["overall"]["mean_score"] == 1.0
    assert summary["overall"]["decision_accuracy"] == 1.0
    assert summary["family_breakdown"]["benchmark_honesty"]["count"] == 1
    assert json.loads((output_dir / "results.json").read_text(encoding="utf-8"))["overall"]["mean_score"] == 1.0


def test_evaluate_eval_pack_heuristic_predictor_runs_end_to_end(tmp_path: Path):
    dataset_dir = _seed_dataset(tmp_path)
    eval_pack_dir = tmp_path / "eval_artifacts" / "ws24"
    output_dir = tmp_path / "eval_artifacts" / "runs" / "heuristic"
    build_eval_pack(
        dataset_dir=dataset_dir,
        eval_pack_dir=eval_pack_dir,
        eval_version="tar-operator-eval-ws24-v1",
    )
    summary = evaluate_eval_pack(
        eval_pack_dir=eval_pack_dir,
        output_dir=output_dir,
        predictor=HeuristicPredictor(),
    )
    assert summary["overall"]["count"] == 4
    assert "tcl" in summary["suite_breakdown"]
    assert (output_dir / "errors.jsonl").exists()


def test_hf_predictor_moves_non_quantized_model_to_cuda_when_available(monkeypatch):
    moved_to: list[str] = []

    class _FakeTensor:
        def __init__(self, device: str = "cpu"):
            self.device = device

        def to(self, device):
            target = str(device)
            return _FakeTensor(target)

        @property
        def shape(self):
            return (1, 4)

        def __getitem__(self, item):
            return _FakeTensor(self.device)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        device=lambda name: name,
        bfloat16="bf16",
        float16="fp16",
        no_grad=lambda: _NoGrad(),
        ones_like=lambda prompt, device=None: _FakeTensor(str(device or prompt.device)),
    )

    class _FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 7
        chat_template = "fake"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, *args, **kwargs):
            return _FakeTensor("cpu")

        def decode(self, tokens, skip_special_tokens=True):
            return "{}"

    class _FakeModel:
        def __init__(self):
            self._device = "cpu"

        def to(self, device):
            target = str(device)
            moved_to.append(target)
            self._device = target
            return self

        def eval(self):
            return self

        def parameters(self):
            yield types.SimpleNamespace(device=self._device)

        def generate(self, **kwargs):
            return [_FakeTensor(self._device)]

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        AutoModelForCausalLM=_FakeAutoModelForCausalLM,
        BitsAndBytesConfig=lambda **kwargs: kwargs,
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from tar_lab.eval_harness import HFCausalLMPredictor

    predictor = HFCausalLMPredictor(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=None,
        local_files_only=True,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        use_4bit=False,
        bf16=True,
    )

    assert predictor.metadata()["predictor_type"] == "hf_causal_lm"
    assert moved_to == ["cuda"]


def test_asc_predictor_loads_checkpoint_and_infers_tokenizer(monkeypatch, tmp_path: Path):
    moved_to: list[str] = []
    loaded_paths: list[str] = []
    tokenizer_sources: list[str] = []

    checkpoint_dir = tmp_path / "ws27_asc_probe" / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "asc_config.json").write_text(
        json.dumps({"base_model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"}),
        encoding="utf-8",
    )

    class _FakeTensor:
        def __init__(self, device: str = "cpu"):
            self.device = device

        def to(self, device):
            target = str(device)
            return _FakeTensor(target)

        @property
        def shape(self):
            return (1, 4)

        def __getitem__(self, item):
            return _FakeTensor(self.device)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        device=lambda name: name,
        bfloat16="bf16",
        float16="fp16",
        no_grad=lambda: _NoGrad(),
        ones_like=lambda prompt, device=None: _FakeTensor(str(device or prompt.device)),
    )

    class _FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 7
        chat_template = "fake"

        @classmethod
        def from_pretrained(cls, model_name_or_path, *args, **kwargs):
            tokenizer_sources.append(model_name_or_path)
            return cls()

        def apply_chat_template(self, *args, **kwargs):
            return _FakeTensor("cpu")

        def decode(self, tokens, skip_special_tokens=True):
            return "{}"

    class _FakeASCModel:
        def __init__(self):
            self._device = "cpu"

        def to(self, device):
            target = str(device)
            moved_to.append(target)
            self._device = target
            return self

        def eval(self):
            return self

        def parameters(self):
            yield types.SimpleNamespace(device=self._device)

        def generate(self, **kwargs):
            return [_FakeTensor(self._device)]

    class _FakeASCForCausalLM:
        @classmethod
        def load(cls, path):
            loaded_paths.append(path)
            return _FakeASCModel()

    fake_transformers = types.SimpleNamespace(AutoTokenizer=_FakeTokenizer)
    fake_asc_model = types.SimpleNamespace(ASCForCausalLM=_FakeASCForCausalLM)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "asc_model", fake_asc_model)

    from tar_lab.eval_harness import ASCCausalLMPredictor

    predictor = ASCCausalLMPredictor(
        model_name_or_path=str(checkpoint_dir),
        tokenizer_name_or_path=None,
        local_files_only=True,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        bf16=True,
    )

    assert predictor.metadata()["predictor_type"] == "asc_causal_lm"
    assert loaded_paths == [str(checkpoint_dir.resolve())]
    assert tokenizer_sources == ["Qwen/Qwen2.5-Coder-7B-Instruct"]
    assert moved_to == ["cuda"]
