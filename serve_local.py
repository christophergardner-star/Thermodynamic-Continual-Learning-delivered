"""
Local serving helpers for coding/research ASC checkpoints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_continue_config(base_url: str = "http://localhost:8000/v1") -> dict:
    return {
        "models": [
            {
                "title": "Local Coding AI",
                "provider": "openai",
                "model": "local-coding-ai",
                "apiBase": base_url,
                "apiKey": "local",
            },
            {
                "title": "Cruxy Researcher",
                "provider": "openai",
                "model": "local-coding-ai",
                "apiBase": base_url,
                "apiKey": "local",
                "systemMessage": "Separate fact, measured result, inference, and hypothesis.",
            },
        ]
    }


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_status_payload(workspace: str = ".") -> dict:
    root = Path(workspace)
    return {
        "status": load_optional_json(root / "status.json"),
        "training_manifest": load_optional_json(root / "training_manifest.json"),
        "corpus_manifest": load_optional_json(root / "corpus_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local coding ASC model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--print_continue", action="store_true")
    parser.add_argument("--print_status", action="store_true")
    return parser.parse_args()


def detect_backend() -> str:
    try:
        import vllm  # noqa: F401
        return "vllm"
    except ImportError:
        return "transformers"


def main() -> int:
    args = parse_args()
    if args.print_continue:
        print(json.dumps(build_continue_config(), indent=2))
        return 0
    if args.print_status:
        print(json.dumps(build_status_payload(args.workspace), indent=2))
        return 0

    backend = detect_backend() if args.backend == "auto" else args.backend
    if backend == "vllm":
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--max-model-len",
            str(args.max_model_len),
            "--tensor-parallel-size",
            str(args.tensor_parallel),
            "--served-model-name",
            "local-coding-ai",
            "--trust-remote-code",
        ]
        raise SystemExit(subprocess.call(cmd))

    print("Transformers fallback is a documentation stub in this repo.")
    print("Use --print_continue or --print_status, or install vLLM for serving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
