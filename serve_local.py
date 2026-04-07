"""
Local serving helpers for coding/research ASC checkpoints.
"""

from __future__ import annotations

import argparse
import http.server
import json
import subprocess
import sys
import time
from socketserver import ThreadingMixIn
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


def build_endpoint_manifest(args: argparse.Namespace) -> dict:
    return {
        "backend": args.backend,
        "model": args.model,
        "host": args.host,
        "port": args.port,
        "role": args.role,
        "served_model_name": args.served_model_name,
        "workspace": str(Path(args.workspace).resolve()),
        "trust_remote_code": bool(args.trust_remote_code),
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
    parser.add_argument("--role", default="assistant")
    parser.add_argument("--served-model-name", default="local-coding-ai", dest="served_model_name")
    parser.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")
    parser.add_argument("--print_continue", action="store_true")
    parser.add_argument("--print_status", action="store_true")
    parser.add_argument("--print_manifest", action="store_true")
    return parser.parse_args()


def detect_backend() -> str:
    try:
        import vllm  # noqa: F401
        return "vllm"
    except ImportError:
        return "transformers"


def _build_prompt(messages: list[dict]) -> str:
    lines: list[str] = []
    for item in messages:
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _serve_transformers(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        print("transformers backend requires torch and transformers to be installed.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=bool(args.trust_remote_code))
    model.to(device)
    model.eval()
    served_model_name = args.served_model_name

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _json(self, payload: dict, status: int = 200) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(
                    {
                        "ok": True,
                        "backend": "transformers",
                        "model": served_model_name,
                        "role": args.role,
                        "trust_remote_code": bool(args.trust_remote_code),
                    }
                )
                return
            if self.path == "/v1/models":
                self._json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": served_model_name,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "local",
                            }
                        ],
                    }
                )
                return
            self._json({"error": "not_found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._json({"error": "not_found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = payload.get("messages") or []
            prompt = _build_prompt(messages)
            max_tokens = int(payload.get("max_tokens", 128))
            temperature = float(payload.get("temperature", 0.0))
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-5),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            self._json(
                {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": served_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": completion},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    class _ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = _ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"Serving transformers checkpoint on http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _serve_mock(args: argparse.Namespace) -> int:
    served_model_name = args.served_model_name

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _json(self, payload: dict, status: int = 200) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(
                    {
                        "ok": True,
                        "backend": "mock",
                        "model": served_model_name,
                        "role": args.role,
                        "detail": "mock endpoint",
                        "trust_remote_code": bool(args.trust_remote_code),
                    }
                )
                return
            if self.path == "/v1/models":
                self._json(
                    {
                        "object": "list",
                        "data": [{"id": served_model_name, "object": "model", "owned_by": "local-mock", "created": int(time.time())}],
                    }
                )
                return
            self._json({"error": "not_found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._json({"error": "not_found"}, status=404)
                return
            self._json(
                {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": served_model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Mock endpoint response."}, "finish_reason": "stop"}],
                }
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    class _ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = _ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main() -> int:
    args = parse_args()
    if args.print_continue:
        print(json.dumps(build_continue_config(), indent=2))
        return 0
    if args.print_status:
        print(json.dumps(build_status_payload(args.workspace), indent=2))
        return 0
    if args.print_manifest:
        print(json.dumps(build_endpoint_manifest(args), indent=2))
        return 0

    model_path = Path(args.model)
    if model_path.is_dir() and (model_path / "mock_endpoint.json").exists():
        return _serve_mock(args)

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
            args.served_model_name,
        ]
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        raise SystemExit(subprocess.call(cmd))

    return _serve_transformers(args)


if __name__ == "__main__":
    raise SystemExit(main())
