from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tar_lab.control import DEFAULT_HOST, DEFAULT_PORT, TARControlServer, send_command
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.voice import SpeechProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAR command and control CLI")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--last-fail", action="store_true", dest="last_fail")
    parser.add_argument("--force-fail-fast", action="store_true", dest="force_fail_fast")
    parser.add_argument("--topic", default="frontier ai")
    parser.add_argument("--max-results", type=int, default=6, dest="max_results")
    parser.add_argument("--trial-id", dest="trial_id")
    parser.add_argument("--message")
    parser.add_argument("--voice-file", dest="voice_file")
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--wake-word", default="lab")
    parser.add_argument("--listen-duration", type=float, default=3.0, dest="listen_duration")
    parser.add_argument("--listen-timeout", type=float, default=15.0, dest="listen_timeout")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--serve", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--dry-run", action="store_true", dest="dry_run")
    group.add_argument("--live-docker-test", action="store_true", dest="live_docker_test")
    group.add_argument("--check-regime", action="store_true", dest="check_regime")
    group.add_argument("--ingest-research", action="store_true", dest="ingest_research")
    group.add_argument("--verify-last-trial", action="store_true", dest="verify_last_trial")
    group.add_argument("--breakthrough-report", action="store_true", dest="breakthrough_report")
    group.add_argument("--chat", action="store_true")
    group.add_argument("--pivot", action="store_true")
    group.add_argument("--explain", action="store_true")
    group.add_argument("--panic", action="store_true")
    return parser.parse_args()


def _direct_dispatch(orchestrator: TAROrchestrator, args: argparse.Namespace) -> Dict[str, Any]:
    if args.status:
        return orchestrator.status()
    if args.dry_run:
        return orchestrator.run_dry_run(force_fail_fast=args.force_fail_fast).model_dump(mode="json")
    if args.live_docker_test:
        return orchestrator.live_docker_test().model_dump(mode="json")
    if args.check_regime:
        return orchestrator.check_regime()
    if args.ingest_research:
        return orchestrator.ingest_research(topic=args.topic, max_results=args.max_results).model_dump(mode="json")
    if args.verify_last_trial:
        return orchestrator.verify_last_trial(trial_id=args.trial_id).model_dump(mode="json")
    if args.breakthrough_report:
        return orchestrator.breakthrough_report(trial_id=args.trial_id).model_dump(mode="json")
    if args.chat:
        return orchestrator.chat(_resolve_chat_prompt(orchestrator, args)).model_dump(mode="json")
    if args.pivot:
        return orchestrator.pivot_force(force=args.force)
    if args.explain:
        return orchestrator.explain_last_fail().model_dump(mode="json")
    if args.panic:
        return orchestrator.panic()
    raise ValueError("No command selected")


def _render_status(payload: Dict[str, Any]) -> str:
    recovery = payload["recovery"]
    last = payload["last_three_metrics"][-1] if payload["last_three_metrics"] else {}
    gpu = payload.get("gpu", {})
    lines = [
        f"Trial ID: {recovery.get('trial_id') or 'none'}",
        f"Status: {recovery.get('status')}",
        f"Fail-Fast Streak: {recovery.get('consecutive_fail_fast', 0)}",
        f"E: {last.get('energy_e', 'n/a')}",
        f"sigma: {last.get('entropy_sigma', 'n/a')}",
        f"rho: {last.get('drift_rho', 'n/a')}",
        f"grad: {last.get('grad_norm', 'n/a')}",
        f"D_PR: {last.get('effective_dimensionality', 'n/a')}",
        f"D ratio: {last.get('dimensionality_ratio', 'n/a')}",
        f"Eq frac: {last.get('equilibrium_fraction', 'n/a')}",
        f"GPU Temp C: {gpu.get('temperature_c', 'n/a')}",
        f"GPU Power W: {gpu.get('power_w', 'n/a')}",
    ]
    return "\n".join(lines)


def _render_regime(payload: Dict[str, Any]) -> str:
    lines = [
        f"Trial ID: {payload.get('trial_id', 'none')}",
        f"Step: {payload.get('step', 'n/a')}",
        f"Regime: {payload.get('regime', 'unknown')}",
        f"D_PR: {payload.get('effective_dimensionality', 'n/a')}",
        f"D ratio: {payload.get('dimensionality_ratio', 'n/a')}",
        f"Eq frac: {payload.get('equilibrium_fraction', 'n/a')}",
        f"Eq gate: {payload.get('equilibrium_gate', 'n/a')}",
        f"Loss: {payload.get('training_loss', 'n/a')}",
    ]
    warning = payload.get("warning")
    if warning:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def _resolve_chat_prompt(orchestrator: TAROrchestrator, args: argparse.Namespace) -> str:
    if args.message:
        return args.message
    if args.voice_file:
        processor = SpeechProcessor(
            workspace=orchestrator.workspace,
            wake_word=args.wake_word,
        )
        processor.start()
        try:
            processor.submit_audio(args.voice_file)
            prompt = processor.poll_command(timeout=args.listen_timeout)
        finally:
            processor.stop()
        if not prompt:
            raise RuntimeError("No wake-word command was detected in the supplied audio file")
        return prompt
    if args.listen:
        processor = SpeechProcessor(
            workspace=orchestrator.workspace,
            wake_word=args.wake_word,
        )
        processor.start()
        try:
            prompt = processor.listen_once(
                duration_s=args.listen_duration,
                timeout=args.listen_timeout,
            )
        finally:
            processor.stop()
        if not prompt:
            raise RuntimeError("No wake-word command was detected from the microphone")
        return prompt
    return input("lab> ").strip()


def main() -> int:
    args = parse_args()
    orchestrator = TAROrchestrator(workspace=args.workspace, start_memory_indexer=args.serve)
    try:
        orchestrator.store.append_audit_event("cli", "command", {"argv": vars(args)})

        if args.serve:
            server = TARControlServer(orchestrator, host=args.host, port=args.port)
            try:
                server.serve_forever()
            finally:
                server.shutdown()
            return 0

        if args.direct:
            payload = _direct_dispatch(orchestrator, args)
        else:
            if args.status:
                response = send_command("status", host=args.host, port=args.port)
            elif args.dry_run:
                response = send_command("dry_run", payload={"force_fail_fast": args.force_fail_fast}, host=args.host, port=args.port)
            elif args.live_docker_test:
                response = send_command("live_docker_test", host=args.host, port=args.port)
            elif args.check_regime:
                response = send_command("check_regime", host=args.host, port=args.port)
            elif args.ingest_research:
                response = send_command(
                    "ingest_research",
                    payload={"topic": args.topic, "max_results": args.max_results},
                    host=args.host,
                    port=args.port,
                )
            elif args.verify_last_trial:
                response = send_command(
                    "verify_last_trial",
                    payload={"trial_id": args.trial_id},
                    host=args.host,
                    port=args.port,
                )
            elif args.breakthrough_report:
                response = send_command(
                    "breakthrough_report",
                    payload={"trial_id": args.trial_id},
                    host=args.host,
                    port=args.port,
                )
            elif args.chat:
                response = send_command(
                    "chat",
                    payload={"prompt": _resolve_chat_prompt(orchestrator, args)},
                    host=args.host,
                    port=args.port,
                )
            elif args.pivot:
                response = send_command("pivot", payload={"force": args.force}, host=args.host, port=args.port)
            elif args.explain:
                response = send_command("explain_last_fail", host=args.host, port=args.port)
            else:
                response = send_command("panic", host=args.host, port=args.port)
            if not response.ok:
                raise SystemExit(response.error or "Unknown TAR control error")
            payload = response.payload

        if args.json:
            print(json.dumps(payload, indent=2))
        elif args.status:
            print(_render_status(payload))
        elif args.check_regime:
            print(_render_regime(payload))
        elif args.ingest_research:
            print(f"Ingested {payload.get('indexed', 0)} research documents for topic: {payload.get('topic', '')}")
        elif args.verify_last_trial or args.breakthrough_report:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return 0
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
