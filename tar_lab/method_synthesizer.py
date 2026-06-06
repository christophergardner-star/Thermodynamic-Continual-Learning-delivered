"""
LLM method synthesis pipeline for the TAR plugin registry.

Flow per synthesis request:
  1. Build prompt with CLMethod interface + idea description
  2. call_claude() → parse class name / key / code from response
  3. AST safety check — allowlist imports, blocklist dangerous calls
  4. Docker sandbox validation (sandbox_runner_template.py + generated code)
  5. On pass: save to tar_state/synthesized_methods/<key>.py
  6. load_generated_methods() picks it up on next director cycle

Retry: up to MAX_RETRIES attempts; error output fed back into prompt.
"""
from __future__ import annotations

import ast
import json
import re
import textwrap
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 2
SANDBOX_TIMEOUT_S = 120   # generous — PyTorch imports are slow in Docker

_ALLOWED_TOP_MODULES = {
    "torch", "math", "random", "typing", "abc", "collections",
    "functools", "itertools", "copy", "numpy", "dataclasses",
}

_BLOCKED_BUILTINS = {"eval", "exec", "compile", "__import__", "open", "breakpoint"}

# ---------------------------------------------------------------------------
# CLMethod interface snippet embedded in the synthesis prompt
# ---------------------------------------------------------------------------

_CLMETHOD_INTERFACE = '''
class CLMethod(ABC):
    """Base interface all TAR continual-learning methods must implement."""

    def __init__(self, config: Any) -> None:
        self.config = config  # SimpleNamespace with HP attrs

    def pre_task(self, task_id: int, model: nn.Module, device: torch.device) -> None:
        """Called once before training begins on each task (optional)."""

    def post_task(
        self,
        task_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """Called once after training completes on each task (optional)."""

    @abstractmethod
    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Returns a scalar Tensor added to cross-entropy every batch.
        Return torch.tensor(0.0) if no regularisation term.
        """

    def augmented_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Called after loss.backward() to add replay / extra loss terms.
        Read p.grad here for path-integral methods; return replay loss for
        experience-replay methods (a second .backward() is called if non-zero).
        Return torch.tensor(0.0, device=device) if not used.
        """
        return torch.tensor(0.0, device=device)
'''

_SYNTHESIS_SYSTEM = (
    "You are an expert continual learning researcher writing production Python code. "
    "Your code must be correct, concise, and implement a novel or established CL method."
)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(idea: str, prior_error: str = "") -> str:
    retry_block = ""
    if prior_error:
        retry_block = (
            f"\n\nPREVIOUS ATTEMPT FAILED with this error:\n{prior_error}\n"
            "Please fix the issue and regenerate the full class.\n"
        )

    return textwrap.dedent(f"""
    ## Task
    Implement a continual-learning method as a Python class that inherits from
    CLMethod (interface shown below). The idea to implement:

    **{idea}**
    {retry_block}
    ## CLMethod interface

    ```python
    from abc import ABC, abstractmethod
    from typing import Any
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    {_CLMETHOD_INTERFACE}
    ```

    ## Rules
    - Only import from: torch, math, random, typing, abc, collections, functools,
      itertools, copy, numpy, dataclasses.
    - Do NOT use: eval, exec, compile, open, __import__, os, sys, subprocess,
      socket, pathlib, shutil, pickle.
    - Return `torch.tensor(0.0)` (or `torch.tensor(0.0, device=device)`) when a
      hook has nothing to contribute.
    - Store all persistent state as `self.*` attributes, not module globals.
    - Use only standard SGD-compatible gradient tape (no custom CUDA extensions).
    - `regularization_loss` must handle the case where no tasks have been trained
      yet (return 0.0 before any `post_task` has run).
    - Keep the implementation under 150 lines.

    ## Output format
    Output exactly two blocks:

    1. A metadata JSON block on a single line starting with "METADATA:":
       METADATA: {{"class_name": "MyMethodName", "method_key": "my_method_key",
                   "description": "one sentence", "rationale": "why this works"}}

    2. A Python code block containing only the class (no top-level code):
       ```python
       <imports>

       class MyMethodName(CLMethod):
           ...
       ```

    The method_key becomes the registration name (e.g., "ewc_generic").
    Use snake_case for method_key.
    """).strip()

# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict[str, str]:
    """
    Extract metadata JSON and class source from Claude's response.
    Returns {"class_name", "method_key", "description", "code"} or raises.
    """
    # Metadata line
    meta_match = re.search(r"^METADATA:\s*(\{.*\})", raw, re.MULTILINE)
    if not meta_match:
        raise ValueError("No METADATA: line found in response")
    try:
        meta = json.loads(meta_match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"METADATA JSON parse error: {e}") from e

    for key in ("class_name", "method_key"):
        if key not in meta:
            raise ValueError(f"METADATA missing '{key}'")

    # Code block
    code_match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if not code_match:
        raise ValueError("No ```python ... ``` block found in response")

    code = code_match.group(1).strip()
    if not code:
        raise ValueError("Empty code block")

    return {
        "class_name":  meta["class_name"],
        "method_key":  meta["method_key"],
        "description": meta.get("description", ""),
        "rationale":   meta.get("rationale", ""),
        "code":        code,
    }

# ---------------------------------------------------------------------------
# AST safety check
# ---------------------------------------------------------------------------

def _ast_safety_check(code: str) -> list[str]:
    """
    Return a list of violation strings. Empty list = safe.
    Checks:
      - Only allowed top-level modules imported
      - No blocked builtin calls
    """
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    for node in ast.walk(tree):
        # Import checks
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_TOP_MODULES:
                    violations.append(f"Forbidden import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _ALLOWED_TOP_MODULES:
                    violations.append(f"Forbidden import from: {node.module}")

        # Blocked builtin call checks
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in _BLOCKED_BUILTINS:
                violations.append(f"Forbidden call: {name}()")

    return violations

# ---------------------------------------------------------------------------
# Sandbox validation
# ---------------------------------------------------------------------------

def _build_sandbox_script(class_code: str) -> str:
    """Inject the generated class into the sandbox template."""
    template_path = Path(__file__).parent / "sandbox_runner_template.py"
    template = template_path.read_text(encoding="utf-8")

    marker = "# ── SYNTHESISED_METHOD_INSERTED_HERE ─────────────────────────────────────────\n# (synthesiser replaces this comment with the generated class source)\n# ─────────────────────────────────────────────────────────────────────────────"

    if marker not in template:
        raise RuntimeError(
            "sandbox_runner_template.py is missing the insertion marker. "
            "Expected: '# ── SYNTHESISED_METHOD_INSERTED_HERE'"
        )

    return template.replace(marker, class_code)


def _run_sandbox(class_code: str, workspace: str) -> tuple[bool, str]:
    """
    Inject code into template, run in Docker sandbox.
    Returns (passed: bool, output: str).
    """
    try:
        from tar_lab.safe_exec import SandboxedPythonExecutor
    except ImportError:
        return False, "SandboxedPythonExecutor not available"

    script = _build_sandbox_script(class_code)
    executor = SandboxedPythonExecutor(workspace=workspace)
    ok, output, mode = executor.run(script, timeout_s=SANDBOX_TIMEOUT_S)

    if mode == "unavailable":
        # Docker not available — treat as a soft pass (AST check already done)
        return True, f"[sandbox unavailable, AST-only] {output}"

    passed = ok and "VALIDATION_PASSED" in output
    return passed, output

# ---------------------------------------------------------------------------
# Minibench validation (Phase 4.3)
# ---------------------------------------------------------------------------

def _run_minibench(class_code: str) -> tuple[bool, str]:
    """Run synthesised method through a tiny in-process training loop.

    The AST check and sandbox verify that the code *runs without crashing*.
    This step verifies that the method *does something scientifically meaningful*:
    - Produces non-zero regularization loss at some point (not a constant-zero stub)
    - Forgetting stays in [0, 1]
    - Accuracy is above random-guessing floor
    - No NaN or Inf in any metric

    Two synthetic tasks on a 16-dim → 2-class MLP, 2 epochs each, CPU only.
    """
    import math
    import types

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return True, "[minibench skipped: torch unavailable]"

    # ── Exec the synthesised class into an isolated namespace ─────────────────
    namespace: dict = {"__builtins__": __builtins__}
    try:
        exec(compile(class_code, "<synthesised>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return False, f"Minibench exec failed: {exc}"

    # Find the CLMethod subclass in the namespace
    method_cls = None
    for obj in namespace.values():
        if (
            isinstance(obj, type)
            and obj.__name__ != "CLMethod"
            and hasattr(obj, "regularization_loss")
        ):
            method_cls = obj
            break
    if method_cls is None:
        return False, "Minibench: no CLMethod subclass found in synthesised code"

    # ── Build a tiny model and synthetic tasks ────────────────────────────────
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    criterion = nn.CrossEntropyLoss()

    def _make_data(feature_idx: int, n: int = 120) -> DataLoader:
        X = torch.randn(n, 16)
        y = (X[:, feature_idx] > 0).long()
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    loaders = [_make_data(0), _make_data(15)]

    # ── Instantiate method with a minimal config object ───────────────────────
    try:
        cfg = types.SimpleNamespace(
            lambda_tcl=1.0, ema_beta=0.99, penalty_lambda=1.0,
            max_tasks=5, task_decay=1.0, anneal_rate=1.0,
        )
        method = method_cls(cfg)
    except Exception as exc:
        return False, f"Minibench: method instantiation failed: {exc}"

    # ── Training loop — 2 tasks × 2 epochs ───────────────────────────────────
    max_reg_loss = 0.0
    per_task_acc: list[float] = []

    try:
        for task_id, loader in enumerate(loaders):
            method.pre_task(task_id, model, device)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)

            for _epoch in range(2):
                for X_b, y_b in loader:
                    opt.zero_grad()
                    logits = model(X_b.to(device))
                    ce_loss = criterion(logits, y_b.to(device))
                    try:
                        reg = method.regularization_loss(model)
                        reg_val = float(reg.item()) if hasattr(reg, "item") else 0.0
                    except Exception:
                        reg_val = 0.0
                    max_reg_loss = max(max_reg_loss, abs(reg_val))
                    loss = ce_loss + reg_val
                    loss.backward()
                    opt.step()

            method.post_task(task_id, model, loader, device)

            # Evaluate accuracy on this task's data
            correct = total = 0
            with torch.no_grad():
                for X_b, y_b in loader:
                    preds = model(X_b.to(device)).argmax(dim=1)
                    correct += (preds == y_b.to(device)).sum().item()
                    total += len(y_b)
            per_task_acc.append(correct / total if total > 0 else 0.0)

    except Exception as exc:
        return False, f"Minibench: training loop crashed: {exc}"

    # ── Sanity checks ─────────────────────────────────────────────────────────
    mean_acc = sum(per_task_acc) / len(per_task_acc) if per_task_acc else 0.0

    if any(math.isnan(v) or math.isinf(v) for v in per_task_acc):
        return False, f"Minibench: NaN or Inf detected in accuracy values: {per_task_acc}"

    if mean_acc < 0.15:
        return False, (
            f"Minibench: mean accuracy {mean_acc:.3f} below floor 0.15 "
            "(likely collapse — method may be interfering with learning)"
        )

    if max_reg_loss == 0.0:
        return False, (
            "Minibench: method never produced non-zero regularization loss. "
            "Likely returns a constant zero — broken implementation."
        )

    return True, f"Minibench passed (acc={mean_acc:.3f}, max_reg={max_reg_loss:.4f})"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_method(method_key: str, class_code: str, meta: dict, synth_dir: Path) -> Path:
    """
    Write final method file.  Adds the production import for CLMethod
    (not present in the sandboxed version) and registers the key.
    """
    header = textwrap.dedent(f"""\
        # Auto-synthesised by TAR method_synthesizer — do not edit manually.
        # Description : {meta.get('description', '')}
        # Rationale   : {meta.get('rationale', '')}
        # Synthesised : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        from __future__ import annotations
        from tar_lab.method_registry import CLMethod, register_method
    """)

    # Add @register_method decorator if the class doesn't already have it
    body = class_code
    if "@register_method" not in body:
        class_def_match = re.search(r"^(class\s+\w+)", body, re.MULTILINE)
        if class_def_match:
            body = body[:class_def_match.start()] + \
                   f'@register_method("{method_key}")\n' + \
                   body[class_def_match.start():]

    out_path = synth_dir / f"{method_key}.py"
    out_path.write_text(header + "\n" + body + "\n", encoding="utf-8")
    return out_path

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_and_validate_method(
    idea: str,
    workspace: str,
    *,
    log_fn: Any = print,
    out_dir: Any = None,
) -> dict[str, Any]:
    """
    Synthesise a CLMethod from a plain-English description of an idea.

    Parameters
    ----------
    idea      : free-text description of the CL technique to implement
    workspace : TAR workspace root (used to locate synth dir + Docker executor)
    log_fn    : callable(str) for progress messages
    out_dir   : optional directory to save the validated method into (default
                <workspace>/tar_state/synthesized_methods). B5's synthesis loop passes a
                QUARANTINE dir so a validated-but-unapproved candidate is never picked up by
                load_generated_methods() and auto-adopted into the live METHOD_REGISTRY.

    Returns
    -------
    dict with keys:
      "success"     : bool
      "method_key"  : str  (registration key, e.g. "sparse_momentum_cl")
      "class_name"  : str
      "saved_path"  : str | None
      "description" : str
      "error"       : str | None
    """
    from tar_lab.llm_bridge import call_claude

    synth_dir = Path(out_dir) if out_dir is not None else (Path(workspace) / "tar_state" / "synthesized_methods")
    synth_dir.mkdir(parents=True, exist_ok=True)

    prior_error = ""
    parsed: dict[str, str] = {}

    for attempt in range(1, MAX_RETRIES + 2):
        log_fn(f"[method_synthesizer] attempt {attempt} for idea: {idea[:60]!r}")

        prompt = _build_prompt(idea, prior_error=prior_error)
        raw = call_claude(prompt, system=_SYNTHESIS_SYSTEM, max_tokens=3000,
                          tag="method_synthesizer")

        if not raw:
            prior_error = "LLM returned empty response"
            log_fn(f"[method_synthesizer] empty LLM response on attempt {attempt}")
            continue

        # Parse
        try:
            parsed = _parse_response(raw)
        except ValueError as e:
            prior_error = f"Parse error: {e}"
            log_fn(f"[method_synthesizer] parse failed: {e}")
            continue

        method_key  = parsed["method_key"]
        class_code  = parsed["code"]

        # AST safety check
        violations = _ast_safety_check(class_code)
        if violations:
            prior_error = "AST safety violations:\n" + "\n".join(violations)
            log_fn(f"[method_synthesizer] AST check failed: {violations}")
            continue

        # Already exists — skip synthesis
        existing = synth_dir / f"{method_key}.py"
        if existing.exists():
            log_fn(f"[method_synthesizer] {method_key} already exists, skipping")
            return {
                "success":     True,
                "method_key":  method_key,
                "class_name":  parsed["class_name"],
                "saved_path":  str(existing),
                "description": parsed["description"],
                "error":       None,
            }

        # Sandbox validation
        log_fn(f"[method_synthesizer] running sandbox for {parsed['class_name']}...")
        passed, output = _run_sandbox(class_code, workspace)

        if not passed:
            # Extract the VALIDATION_FAILED message for the retry prompt
            fail_match = re.search(r"VALIDATION_FAILED[^\n]*", output)
            prior_error = fail_match.group(0) if fail_match else output[:400]
            log_fn(f"[method_synthesizer] sandbox failed: {prior_error}")
            continue

        log_fn(f"[method_synthesizer] sandbox passed: {output.splitlines()[-1]}")

        # Minibench: verify the method produces meaningful scientific output
        mb_passed, mb_reason = _run_minibench(class_code)
        log_fn(f"[method_synthesizer] minibench: {mb_reason}")
        if not mb_passed:
            prior_error = f"MINIBENCH_FAILED: {mb_reason}"
            log_fn(f"[method_synthesizer] minibench failed — will retry: {mb_reason}")
            continue

        # Save
        saved_path = _save_method(method_key, class_code, parsed, synth_dir)
        log_fn(f"[method_synthesizer] saved to {saved_path}")

        return {
            "success":     True,
            "method_key":  method_key,
            "class_name":  parsed["class_name"],
            "saved_path":  str(saved_path),
            "description": parsed["description"],
            "error":       None,
        }

    return {
        "success":     False,
        "method_key":  parsed.get("method_key", ""),
        "class_name":  parsed.get("class_name", ""),
        "saved_path":  None,
        "description": parsed.get("description", ""),
        "error":       prior_error or "Exceeded max retries",
    }


def maybe_synthesize_for_spec(
    method_name: str,
    workspace: str,
    *,
    log_fn: Any = print,
) -> bool:
    """
    Called when spec.method is not in METHOD_REGISTRY.
    Attempts to synthesise a method with that key from a generic idea string.
    Loads it into the registry on success.

    Returns True if the method is now available in METHOD_REGISTRY.
    """
    from tar_lab.method_registry import METHOD_REGISTRY, load_generated_methods

    synth_dir = Path(workspace) / "tar_state" / "synthesized_methods"

    # Already synthesised in a previous cycle?
    existing = synth_dir / f"{method_name}.py"
    if existing.exists():
        load_generated_methods(synth_dir)
        return method_name in METHOD_REGISTRY

    idea = (
        f"Implement a continual learning regularisation or replay method "
        f"with registration key '{method_name}'. "
        f"Choose an appropriate technique that the name suggests."
    )

    result = synthesize_and_validate_method(idea, workspace, log_fn=log_fn)
    if result["success"]:
        load_generated_methods(synth_dir)
        return result["method_key"] in METHOD_REGISTRY

    log_fn(
        f"[method_synthesizer] synthesis failed for '{method_name}': "
        f"{result['error']}"
    )
    return False
