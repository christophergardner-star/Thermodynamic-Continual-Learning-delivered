"""AST safety-check regression tests for the method synthesizer.

The check must block the dangerous BUILTINS (eval/exec/open/compile/__import__ called
as bare names) and forbidden imports, but MUST allow legitimate torch METHOD calls
(model.eval(), model.train(), torch.compile()) — flagging those attribute calls made
synthesis reject essentially every realistic CLMethod and never succeed.
"""
from __future__ import annotations

from tar_lab.method_synthesizer import _ast_safety_check


def test_allows_torch_method_calls():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "class M:\n"
        "    def post_task(self, model):\n"
        "        model.eval()\n"
        "        model.train()\n"
        "        torch.compile(model)\n"
        "        return torch.tensor(0.0)\n"
    )
    assert _ast_safety_check(code) == []


def test_blocks_dangerous_builtins():
    for bad in ("eval('1+1')", "exec('x=1')", "open('f')", "__import__('os')",
                "compile('1','<s>','eval')", "breakpoint()"):
        code = f"def f():\n    return {bad}\n"
        assert _ast_safety_check(code), f"should have blocked builtin: {bad}"


def test_blocks_forbidden_imports():
    assert any("import" in v.lower() for v in _ast_safety_check("import os\n"))
    assert any("import" in v.lower() for v in _ast_safety_check("import subprocess\n"))
    assert any("import" in v.lower() for v in _ast_safety_check("from os import system\n"))


def test_allows_allowlisted_imports():
    assert _ast_safety_check("import torch\nimport numpy as np\nimport math\n") == []
