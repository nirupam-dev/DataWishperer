"""
Sandbox module — Secure code validation and execution.

Provides a three-layer defense-in-depth model:
    1. ``restrictions.py`` — Security policy (whitelists + blocklists)
    2. ``validator.py``    — AST-based static analysis
    3. ``executor.py``     — Process-isolated execution with resource limits

Usage::

    from backend.sandbox import SandboxExecutor, CodeValidator

    validator = CodeValidator()
    if validator.is_safe(code):
        executor = SandboxExecutor()
        result = executor.execute(code, csv_path)
"""

from backend.sandbox.executor import SandboxExecutor
from backend.sandbox.validator import CodeValidator

__all__ = ["SandboxExecutor", "CodeValidator"]
