"""
Code validator — AST-based security analysis of generated Python code.

Implements Layer 2 of defense-in-depth: before any code runs, the validator
parses it into an AST and walks every node to check for violations against
the security policy defined in ``restrictions.py``.
"""

from __future__ import annotations

import ast
from typing import List, Optional

from backend.core.exceptions import CodeValidationError
from backend.core.logging_config import get_logger
from backend.sandbox.restrictions import (
    ALLOWED_MODULES,
    BLOCKED_ATTRIBUTES,
    BLOCKED_CALLS,
    BLOCKED_MODULES,
    BLOCKED_STRING_PATTERNS,
)

logger = get_logger(__name__)


class CodeValidator:
    """
    Validates generated Python code using static AST analysis.

    This validator is the gatekeeper before sandbox execution. It checks:
        1. Syntax validity (can the code be parsed?)
        2. Import safety (only whitelisted modules)
        3. Function call safety (no dangerous builtins)
        4. Attribute access safety (no dunder abuse)
        5. String literal safety (no obfuscated attacks)
    """

    def validate(self, code: str) -> List[str]:
        """
        Run all validation checks on the provided code.

        Args:
            code: The Python source code to validate.

        Returns:
            A list of violation descriptions. Empty means the code is safe.

        Raises:
            CodeValidationError: If any violations are found (when called
                via ``validate_or_raise``).
        """
        violations: List[str] = []

        # Pre-AST string pattern check (catches obfuscation attempts)
        violations.extend(self._check_string_patterns(code))

        # Parse AST
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            violations.append(f"Syntax error: {e.msg} (line {e.lineno})")
            return violations

        # Walk every node
        for node in ast.walk(tree):
            violations.extend(self._check_imports(node))
            violations.extend(self._check_calls(node))
            violations.extend(self._check_attributes(node))

        if violations:
            logger.warning(
                "Code validation failed with %d violation(s): %s",
                len(violations), "; ".join(violations[:3]),
            )

        return violations

    def validate_or_raise(self, code: str) -> None:
        """
        Validate code and raise ``CodeValidationError`` if unsafe.

        Args:
            code: The Python source code to validate.

        Raises:
            CodeValidationError: If any violations are detected.
        """
        violations = self.validate(code)
        if violations:
            raise CodeValidationError(violations=violations, code=code)

    def is_safe(self, code: str) -> bool:
        """
        Return ``True`` if the code passes all safety checks.

        Args:
            code: The Python source code to validate.

        Returns:
            Boolean safety verdict.
        """
        return len(self.validate(code)) == 0

    # ── Private Checks ───────────────────────────────────────────────────

    @staticmethod
    def _check_string_patterns(code: str) -> List[str]:
        """Check raw code text for blocked string patterns."""
        violations: List[str] = []
        code_lower = code.lower()
        for pattern in BLOCKED_STRING_PATTERNS:
            if pattern.lower() in code_lower:
                violations.append(f"Blocked pattern detected: '{pattern}'")
        return violations

    @staticmethod
    def _check_imports(node: ast.AST) -> List[str]:
        """Check import statements against module whitelist."""
        violations: List[str] = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in BLOCKED_MODULES:
                    violations.append(f"Blocked import: '{alias.name}'")
                elif alias.name not in ALLOWED_MODULES and module_root not in ALLOWED_MODULES:
                    violations.append(f"Unauthorized import: '{alias.name}'")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split(".")[0]
                if module_root in BLOCKED_MODULES:
                    violations.append(f"Blocked import: 'from {node.module}'")
                elif node.module not in ALLOWED_MODULES and module_root not in ALLOWED_MODULES:
                    violations.append(f"Unauthorized import: 'from {node.module}'")

        return violations

    @staticmethod
    def _check_calls(node: ast.AST) -> List[str]:
        """Check function calls against the blocked call list."""
        violations: List[str] = []

        if isinstance(node, ast.Call):
            func = node.func

            # Direct call: eval(...), exec(...)
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                violations.append(f"Blocked function call: '{func.id}()'")

            # Method call: os.system(...)
            elif isinstance(func, ast.Attribute) and func.attr in BLOCKED_CALLS:
                violations.append(f"Blocked method call: '.{func.attr}()'")

        return violations

    @staticmethod
    def _check_attributes(node: ast.AST) -> List[str]:
        """Check attribute access against blocked attribute list."""
        violations: List[str] = []

        if isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRIBUTES:
            violations.append(f"Blocked attribute access: '.{node.attr}'")

        return violations
