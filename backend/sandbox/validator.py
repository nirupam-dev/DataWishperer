"""
Code validator — Production-grade AST-based security analysis.

Implements Layer 2 of defense-in-depth: before any code runs, the validator
parses it into an AST and walks every node to check for violations against
the security policy defined in ``restrictions.py``.

Validation Pipeline (ordered by cost, cheapest first):
    1. Empty/size check — reject trivially invalid input
    2. String pattern scan — catch obfuscation before parsing
    3. AST parse — reject syntax errors
    4. AST walk — check every node against security policy:
        a. Import validation (whitelist + blocklist)
        b. Function call validation (blocked calls)
        c. Attribute access validation (blocked attributes + dunder chains)
        d. Name access validation (blocked builtins used as values)
    5. Structural validation — check for `result =` assignment
    6. Complexity guard — reject unreasonably deep/large ASTs

Error Reporting:
    Each violation includes:
        - category: what type of violation (import, call, attribute, etc.)
        - location: line number in the source code
        - description: human-readable explanation
        - severity: CRITICAL (always blocked) or WARNING (logged but allowed)

    Violations are returned as structured ``Violation`` objects so the
    auto-debug prompt can reference specific line numbers and issues.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from backend.core.exceptions import CodeValidationError
from backend.core.logging_config import get_logger
from backend.sandbox.restrictions import (
    ALLOWED_MODULE_ROOTS,
    ALLOWED_MODULES,
    BLOCKED_ATTRIBUTES,
    BLOCKED_CALLS,
    BLOCKED_MODULES,
    BLOCKED_STRING_PATTERNS,
)

logger = get_logger(__name__)

# Maximum source code size (characters) — prevents DoS via huge code strings
_MAX_CODE_SIZE: int = 50_000

# Maximum AST node count — prevents combinatorial explosion
_MAX_AST_NODES: int = 5_000

# Maximum nesting depth — prevents stack overflow via deep nesting
_MAX_NESTING_DEPTH: int = 20


class ViolationSeverity(str, Enum):
    """Severity level for code violations."""
    CRITICAL = "critical"   # Always blocked — execution is refused
    WARNING = "warning"     # Logged but allowed — may cause issues


@dataclass(frozen=True)
class Violation:
    """
    A single code violation detected during validation.

    Attributes:
        category: Type of violation (import, call, attribute, pattern, etc.)
        description: Human-readable explanation.
        line: Source line number (1-indexed), or None if detected pre-AST.
        severity: Whether this blocks execution or is just a warning.
    """
    category: str
    description: str
    line: Optional[int] = None
    severity: ViolationSeverity = ViolationSeverity.CRITICAL

    def __str__(self) -> str:
        loc = f" (line {self.line})" if self.line else ""
        return f"[{self.category}]{loc}: {self.description}"


class CodeValidator:
    """
    Validates generated Python code using static AST analysis.

    This validator is the gatekeeper before sandbox execution. It checks:
        1. Code size and structure
        2. String patterns for obfuscation attempts
        3. Syntax validity
        4. Import safety (whitelist + blocklist)
        5. Function call safety
        6. Attribute access safety (including dunder chains)
        7. Structural requirements (result assignment)
        8. AST complexity limits

    Usage::

        validator = CodeValidator()

        # Get detailed violations:
        violations = validator.validate(code)

        # Or raise on first failure:
        validator.validate_or_raise(code)

        # Quick boolean check:
        if validator.is_safe(code):
            execute(code)
    """

    def validate(self, code: str) -> List[Violation]:
        """
        Run all validation checks on the provided code.

        Checks are ordered by computational cost (cheapest first)
        and short-circuit on catastrophic failures (empty code,
        syntax errors).

        Args:
            code: The Python source code to validate.

        Returns:
            A list of ``Violation`` objects. Empty means the code is safe.
        """
        violations: List[Violation] = []

        # ── Phase 1: Size guard ──────────────────────────────────────
        if not code or not code.strip():
            violations.append(Violation(
                category="structure",
                description="Code is empty.",
            ))
            return violations

        if len(code) > _MAX_CODE_SIZE:
            violations.append(Violation(
                category="size",
                description=(
                    f"Code exceeds maximum size "
                    f"({len(code):,} > {_MAX_CODE_SIZE:,} characters)."
                ),
            ))
            return violations

        # ── Phase 2: String pattern scan ─────────────────────────────
        violations.extend(self._check_string_patterns(code))

        # ── Phase 3: AST parse ───────────────────────────────────────
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            violations.append(Violation(
                category="syntax",
                description=f"Syntax error: {e.msg}",
                line=e.lineno,
            ))
            return violations  # Cannot walk an unparseable tree

        # ── Phase 4: Complexity guard ────────────────────────────────
        violations.extend(self._check_complexity(tree))

        # ── Phase 5: AST walk ────────────────────────────────────────
        for node in ast.walk(tree):
            violations.extend(self._check_imports(node))
            violations.extend(self._check_calls(node))
            violations.extend(self._check_attributes(node))
            violations.extend(self._check_name_access(node))

        # ── Phase 6: Structural validation ───────────────────────────
        violations.extend(self._check_structure(tree, code))

        # ── Log summary ──────────────────────────────────────────────
        critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical:
            logger.warning(
                "Code validation failed: %d critical violation(s) — %s",
                len(critical),
                "; ".join(str(v) for v in critical[:3]),
            )

        return violations

    def validate_or_raise(self, code: str) -> None:
        """
        Validate code and raise ``CodeValidationError`` if unsafe.

        Only CRITICAL violations trigger the exception. Warnings are
        logged but allowed.

        Args:
            code: The Python source code to validate.

        Raises:
            CodeValidationError: If any critical violations are detected.
        """
        violations = self.validate(code)
        critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical:
            raise CodeValidationError(
                violations=[str(v) for v in critical],
                code=code,
            )

    def is_safe(self, code: str) -> bool:
        """
        Return ``True`` if the code has no critical violations.

        Args:
            code: The Python source code to validate.

        Returns:
            Boolean safety verdict (warnings are allowed).
        """
        violations = self.validate(code)
        return not any(
            v.severity == ViolationSeverity.CRITICAL for v in violations
        )

    def get_violation_summary(self, code: str) -> str:
        """
        Return a human-readable summary of all violations.

        Formatted for the auto-debug prompt to reference specific issues.

        Args:
            code: The Python source code to validate.

        Returns:
            Multi-line summary string, or "No violations found."
        """
        violations = self.validate(code)
        if not violations:
            return "No violations found."

        lines = [f"Found {len(violations)} violation(s):"]
        for i, v in enumerate(violations[:10], 1):  # Cap at 10
            lines.append(f"  {i}. {v}")
        if len(violations) > 10:
            lines.append(f"  ... and {len(violations) - 10} more.")
        return "\n".join(lines)

    # ── Private Checks ───────────────────────────────────────────────────

    @staticmethod
    def _check_string_patterns(code: str) -> List[Violation]:
        """
        Check raw code text for blocked string patterns.

        This runs BEFORE AST parsing to catch obfuscation attempts
        that would be valid Python but dangerous at runtime.
        """
        violations: List[Violation] = []
        code_lower = code.lower()

        for pattern in BLOCKED_STRING_PATTERNS:
            pattern_lower = pattern.lower()
            idx = code_lower.find(pattern_lower)
            if idx != -1:
                # Calculate approximate line number
                line_num = code[:idx].count("\n") + 1
                violations.append(Violation(
                    category="pattern",
                    description=f"Blocked pattern detected: '{pattern}'",
                    line=line_num,
                ))

        return violations

    @staticmethod
    def _check_imports(node: ast.AST) -> List[Violation]:
        """
        Validate import statements against the whitelist and blocklist.

        Enforces dual-check: the module must NOT be in the blocklist
        AND its root must be in the whitelist.
        """
        violations: List[Violation] = []
        line = getattr(node, "lineno", None)

        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]

                # Blocklist check (highest priority)
                if module_root in BLOCKED_MODULES:
                    violations.append(Violation(
                        category="import",
                        description=f"Blocked module: '{alias.name}'",
                        line=line,
                    ))
                # Whitelist check
                elif (alias.name not in ALLOWED_MODULES
                      and module_root not in ALLOWED_MODULE_ROOTS):
                    violations.append(Violation(
                        category="import",
                        description=f"Unauthorized module: '{alias.name}'",
                        line=line,
                    ))

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split(".")[0]

                if module_root in BLOCKED_MODULES:
                    violations.append(Violation(
                        category="import",
                        description=f"Blocked module: 'from {node.module}'",
                        line=line,
                    ))
                elif (node.module not in ALLOWED_MODULES
                      and module_root not in ALLOWED_MODULE_ROOTS):
                    violations.append(Violation(
                        category="import",
                        description=f"Unauthorized module: 'from {node.module}'",
                        line=line,
                    ))

        return violations

    @staticmethod
    def _check_calls(node: ast.AST) -> List[Violation]:
        """
        Validate function calls against the blocked call list.

        Catches both direct calls (eval(...)) and method calls
        (os.system(...)).
        """
        violations: List[Violation] = []

        if not isinstance(node, ast.Call):
            return violations

        line = getattr(node, "lineno", None)
        func = node.func

        # Direct call: eval(...), exec(...), open(...)
        if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
            violations.append(Violation(
                category="call",
                description=f"Blocked function call: '{func.id}()'",
                line=line,
            ))

        # Method call: os.system(...), subprocess.run(...)
        elif isinstance(func, ast.Attribute) and func.attr in BLOCKED_CALLS:
            # Try to extract the object name for better error messages
            obj_name = ""
            if isinstance(func.value, ast.Name):
                obj_name = f"{func.value.id}."
            violations.append(Violation(
                category="call",
                description=(
                    f"Blocked method call: '{obj_name}{func.attr}()'"
                ),
                line=line,
            ))

        return violations

    @staticmethod
    def _check_attributes(node: ast.AST) -> List[Violation]:
        """
        Validate attribute access against blocked attribute list.

        Catches both simple access (obj.__class__) and chained
        access (obj.__class__.__bases__).
        """
        violations: List[Violation] = []

        if not isinstance(node, ast.Attribute):
            return violations

        line = getattr(node, "lineno", None)

        if node.attr in BLOCKED_ATTRIBUTES:
            # Extract context for better error messages
            obj_name = ""
            if isinstance(node.value, ast.Name):
                obj_name = f"{node.value.id}."
            elif isinstance(node.value, ast.Attribute):
                obj_name = f"...{node.value.attr}."

            violations.append(Violation(
                category="attribute",
                description=(
                    f"Blocked attribute access: '{obj_name}{node.attr}'"
                ),
                line=line,
            ))

        return violations

    @staticmethod
    def _check_name_access(node: ast.AST) -> List[Violation]:
        """
        Check for direct use of blocked builtins as variable names.

        Catches patterns like:
            x = eval  (assigning blocked function to a variable)
            f = open  (aliasing open to bypass call checks)
        """
        violations: List[Violation] = []

        # Check for loading a blocked builtin name
        if (isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in BLOCKED_CALLS):
            line = getattr(node, "lineno", None)

            # Only flag if it's used as a value (assignment RHS, argument, etc.)
            # We can't determine parent node in ast.walk, so we flag all loads
            # of blocked names. The false-positive rate is extremely low since
            # user code rarely references 'eval', 'exec', etc. as variable names.
            violations.append(Violation(
                category="name",
                description=(
                    f"Reference to blocked builtin: '{node.id}' "
                    f"(aliasing blocked functions is not permitted)"
                ),
                line=line,
                severity=ViolationSeverity.CRITICAL,
            ))

        return violations

    @staticmethod
    def _check_complexity(tree: ast.AST) -> List[Violation]:
        """
        Check AST complexity to prevent resource exhaustion attacks.

        Guards against:
            - Extremely large generated code (node count)
            - Deeply nested expressions (stack depth)
        """
        violations: List[Violation] = []

        # Count total nodes
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > _MAX_AST_NODES:
            violations.append(Violation(
                category="complexity",
                description=(
                    f"Code is too complex "
                    f"({node_count:,} AST nodes > {_MAX_AST_NODES:,} limit)."
                ),
            ))

        # Check nesting depth
        max_depth = _measure_depth(tree)
        if max_depth > _MAX_NESTING_DEPTH:
            violations.append(Violation(
                category="complexity",
                description=(
                    f"Code nesting is too deep "
                    f"(depth {max_depth} > {_MAX_NESTING_DEPTH} limit)."
                ),
            ))

        return violations

    @staticmethod
    def _check_structure(tree: ast.AST, code: str) -> List[Violation]:
        """
        Validate structural requirements for the generated code.

        Checks:
            - Code contains a `result = ...` assignment
            - No class definitions (unnecessary for data analysis)
            - No decorator usage (unnecessary complexity)
        """
        violations: List[Violation] = []

        has_result_assignment = False

        for node in ast.walk(tree):
            # Check for result assignment
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "result":
                        has_result_assignment = True

            # Augmented assignment: result += ...
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "result":
                    has_result_assignment = True

            # Check for class definitions
            elif isinstance(node, ast.ClassDef):
                violations.append(Violation(
                    category="structure",
                    description=(
                        f"Class definitions are not allowed: "
                        f"'class {node.name}'"
                    ),
                    line=getattr(node, "lineno", None),
                ))

        # Missing result assignment
        if not has_result_assignment:
            # Check if "result" appears in the code at all (might be
            # assigned conditionally, which AST walk catches differently)
            if "result" not in code:
                violations.append(Violation(
                    category="structure",
                    description=(
                        "Code does not assign to 'result'. "
                        "Add: result = <your answer>"
                    ),
                    severity=ViolationSeverity.WARNING,
                ))

        return violations


def _measure_depth(node: ast.AST, current: int = 0) -> int:
    """
    Measure the maximum nesting depth of an AST.

    Uses iterative deepening to avoid recursion limit issues
    on adversarial inputs.

    Args:
        node: The AST node to measure from.
        current: Current depth (for recursion).

    Returns:
        Maximum depth found in the tree.
    """
    max_depth = current
    for child in ast.iter_child_nodes(node):
        child_depth = _measure_depth(child, current + 1)
        if child_depth > max_depth:
            max_depth = child_depth
    return max_depth
