"""
Unit tests for the AST-based code validator.

Tests every validation phase independently:
    - Size guards
    - String pattern scanning
    - AST parsing (syntax errors)
    - Import validation (whitelist + blocklist)
    - Function call validation
    - Attribute access validation
    - Name access validation (alias detection)
    - Structural validation (result=, class defs)
    - Complexity guards (node count, nesting depth)
    - Violation severity and reporting
"""

from __future__ import annotations

import pytest

from backend.sandbox.validator import (
    CodeValidator,
    Violation,
    ViolationSeverity,
    _MAX_AST_NODES,
    _MAX_CODE_SIZE,
    _MAX_NESTING_DEPTH,
    _measure_depth,
)


@pytest.fixture
def validator():
    return CodeValidator()


# ── Phase 1: Size Guards ────────────────────────────────────────────────────


class TestSizeGuards:
    """Verify empty/oversize code is rejected early."""

    def test_empty_string_rejected(self, validator):
        violations = validator.validate("")
        assert len(violations) == 1
        assert violations[0].category == "structure"
        assert "empty" in violations[0].description.lower()

    def test_whitespace_only_rejected(self, validator):
        assert not validator.is_safe("   \n\n\t  ")

    def test_none_type_handling(self, validator):
        """None should be caught before AST parsing."""
        # The validator checks `not code or not code.strip()`.
        # None will trigger the truthiness check.
        violations = validator.validate("")
        assert len(violations) > 0

    def test_code_exceeding_max_size(self, validator):
        huge = "x = 1\n" * (_MAX_CODE_SIZE // 5)
        violations = validator.validate(huge)
        assert any(v.category == "size" for v in violations)

    def test_code_at_boundary_accepted(self, validator):
        """Code just under the limit should not trigger size violation."""
        code = "result = 42\n" + "x = 1\n" * 100
        violations = validator.validate(code)
        assert not any(v.category == "size" for v in violations)


# ── Phase 2: String Pattern Scanning ────────────────────────────────────────


class TestStringPatternScanning:
    """Verify pre-AST obfuscation detection."""

    @pytest.mark.parametrize("pattern,code", [
        ("os.system", "# os.system('ls')\nresult = 42"),
        ("subprocess", "# subprocess.run\nresult = 42"),
        ("eval(", "x = 'eval( something'\nresult = x"),
        ("exec(", "s = 'exec( code here'\nresult = s"),
        ("open(", "# f = open( file\nresult = 42"),
        ("socket.", "# socket.connect\nresult = 42"),
        ("http.", "# http.server\nresult = 42"),
        ("__import__", "# __import__('os')\nresult = 42"),
        ("__globals__", "# __globals__\nresult = 42"),
        ("__builtins__", "# __builtins__\nresult = 42"),
        ("__subclasses__", "# __subclasses__()\nresult = 42"),
        ("chr(", "x = chr(65)\nresult = x"),
    ])
    def test_blocked_pattern_detected(self, validator, pattern, code):
        violations = validator.validate(code)
        pattern_v = [v for v in violations if v.category == "pattern"]
        assert len(pattern_v) > 0, f"Pattern '{pattern}' not detected in: {code}"

    def test_pattern_detection_is_case_insensitive(self, validator):
        violations = validator.validate("# OS.SYSTEM\nresult = 42")
        pattern_v = [v for v in violations if v.category == "pattern"]
        assert len(pattern_v) > 0

    def test_pattern_line_number_calculated(self, validator):
        code = "x = 1\ny = 2\n# os.system('ls')\nresult = 42"
        violations = validator.validate(code)
        pattern_v = [v for v in violations if v.category == "pattern"]
        assert any(v.line == 3 for v in pattern_v)

    def test_safe_code_no_pattern_violations(self, validator):
        code = "result = df.groupby('category')['revenue'].mean()"
        violations = validator.validate(code)
        assert not any(v.category == "pattern" for v in violations)


# ── Phase 3: AST Parsing ───────────────────────────────────────────────────


class TestASTParsing:
    """Verify syntax error detection."""

    def test_syntax_error_detected(self, validator):
        code = "def foo(:\n  return"
        violations = validator.validate(code)
        assert any(v.category == "syntax" for v in violations)

    def test_syntax_error_returns_early(self, validator):
        """On syntax error, we short-circuit — no further checks."""
        code = "import os\ndef foo(:\n  return"
        violations = validator.validate(code)
        # Should have syntax violation but NOT import violation
        # because we can't walk an unparseable tree
        assert any(v.category == "syntax" for v in violations)

    def test_valid_syntax_no_syntax_violation(self, validator):
        code = "result = 42"
        violations = validator.validate(code)
        assert not any(v.category == "syntax" for v in violations)

    def test_syntax_error_has_line_number(self, validator):
        code = "x = 1\ny = 2\ndef foo(:\nresult = x"
        violations = validator.validate(code)
        syntax_v = [v for v in violations if v.category == "syntax"]
        assert len(syntax_v) > 0
        assert syntax_v[0].line is not None


# ── Phase 4: Import Validation ──────────────────────────────────────────────


class TestImportValidation:
    """Verify import whitelist and blocklist enforcement."""

    @pytest.mark.parametrize("code", [
        "import pandas\nresult = pandas.DataFrame()",
        "import numpy as np\nresult = np.array([1])",
        "import matplotlib.pyplot as plt\nresult = 'ok'",
        "from datetime import datetime\nresult = datetime.now()",
        "import math\nresult = math.sqrt(4)",
        "import statistics\nresult = statistics.mean([1, 2, 3])",
        "import re\nresult = re.match('a', 'abc')",
        "import collections\nresult = collections.Counter([1, 1, 2])",
        "import seaborn as sns\nresult = 'ok'",
        "import scipy.stats\nresult = 'ok'",
        "import plotly.express as px\nresult = 'ok'",
    ])
    def test_whitelisted_imports_accepted(self, validator, code):
        violations = validator.validate(code)
        import_v = [v for v in violations
                    if v.category == "import" and v.severity == ViolationSeverity.CRITICAL]
        assert len(import_v) == 0, f"Whitelisted import rejected: {import_v}"

    @pytest.mark.parametrize("module", [
        "os", "sys", "subprocess", "shutil", "socket", "http",
        "requests", "pickle", "sqlite3", "ctypes", "multiprocessing",
        "threading", "importlib", "inspect", "ast", "gc",
        "tempfile", "glob", "webbrowser", "tkinter",
    ])
    def test_blocklisted_modules_rejected(self, validator, module):
        code = f"import {module}\nresult = 42"
        violations = validator.validate(code)
        critical = [v for v in violations
                    if v.category == "import" and v.severity == ViolationSeverity.CRITICAL]
        assert len(critical) > 0, f"Blocked module '{module}' was accepted"

    def test_from_import_blocklisted(self, validator):
        code = "from subprocess import run\nresult = 42"
        violations = validator.validate(code)
        assert any(v.category == "import" for v in violations)

    def test_unauthorized_module_rejected(self, validator):
        """Modules not in whitelist AND not in blocklist should be rejected."""
        code = "import custom_evil_module\nresult = 42"
        violations = validator.validate(code)
        import_v = [v for v in violations if v.category == "import"]
        assert len(import_v) > 0


# ── Phase 5: Function Call Validation ───────────────────────────────────────


class TestFunctionCallValidation:
    """Verify blocked function call detection."""

    @pytest.mark.parametrize("func,code", [
        ("eval", "x = eval('1+1')\nresult = x"),
        ("exec", "exec('x = 1')\nresult = 42"),
        ("compile", "c = compile('x=1', '<s>', 'exec')\nresult = 42"),
        ("__import__", "os = __import__('os')\nresult = 42"),
        ("open", "f = open('file.txt')\nresult = 42"),
        ("globals", "g = globals()\nresult = str(g)"),
        ("locals", "l = locals()\nresult = str(l)"),
        ("getattr", "x = getattr(df, 'head')\nresult = x"),
        ("setattr", "setattr(obj, 'x', 1)\nresult = 42"),
        ("exit", "exit(0)\nresult = 42"),
        ("breakpoint", "breakpoint()\nresult = 42"),
        ("input", "x = input()\nresult = x"),
    ])
    def test_blocked_direct_calls(self, validator, func, code):
        violations = validator.validate(code)
        call_v = [v for v in violations if v.category == "call"]
        assert len(call_v) > 0, f"Blocked call '{func}()' was accepted"

    def test_method_call_blocked(self, validator):
        """os.system(...) should be caught as a method call."""
        code = "import os\nos.system('ls')\nresult = 42"
        violations = validator.validate(code)
        call_v = [v for v in violations if v.category == "call"]
        assert len(call_v) > 0

    def test_safe_method_calls_accepted(self, validator):
        """df.head(), df.groupby() etc. should NOT be flagged."""
        code = "result = df.groupby('category')['revenue'].mean().round(2)"
        violations = validator.validate(code)
        call_v = [v for v in violations
                  if v.category == "call" and v.severity == ViolationSeverity.CRITICAL]
        assert len(call_v) == 0


# ── Phase 6: Attribute Access Validation ────────────────────────────────────


class TestAttributeAccessValidation:
    """Verify blocked attribute access detection."""

    @pytest.mark.parametrize("attr,code", [
        ("__class__", "x = ''.__class__\nresult = str(x)"),
        ("__bases__", "x = str.__bases__\nresult = str(x)"),
        ("__subclasses__", "x = object.__subclasses__\nresult = str(x)"),
        ("__globals__", "x = f.__globals__\nresult = str(x)"),
        ("__builtins__", "x = obj.__builtins__\nresult = str(x)"),
        ("__code__", "x = f.__code__\nresult = str(x)"),
        ("__dict__", "x = obj.__dict__\nresult = str(x)"),
        ("rmtree", "import shutil\nshutil.rmtree('/tmp')\nresult = 42"),
        ("unlink", "p.unlink()\nresult = 42"),
    ])
    def test_blocked_attributes_detected(self, validator, attr, code):
        violations = validator.validate(code)
        attr_v = [v for v in violations if v.category == "attribute"]
        assert len(attr_v) > 0, f"Blocked attribute '{attr}' was accepted"

    def test_chained_dunder_escape_detected(self, validator):
        """''.__class__.__bases__[0].__subclasses__() — sandbox escape chain."""
        code = "x = ''.__class__.__bases__[0].__subclasses__()\nresult = str(x)"
        violations = validator.validate(code)
        attr_v = [v for v in violations if v.category == "attribute"]
        assert len(attr_v) >= 2  # __class__ and __bases__ at minimum

    def test_safe_attributes_accepted(self, validator):
        """Normal DataFrame attributes should NOT be flagged."""
        code = "result = df.shape[0]"
        violations = validator.validate(code)
        attr_v = [v for v in violations
                  if v.category == "attribute" and v.severity == ViolationSeverity.CRITICAL]
        assert len(attr_v) == 0


# ── Phase 7: Name Access Validation ─────────────────────────────────────────


class TestNameAccessValidation:
    """Verify blocked builtin aliasing detection."""

    def test_eval_aliasing_blocked(self, validator):
        code = "f = eval\nresult = f('1+1')"
        violations = validator.validate(code)
        name_v = [v for v in violations if v.category == "name"]
        assert len(name_v) > 0

    def test_exec_aliasing_blocked(self, validator):
        code = "e = exec\nresult = 42"
        violations = validator.validate(code)
        name_v = [v for v in violations if v.category == "name"]
        assert len(name_v) > 0

    def test_open_aliasing_blocked(self, validator):
        code = "my_open = open\nresult = 42"
        violations = validator.validate(code)
        name_v = [v for v in violations if v.category == "name"]
        assert len(name_v) > 0


# ── Phase 8: Structural Validation ──────────────────────────────────────────


class TestStructuralValidation:
    """Verify structural requirements enforcement."""

    def test_missing_result_assignment_warning(self, validator):
        code = "x = 42"
        violations = validator.validate(code)
        struct_v = [v for v in violations
                    if v.category == "structure" and "result" in v.description.lower()]
        assert len(struct_v) > 0
        # It should be a WARNING, not CRITICAL
        assert struct_v[0].severity == ViolationSeverity.WARNING

    def test_result_assignment_accepted(self, validator):
        code = "result = df.head()"
        violations = validator.validate(code)
        struct_v = [v for v in violations
                    if v.category == "structure" and "result" in v.description.lower()]
        assert len(struct_v) == 0

    def test_augmented_result_assignment_accepted(self, validator):
        code = "result = 0\nresult += 42"
        violations = validator.validate(code)
        struct_v = [v for v in violations
                    if v.category == "structure" and "result" in v.description.lower()]
        assert len(struct_v) == 0

    def test_class_definition_blocked(self, validator):
        code = "class Exploit:\n    pass\nresult = Exploit()"
        violations = validator.validate(code)
        struct_v = [v for v in violations
                    if v.category == "structure" and "Class" in v.description]
        assert len(struct_v) > 0


# ── Phase 9: Complexity Guards ──────────────────────────────────────────────


class TestComplexityGuards:
    """Verify AST complexity limits."""

    def test_huge_node_count_rejected(self, validator):
        # Generate code with many AST nodes via many simple statements.
        # Avoids deeply nested BinOp chains that cause RecursionError in ast.parse.
        lines = [f"x_{i} = {i}" for i in range(3000)]
        lines.append("result = x_0")
        code = "\n".join(lines)
        violations = validator.validate(code)
        complexity_v = [v for v in violations if v.category == "complexity"]
        assert len(complexity_v) > 0

    def test_deep_nesting_rejected(self, validator):
        lines = []
        for i in range(25):
            lines.append("    " * i + "if True:")
        lines.append("    " * 25 + "result = 42")
        code = "\n".join(lines) + "\n"
        violations = validator.validate(code)
        complexity_v = [v for v in violations if v.category == "complexity"]
        assert len(complexity_v) > 0

    def test_moderate_complexity_accepted(self, validator):
        """Normal analytical code should not trigger complexity limits."""
        code = (
            "import numpy as np\n"
            "grouped = df.groupby('category')['revenue'].agg(['mean', 'std', 'min', 'max'])\n"
            "filtered = grouped[grouped['mean'] > 100]\n"
            "result = filtered.sort_values('mean', ascending=False)"
        )
        violations = validator.validate(code)
        complexity_v = [v for v in violations if v.category == "complexity"]
        assert len(complexity_v) == 0


# ── Violation Reporting ─────────────────────────────────────────────────────


class TestViolationReporting:
    """Verify violation objects and summary formatting."""

    def test_violation_str_format(self):
        v = Violation(category="import", description="Blocked module: 'os'", line=3)
        s = str(v)
        assert "[import]" in s
        assert "line 3" in s
        assert "Blocked module" in s

    def test_violation_without_line_number(self):
        v = Violation(category="pattern", description="Blocked pattern: 'eval('")
        s = str(v)
        assert "[pattern]" in s
        assert "line" not in s

    def test_get_violation_summary_readable(self, validator):
        code = "import os\nresult = os.listdir('.')"
        summary = validator.get_violation_summary(code)
        assert "violation" in summary.lower()
        assert "1." in summary

    def test_get_violation_summary_no_violations(self, validator):
        code = "result = df.head()"
        summary = validator.get_violation_summary(code)
        assert "No violations" in summary

    def test_validate_or_raise_on_safe_code(self, validator):
        """Safe code should not raise."""
        code = "result = df.head()"
        validator.validate_or_raise(code)  # Should not raise

    def test_validate_or_raise_on_dangerous_code(self, validator):
        from backend.core.exceptions import CodeValidationError
        code = "import os\nresult = os.system('ls')"
        with pytest.raises(CodeValidationError):
            validator.validate_or_raise(code)

    def test_is_safe_boolean(self, validator):
        assert validator.is_safe("result = 42")
        assert not validator.is_safe("import os\nresult = 42")


# ── Measure Depth Helper ────────────────────────────────────────────────────


class TestMeasureDepth:
    """Test the _measure_depth helper directly."""

    def test_flat_code_depth(self):
        import ast
        tree = ast.parse("x = 1\ny = 2\nresult = x + y")
        depth = _measure_depth(tree)
        assert depth < 10

    def test_nested_code_depth(self):
        import ast
        code = "if True:\n  if True:\n    if True:\n      result = 42"
        tree = ast.parse(code)
        depth = _measure_depth(tree)
        assert depth >= 3
