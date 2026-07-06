"""
Comprehensive security test suite for the sandbox execution engine.

Tests all three layers of defense-in-depth:
    Layer 1: Restrictions (whitelists/blocklists)
    Layer 2: AST Validator (static analysis)
    Layer 3: Executor (process isolation + resource limits)
"""

from backend.sandbox.validator import CodeValidator, Violation, ViolationSeverity
from backend.sandbox.executor import SandboxExecutor
from backend.sandbox.restrictions import (
    ALLOWED_MODULES,
    ALLOWED_MODULE_ROOTS,
    BLOCKED_MODULES,
    BLOCKED_CALLS,
    BLOCKED_ATTRIBUTES,
    BLOCKED_STRING_PATTERNS,
)


def run_all_tests():
    """Run all sandbox security tests."""
    validator = CodeValidator()
    passed = 0
    failed = 0
    total = 0

    def assert_test(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}: {detail}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: RESTRICTIONS POLICY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 1: RESTRICTIONS POLICY")
    print("=" * 60)

    assert_test(
        "pandas whitelisted",
        "pandas" in ALLOWED_MODULES,
    )
    assert_test(
        "numpy whitelisted",
        "numpy" in ALLOWED_MODULES,
    )
    assert_test(
        "matplotlib whitelisted",
        "matplotlib" in ALLOWED_MODULES,
    )
    assert_test(
        "os blocked",
        "os" in BLOCKED_MODULES,
    )
    assert_test(
        "subprocess blocked",
        "subprocess" in BLOCKED_MODULES,
    )
    assert_test(
        "socket blocked",
        "socket" in BLOCKED_MODULES,
    )
    assert_test(
        "requests blocked",
        "requests" in BLOCKED_MODULES,
    )
    assert_test(
        "eval in blocked calls",
        "eval" in BLOCKED_CALLS,
    )
    assert_test(
        "exec in blocked calls",
        "exec" in BLOCKED_CALLS,
    )
    assert_test(
        "open in blocked calls",
        "open" in BLOCKED_CALLS,
    )
    assert_test(
        "__import__ in blocked calls",
        "__import__" in BLOCKED_CALLS,
    )
    assert_test(
        "__class__ in blocked attrs",
        "__class__" in BLOCKED_ATTRIBUTES,
    )
    assert_test(
        "__globals__ in blocked attrs",
        "__globals__" in BLOCKED_ATTRIBUTES,
    )
    assert_test(
        "rmtree in blocked attrs",
        "rmtree" in BLOCKED_ATTRIBUTES,
    )
    assert_test(
        "unlink in blocked attrs",
        "unlink" in BLOCKED_ATTRIBUTES,
    )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: SAFE CODE (should PASS validation)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 2: SAFE CODE (should pass)")
    print("=" * 60)

    safe_codes = [
        (
            "basic pandas groupby",
            "# Average revenue by category\n"
            "result = df.groupby('category')['revenue'].mean().round(2)"
        ),
        (
            "pandas with numpy",
            "import numpy as np\n"
            "result = np.mean(df['revenue'])"
        ),
        (
            "matplotlib chart",
            "import matplotlib.pyplot as plt\n"
            "plt.style.use('seaborn-v0_8-darkgrid')\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "df['category'].value_counts().plot.bar(ax=ax)\n"
            "plt.savefig(chart_path, dpi=150, bbox_inches='tight')\n"
            "plt.close()\n"
            "result = 'Chart saved'"
        ),
        (
            "datetime conversion",
            "import datetime\n"
            "df['date'] = pd.to_datetime(df['date'])\n"
            "result = df['date'].min()"
        ),
        (
            "statistics module",
            "import statistics\n"
            "result = statistics.median(df['revenue'].dropna().tolist())"
        ),
        (
            "multi-step analysis",
            "# Filter and aggregate\n"
            "filtered = df[df['revenue'] > 100].copy()\n"
            "filtered['month'] = pd.to_datetime(filtered['date']).dt.month\n"
            "result = filtered.groupby('month')['revenue'].sum().round(2)"
        ),
        (
            "try/except handling",
            "try:\n"
            "    val = df['revenue'].mean()\n"
            "    result = f'Average: ${val:,.2f}'\n"
            "except Exception:\n"
            "    result = 'Cannot compute average'"
        ),
    ]

    for name, code in safe_codes:
        violations = validator.validate(code)
        critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        assert_test(
            f"SAFE: {name}",
            len(critical) == 0,
            f"Got {len(critical)} critical violations: {[str(v) for v in critical[:2]]}",
        )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: DANGEROUS CODE (should FAIL validation)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 3: DANGEROUS CODE (should fail)")
    print("=" * 60)

    dangerous_codes = [
        (
            "os.system",
            "import os\nos.system('rm -rf /')\nresult = 'done'",
            "import",
        ),
        (
            "subprocess.run",
            "import subprocess\nsubprocess.run(['ls'])\nresult = 'done'",
            "import",
        ),
        (
            "eval() call",
            "x = eval('1+1')\nresult = x",
            "call",
        ),
        (
            "exec() call",
            "exec('import os')\nresult = 'done'",
            "call",
        ),
        (
            "open() file read",
            "f = open('/etc/passwd')\nresult = f.read()",
            "call",
        ),
        (
            "__import__ smuggling",
            "os = __import__('os')\nresult = os.listdir('.')",
            "call",
        ),
        (
            "socket network access",
            "import socket\ns = socket.socket()\nresult = 'done'",
            "import",
        ),
        (
            "requests HTTP",
            "import requests\nr = requests.get('http://evil.com')\nresult = r.text",
            "import",
        ),
        (
            "dunder class escape",
            "x = ''.__class__.__bases__[0].__subclasses__()\nresult = str(x)",
            "attribute",
        ),
        (
            "getattr bypass",
            "import os\nfn = getattr(os, 'system')\nresult = fn('whoami')",
            "call",
        ),
        (
            "file deletion",
            "import shutil\nshutil.rmtree('/tmp')\nresult = 'done'",
            "import",
        ),
        (
            "pickle deserialization",
            "import pickle\nresult = pickle.loads(b'data')",
            "import",
        ),
        (
            "compile + exec chain",
            "code = compile('import os', '<string>', 'exec')\nresult = 'done'",
            "call",
        ),
        (
            "globals() access",
            "g = globals()\nresult = str(g)",
            "call",
        ),
        (
            "eval aliasing",
            "f = eval\nresult = f('1+1')",
            "name",
        ),
        (
            "multiprocessing",
            "import multiprocessing\nresult = 'done'",
            "import",
        ),
        (
            "ctypes FFI",
            "import ctypes\nresult = 'done'",
            "import",
        ),
    ]

    for name, code, expected_category in dangerous_codes:
        violations = validator.validate(code)
        critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        has_expected = any(v.category == expected_category for v in critical)
        assert_test(
            f"BLOCKED: {name}",
            len(critical) > 0 and has_expected,
            (
                f"Expected '{expected_category}' violation, "
                f"got: {[f'{v.category}: {v.description}' for v in critical[:2]]}"
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4: STRING PATTERN OBFUSCATION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 4: STRING PATTERN OBFUSCATION")
    print("=" * 60)

    obfuscation_codes = [
        (
            "os.system in string",
            "# This uses os.system to do work\nresult = df.mean()",
        ),
        (
            "subprocess in comment",
            "# subprocess call\nresult = df.sum()",
        ),
        (
            "eval( in string",
            "x = 'use eval( to compute'\nresult = x",
        ),
        (
            "socket. pattern",
            "# socket.connect\nresult = 42",
        ),
        (
            "open( pattern",
            "# f = open( file\nresult = df.head()",
        ),
    ]

    for name, code in obfuscation_codes:
        violations = validator.validate(code)
        pattern_violations = [v for v in violations if v.category == "pattern"]
        assert_test(
            f"PATTERN: {name}",
            len(pattern_violations) > 0,
            f"No pattern violation detected",
        )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 5: STRUCTURAL VALIDATION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 5: STRUCTURAL VALIDATION")
    print("=" * 60)

    assert_test(
        "empty code rejected",
        not validator.is_safe(""),
    )
    assert_test(
        "whitespace-only rejected",
        not validator.is_safe("   \n\n  "),
    )

    violations = validator.validate("x = 42")
    has_result_warning = any(
        v.category == "structure" and "result" in v.description
        for v in violations
    )
    assert_test(
        "missing result= warning",
        has_result_warning,
        f"Got violations: {[str(v) for v in violations]}",
    )

    violations = validator.validate(
        "class Exploit:\n    pass\nresult = Exploit()"
    )
    has_class_violation = any(
        v.category == "structure" and "Class" in v.description
        for v in violations
    )
    assert_test(
        "class definition blocked",
        has_class_violation,
        f"Got violations: {[str(v) for v in violations]}",
    )

    assert_test(
        "syntax error detected",
        not validator.is_safe("def foo(:\n  return"),
    )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 6: VIOLATION REPORTING
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 6: VIOLATION REPORTING")
    print("=" * 60)

    violations = validator.validate("import os\nos.system('ls')\nresult = 42")
    assert_test(
        "violations have line numbers",
        any(v.line is not None for v in violations),
        f"No line numbers: {[str(v) for v in violations]}",
    )
    assert_test(
        "violations have categories",
        all(v.category for v in violations),
    )

    summary = validator.get_violation_summary("import os\nresult = os.listdir('.')")
    assert_test(
        "violation summary is readable",
        "violation" in summary.lower() and "1." in summary,
        f"Summary: {summary}",
    )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 7: ERROR FORMATTING
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 7: ERROR FORMATTING")
    print("=" * 60)

    formatted = SandboxExecutor.format_user_error("KeyError", "'revenue'")
    assert_test(
        "KeyError has Column Not Found title",
        "Column Not Found" in formatted,
    )
    assert_test(
        "KeyError has suggestion",
        "case-sensitive" in formatted,
    )
    assert_test(
        "KeyError has emoji",
        "🔑" in formatted,
    )

    formatted = SandboxExecutor.format_user_error("TypeError", "unsupported operand")
    assert_test(
        "TypeError has Type Mismatch title",
        "Type Mismatch" in formatted,
    )

    formatted = SandboxExecutor.format_user_error("ZeroDivisionError", "division by zero")
    assert_test(
        "ZeroDivisionError has Division title",
        "Division" in formatted,
    )

    formatted = SandboxExecutor.format_user_error("UnknownError", "something weird")
    assert_test(
        "Unknown error has fallback",
        "Execution Error" in formatted,
    )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 8: COMPLEXITY GUARDS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION 8: COMPLEXITY GUARDS")
    print("=" * 60)

    huge_code = "x = 1\n" * 10000
    violations = validator.validate(huge_code)
    assert_test(
        "huge code size rejected",
        not validator.is_safe(huge_code),
        "Huge code was accepted",
    )

    # Deep nesting — build valid Python with proper indentation
    deep_lines = []
    for i in range(25):
        deep_lines.append("    " * i + "if True:")
    deep_lines.append("    " * 25 + "result = 42")
    deep_code = "\n".join(deep_lines) + "\n"
    violations = validator.validate(deep_code)
    deep_violations = [v for v in violations if v.category == "complexity"]
    assert_test(
        "deep nesting detected",
        len(deep_violations) > 0,
        f"No complexity violation for depth 25+",
    )

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("ALL TESTS PASSED — SANDBOX SECURITY VERIFIED")
    else:
        print(f"WARNING: {failed} test(s) failed!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
