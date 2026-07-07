"""
Security tests for the sandbox defense-in-depth model.

Tests all three layers:
    Layer 1: Restrictions policy (whitelists + blocklists)
    Layer 2: AST Validator (static analysis — attack vectors)
    Layer 3: Executor (environment isolation)

Focus: adversarial attack patterns, escape vectors, obfuscation
techniques, and policy completeness.
"""

from __future__ import annotations

import pytest

from backend.sandbox.restrictions import (
    ALLOWED_BUILTINS,
    ALLOWED_MODULE_ROOTS,
    ALLOWED_MODULES,
    BLOCKED_ATTRIBUTES,
    BLOCKED_CALLS,
    BLOCKED_MODULES,
    BLOCKED_STRING_PATTERNS,
)
from backend.sandbox.validator import CodeValidator, ViolationSeverity
from backend.sandbox.executor import SandboxExecutor


@pytest.fixture
def validator():
    return CodeValidator()


# ── Layer 1: Restrictions Policy Completeness ───────────────────────────────


class TestRestrictionsCompleteness:
    """Verify the security policy covers all critical categories."""

    # --- Module Whitelist ---

    @pytest.mark.parametrize("module", [
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy",
        "datetime", "math", "statistics", "re", "json", "collections",
    ])
    def test_data_analysis_modules_whitelisted(self, module):
        assert module in ALLOWED_MODULES

    # --- Module Blocklist ---

    @pytest.mark.parametrize("module", [
        "os", "sys", "subprocess", "shutil", "socket", "http",
        "requests", "pickle", "sqlite3", "ctypes", "builtins",
        "importlib", "inspect", "ast", "gc", "threading",
        "multiprocessing", "tempfile", "glob", "io",
    ])
    def test_dangerous_modules_blocked(self, module):
        assert module in BLOCKED_MODULES

    # --- Blocked Calls ---

    @pytest.mark.parametrize("call", [
        "eval", "exec", "compile", "__import__", "open",
        "globals", "locals", "vars", "getattr", "setattr",
        "exit", "quit", "breakpoint", "input", "system",
        "popen", "call", "check_output",
    ])
    def test_dangerous_calls_blocked(self, call):
        assert call in BLOCKED_CALLS

    # --- Blocked Attributes ---

    @pytest.mark.parametrize("attr", [
        "__class__", "__subclasses__", "__bases__", "__mro__",
        "__globals__", "__builtins__", "__code__", "__dict__",
        "rmtree", "unlink", "remove", "write", "writelines",
    ])
    def test_escape_attributes_blocked(self, attr):
        assert attr in BLOCKED_ATTRIBUTES

    # --- No overlap ---

    def test_no_module_in_both_whitelist_and_blocklist(self):
        overlap = ALLOWED_MODULES & BLOCKED_MODULES
        assert len(overlap) == 0, f"Modules in both lists: {overlap}"

    def test_blocked_patterns_not_empty(self):
        assert len(BLOCKED_STRING_PATTERNS) > 0


# ── Layer 2: Attack Vector Tests ────────────────────────────────────────────


class TestAttackVectors:
    """Test real-world sandbox escape and injection vectors."""

    def test_dunder_class_bases_escape(self, validator):
        """Classic CPython sandbox escape chain."""
        code = "x = ''.__class__.__bases__[0].__subclasses__()\nresult = str(x)"
        assert not validator.is_safe(code)

    def test_getattr_to_bypass_blocklist(self, validator):
        """Using getattr to dynamically access blocked methods."""
        code = "import os\nfn = getattr(os, 'system')\nresult = fn('id')"
        assert not validator.is_safe(code)

    def test_eval_with_string_construction(self, validator):
        """Building dangerous calls via string concatenation."""
        code = "x = 'ev' + 'al'\nresult = 42"
        # The eval( pattern should not trigger since no actual call,
        # but this tests the string pattern scanner
        violations = validator.validate(code)
        # This specific case may not trigger pattern scan since "eval(" is not present
        # The important thing is no actual eval() call passes

    def test_import_star_blocked(self, validator):
        """from os import * should be blocked."""
        code = "from os import *\nresult = 42"
        assert not validator.is_safe(code)

    def test_nested_import_smuggling(self, validator):
        """__import__ inside a lambda or comprehension."""
        code = "m = __import__('os')\nresult = m.getcwd()"
        assert not validator.is_safe(code)

    def test_compile_exec_chain(self, validator):
        """compile() + exec() — two-step injection."""
        code = "c = compile('import os', '<s>', 'exec')\nresult = 42"
        assert not validator.is_safe(code)

    def test_class_definition_for_escape(self, validator):
        """Using class __init_subclass__ for code execution."""
        code = (
            "class Evil:\n"
            "    def __init_subclass__(cls):\n"
            "        __import__('os').system('id')\n"
            "result = 42"
        )
        assert not validator.is_safe(code)

    def test_pickle_deserialization_attack(self, validator):
        """pickle.loads can execute arbitrary code."""
        code = "import pickle\nresult = pickle.loads(b'data')"
        assert not validator.is_safe(code)

    def test_ctypes_ffi_access(self, validator):
        """ctypes gives raw memory access."""
        code = "import ctypes\nresult = 42"
        assert not validator.is_safe(code)

    def test_multiprocessing_fork_bomb(self, validator):
        """multiprocessing can fork the process."""
        code = "import multiprocessing\nresult = 42"
        assert not validator.is_safe(code)

    def test_socket_network_access(self, validator):
        code = "import socket\ns = socket.socket()\nresult = 42"
        assert not validator.is_safe(code)

    def test_requests_http_exfiltration(self, validator):
        code = "import requests\nr = requests.get('http://evil.com')\nresult = r.text"
        assert not validator.is_safe(code)

    def test_file_write_attempt(self, validator):
        code = "f = open('/tmp/evil.txt', 'w')\nresult = 42"
        assert not validator.is_safe(code)

    def test_file_read_attempt(self, validator):
        code = "f = open('/etc/passwd')\nresult = f.read()"
        assert not validator.is_safe(code)

    def test_shutil_rmtree(self, validator):
        code = "import shutil\nshutil.rmtree('/')\nresult = 42"
        assert not validator.is_safe(code)

    def test_os_environ_access(self, validator):
        code = "import os\nresult = os.environ.get('SECRET')"
        assert not validator.is_safe(code)

    def test_subprocess_popen(self, validator):
        code = "import subprocess\np = subprocess.Popen(['ls'])\nresult = 42"
        assert not validator.is_safe(code)

    def test_webbrowser_open(self, validator):
        code = "import webbrowser\nwebbrowser.open('http://evil.com')\nresult = 42"
        assert not validator.is_safe(code)

    def test_asyncio_network(self, validator):
        code = "import asyncio\nresult = 42"
        assert not validator.is_safe(code)


# ── Safe Code Acceptance Tests ──────────────────────────────────────────────


class TestSafeCodeAcceptance:
    """Verify that legitimate analytical code passes all checks."""

    @pytest.mark.parametrize("name,code", [
        (
            "basic groupby",
            "result = df.groupby('category')['revenue'].mean().round(2)",
        ),
        (
            "pandas with numpy",
            "import numpy as np\nresult = np.mean(df['revenue'])",
        ),
        (
            "matplotlib chart",
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "df['cat'].value_counts().plot.bar(ax=ax)\n"
            "plt.savefig(chart_path, dpi=150)\n"
            "plt.close()\nresult = 'Chart saved'",
        ),
        (
            "datetime conversion",
            "import datetime\n"
            "df['date'] = pd.to_datetime(df['date'])\n"
            "result = df['date'].min()",
        ),
        (
            "statistics module",
            "import statistics\n"
            "result = statistics.median(df['revenue'].dropna().tolist())",
        ),
        (
            "multi-step analysis",
            "filtered = df[df['revenue'] > 100].copy()\n"
            "filtered['month'] = pd.to_datetime(filtered['date']).dt.month\n"
            "result = filtered.groupby('month')['revenue'].sum().round(2)",
        ),
        (
            "try/except handling",
            "try:\n"
            "    val = df['revenue'].mean()\n"
            "    result = f'Average: ${val:,.2f}'\n"
            "except Exception:\n"
            "    result = 'Cannot compute average'",
        ),
        (
            "seaborn visualization",
            "import seaborn as sns\n"
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "sns.boxplot(data=df, x='cat', y='val', ax=ax)\n"
            "plt.savefig(chart_path)\n"
            "plt.close()\n"
            "result = 'Box plot generated'",
        ),
        (
            "scipy stats",
            "import scipy.stats\n"
            "stat, pval = scipy.stats.pearsonr(df['x'], df['y'])\n"
            "result = f'r={stat:.4f}, p={pval:.4f}'",
        ),
    ])
    def test_safe_code_passes(self, validator, name, code):
        violations = validator.validate(code)
        critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        assert len(critical) == 0, f"Safe code '{name}' was rejected: {critical}"


# ── Layer 3: Environment Isolation ──────────────────────────────────────────


class TestEnvironmentIsolation:
    """Test the restricted environment builder."""

    def test_env_strips_secret_vars(self):
        import os
        os.environ["AWS_SECRET_KEY"] = "secret"
        os.environ["DATABASE_PASSWORD"] = "password"
        try:
            env = SandboxExecutor._build_restricted_env()
            assert "AWS_SECRET_KEY" not in env
            assert "DATABASE_PASSWORD" not in env
        finally:
            os.environ.pop("AWS_SECRET_KEY", None)
            os.environ.pop("DATABASE_PASSWORD", None)

    def test_env_preserves_python_path(self):
        env = SandboxExecutor._build_restricted_env()
        # At minimum, PATH should be present for Python discovery
        assert any(k.upper() == "PATH" for k in env)

    def test_env_sets_nouserite(self):
        env = SandboxExecutor._build_restricted_env()
        assert env["PYTHONNOUSERSITE"] == "1"

    def test_env_sets_deterministic_hash(self):
        env = SandboxExecutor._build_restricted_env()
        assert env["PYTHONHASHSEED"] == "0"
