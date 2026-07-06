"""
Sandbox restrictions — Security whitelists, blacklists, and limits.

Defines the security boundary for generated code execution.
This is the central policy file that ``CodeValidator`` and
``SandboxExecutor`` reference.
"""

from __future__ import annotations

# ── Module Whitelist ─────────────────────────────────────────────────────────
# Only these modules can be imported inside the sandbox.

ALLOWED_MODULES: frozenset[str] = frozenset({
    "pandas",
    "pd",
    "numpy",
    "np",
    "matplotlib",
    "matplotlib.pyplot",
    "plt",
    "datetime",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "re",
    "json",
})

# ── Import Blacklist ─────────────────────────────────────────────────────────
# These modules are explicitly blocked even if smuggled via aliasing.

BLOCKED_MODULES: frozenset[str] = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "http",
    "http.client",
    "http.server",
    "urllib",
    "urllib.request",
    "requests",
    "importlib",
    "ctypes",
    "pickle",
    "shelve",
    "marshal",
    "code",
    "codeop",
    "compileall",
    "webbrowser",
    "tkinter",
    "multiprocessing",
    "threading",
    "signal",
    "pty",
    "pipes",
    "tempfile",
    "glob",
    "fnmatch",
    "zipfile",
    "tarfile",
    "gzip",
    "bz2",
    "lzma",
    "sqlite3",
    "xml",
    "html",
    "email",
    "smtplib",
    "ftplib",
    "telnetlib",
    "xmlrpc",
    "logging",
    "unittest",
    "pdb",
    "inspect",
    "ast",
    "dis",
    "gc",
    "builtins",
    "__builtin__",
})

# ── Blocked Function Calls ───────────────────────────────────────────────────
# These function/method names are never allowed.

BLOCKED_CALLS: frozenset[str] = frozenset({
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
    "exit",
    "quit",
    "breakpoint",
    "input",
    "raw_input",
    "help",
    "copyright",
    "credits",
    "license",
})

# ── Blocked Attributes ───────────────────────────────────────────────────────
# These attribute names are never allowed in attribute access expressions.

BLOCKED_ATTRIBUTES: frozenset[str] = frozenset({
    "__class__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__builtins__",
    "__code__",
    "__func__",
    "__self__",
    "__dict__",
    "__init_subclass__",
    "__set_name__",
    "__del__",
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",
    "system",
    "popen",
    "call",
    "check_output",
    "check_call",
    "run",  # subprocess.run
})

# ── Blocked String Patterns ──────────────────────────────────────────────────
# Raw string patterns checked before AST parsing to catch obfuscation.

BLOCKED_STRING_PATTERNS: list[str] = [
    "__import__",
    "os.system",
    "os.popen",
    "subprocess",
    "eval(",
    "exec(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
    "open(",        # file I/O — only allowed through the sandbox wrapper
    ".write(",
    "socket.",
    "http.",
    "urllib.",
    "requests.",
]

# ── Allowed Builtins ─────────────────────────────────────────────────────────
# Builtins available inside the sandbox.

ALLOWED_BUILTINS: frozenset[str] = frozenset({
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "True",
    "False",
    "None",
    "NotImplemented",
    "Ellipsis",
    # Exception types for try/except
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "ZeroDivisionError",
    "RuntimeError",
    "StopIteration",
})
