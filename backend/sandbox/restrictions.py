"""
Sandbox restrictions — Security whitelists, blacklists, and limits.

Defines the complete security boundary for generated code execution.
This is the central policy file that ``CodeValidator`` and
``SandboxExecutor`` reference.

Defense-in-Depth Model:
    Layer 1: Prompt-level instructions (tell LLM what not to do)
    Layer 2: AST validation (statically reject dangerous code)  ← THIS FILE
    Layer 3: Runtime restrictions (subprocess isolation + resource limits)

Design Principles:
    - Default-deny: only explicitly whitelisted modules are allowed
    - Dual check: both blocklist AND whitelist are enforced
    - Obfuscation resistance: string patterns catch encoding tricks
    - Layered: AST checks → string checks → runtime checks
"""

from __future__ import annotations


# ── Module Whitelist ─────────────────────────────────────────────────────────
# Only these top-level modules (and their submodules) can be imported.
# This is the PRIMARY security boundary. Everything else is blocked.

ALLOWED_MODULES: frozenset[str] = frozenset({
    # Data analysis core
    "pandas",
    "pd",
    "numpy",
    "np",

    # Visualization
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.dates",
    "matplotlib.ticker",
    "matplotlib.colors",
    "matplotlib.cm",
    "matplotlib.patches",
    "plt",
    "seaborn",
    "sns",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "plotly.subplots",
    "px",
    "go",

    # Statistics
    "scipy",
    "scipy.stats",

    # Standard library — safe, stateless modules
    "datetime",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "re",
    "json",
    "decimal",
    "fractions",
    "operator",
    "string",
    "textwrap",
    "copy",
})

# ── Module Roots ─────────────────────────────────────────────────────────────
# Top-level module names extracted from the whitelist. Submodule imports
# are validated by checking the root against this set.

ALLOWED_MODULE_ROOTS: frozenset[str] = frozenset({
    "pandas", "pd", "numpy", "np", "matplotlib", "plt",
    "seaborn", "sns", "plotly", "px", "go", "scipy",
    "datetime", "math", "statistics", "collections", "itertools",
    "functools", "re", "json", "decimal", "fractions", "operator",
    "string", "textwrap", "copy",
})

# ── Import Blacklist ─────────────────────────────────────────────────────────
# These modules are EXPLICITLY blocked. This is defense-in-depth —
# even if the whitelist check is somehow bypassed, these are rejected.

BLOCKED_MODULES: frozenset[str] = frozenset({
    # System / Process
    "os", "sys", "subprocess", "shutil", "pathlib", "platform",
    "signal", "pty", "pipes", "resource", "msvcrt", "winreg",

    # Network / Internet
    "socket", "http", "http.client", "http.server",
    "urllib", "urllib.request", "urllib.parse",
    "requests", "httpx", "aiohttp",
    "ftplib", "telnetlib", "smtplib", "poplib", "imaplib",
    "xmlrpc", "xmlrpc.client", "xmlrpc.server",
    "ssl", "asyncio",

    # Code execution / introspection
    "importlib", "importlib.util",
    "code", "codeop", "compileall", "py_compile",
    "inspect", "ast", "dis", "gc", "traceback",
    "pdb", "bdb", "profile", "cProfile", "trace",

    # Serialization (arbitrary object loading)
    "pickle", "shelve", "marshal", "copyreg",

    # File / Archive
    "tempfile", "glob", "fnmatch",
    "zipfile", "tarfile", "gzip", "bz2", "lzma", "zlib",
    "io",  # blocked to prevent file-like object creation

    # Database
    "sqlite3", "dbm",

    # Markup / Parsing
    "xml", "xml.etree", "html", "html.parser",
    "email", "mailbox",

    # UI / Display
    "webbrowser", "tkinter", "turtle",

    # Concurrency
    "multiprocessing", "threading", "concurrent",
    "queue",

    # Type / Memory introspection
    "ctypes", "builtins", "__builtin__",
    "types",

    # Testing / Logging
    "logging", "unittest", "doctest", "pytest",
})

# ── Blocked Function Calls ───────────────────────────────────────────────────
# These function/method names are NEVER allowed in any call expression.

BLOCKED_CALLS: frozenset[str] = frozenset({
    # Code execution
    "exec",
    "eval",
    "compile",
    "__import__",

    # Introspection (can be used to escape sandbox)
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",

    # File I/O
    "open",
    "fdopen",

    # Process
    "exit",
    "quit",
    "breakpoint",
    "input",
    "raw_input",

    # Shell
    "system",
    "popen",
    "call",
    "check_output",
    "check_call",

    # Misc interactive
    "help",
    "copyright",
    "credits",
    "license",
})

# ── Blocked Attributes ───────────────────────────────────────────────────────
# These attribute names are blocked in ALL attribute access expressions.
# Prevents sandbox escape via dunder introspection chains.

BLOCKED_ATTRIBUTES: frozenset[str] = frozenset({
    # Object model introspection (sandbox escape vectors)
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
    "__module__",
    "__qualname__",
    "__init_subclass__",
    "__set_name__",
    "__del__",
    "__delattr__",

    # Serialization hooks (arbitrary code via pickle)
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",

    # Process / System
    "system",
    "popen",
    "call",
    "check_output",
    "check_call",
    "run",         # subprocess.run
    "communicate",  # Popen.communicate

    # File operations
    "unlink",
    "remove",
    "rmdir",
    "rmtree",
    "rename",
    "replace",
    "makedirs",
    "mkdir",
    "write",       # except plt.savefig which writes internally
    "writelines",
    "truncate",
})

# ── Blocked String Patterns ──────────────────────────────────────────────────
# Raw string patterns checked BEFORE AST parsing. Catches obfuscation
# attempts like string concatenation: "ev" + "al(" or chr() encoding.
# These are checked case-insensitively against the source code.

BLOCKED_STRING_PATTERNS: list[str] = [
    # Code execution
    "__import__",
    "importlib",
    "eval(",
    "exec(",
    "compile(",

    # System / Process
    "os.system",
    "os.popen",
    "os.exec",
    "os.spawn",
    "os.fork",
    "os.kill",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "subprocess",

    # Network
    "socket.",
    "http.",
    "urllib.",
    "requests.",
    "httpx.",
    "aiohttp.",

    # File I/O (the wrapper template handles all legitimate file ops)
    "open(",

    # Introspection escape chains
    "__class__.__bases__",
    "__subclasses__",
    "__globals__",
    "__builtins__",

    # Encoding tricks
    "\\x",     # hex escape
    "\\u00",   # unicode escape
    "chr(",    # character code construction
    "bytes(",  # byte string construction for obfuscation
]

# ── Allowed Builtins ─────────────────────────────────────────────────────────
# Builtins available inside the sandbox. Used by the wrapper template
# to construct a restricted __builtins__ dict if needed.

ALLOWED_BUILTINS: frozenset[str] = frozenset({
    # Numeric / Type constructors
    "abs", "bin", "bool", "complex", "divmod", "float",
    "format", "hex", "int", "oct", "ord", "pow", "round",

    # Collections
    "dict", "frozenset", "list", "set", "tuple",
    "bytearray", "bytes",

    # Iteration
    "all", "any", "enumerate", "filter", "iter",
    "len", "map", "max", "min", "next", "range",
    "reversed", "slice", "sorted", "sum", "zip",

    # Type checks
    "isinstance", "issubclass", "type",
    "callable", "id",

    # String
    "chr", "hash", "repr", "str", "ascii",

    # Print — allowed so pandas display works, but stdout is captured
    "print",

    # Constants
    "True", "False", "None", "NotImplemented", "Ellipsis",

    # Exception types for try/except
    "Exception", "ValueError", "TypeError", "KeyError",
    "IndexError", "AttributeError", "ZeroDivisionError",
    "RuntimeError", "StopIteration", "OverflowError",
    "ArithmeticError", "LookupError", "UnicodeError",
    "NotImplementedError", "FileNotFoundError",
    "MemoryError", "RecursionError",
    "Warning", "UserWarning", "DeprecationWarning",
    "FutureWarning", "RuntimeWarning",
})
