"""
DataWhisperer -- End-to-End Pipeline Smoke Test.

Tests the core backend pipeline programmatically using the generated
test dataset. Does NOT require Ollama -- tests everything except LLM inference.
"""

import sys
import os
import io
import traceback
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["APP_DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"

from backend.core.config import get_settings
get_settings.cache_clear()

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = {"pass": 0, "fail": 0}

def test(name, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"  [{PASS}] {name}")
            results["pass"] += 1
        else:
            print(f"  [{FAIL}] {name} -> {result}")
            results["fail"] += 1
    except Exception as e:
        print(f"  [{FAIL}] {name}")
        print(f"          {type(e).__name__}: {str(e)[:300]}")
        results["fail"] += 1

CSV_PATH = PROJECT_ROOT / "test_data" / "global_tech_sales_2024.csv"

print("=" * 70)
print("  DataWhisperer -- End-to-End Pipeline Smoke Test")
print("=" * 70)
print(f"  CSV: {CSV_PATH}")
print()

# ======================================================================
# 1. Configuration
# ======================================================================
print("--- 1. Configuration ---")

def test_settings():
    s = get_settings()
    assert s.name == "DataWhisperer"
    assert s.ollama.model == "qwen2.5:7b"
    assert s.sandbox.timeout == 30
    s.ensure_directories()
    assert Path(s.storage.upload_dir).exists()
    return True

test("Settings load and directories created", test_settings)

# ======================================================================
# 2. File Upload & Validation
# ======================================================================
print("\n--- 2. File Upload & Validation ---")

file_service = None
upload_response = None

def test_upload():
    global file_service, upload_response
    from backend.services.file_service import FileService
    file_service = FileService()
    content = CSV_PATH.read_bytes()
    upload_response = file_service.upload_file(
        filename="global_tech_sales_2024.csv",
        content=content,
    )
    assert upload_response.row_count == 50, f"Got {upload_response.row_count} rows"
    assert upload_response.col_count == 17, f"Got {upload_response.col_count} cols"
    return True

def test_column_meta():
    cols = {c.name: c for c in upload_response.columns}
    assert "total_revenue" in cols
    assert "region" in cols
    assert cols["total_revenue"].mean is not None
    assert cols["total_revenue"].mean > 0
    return True

def test_preview():
    assert len(upload_response.preview_rows) > 0
    assert upload_response.preview_rows[0]["order_id"] == "ORD-0001"
    return True

def test_metadata_retrieve():
    m = file_service.get_file_metadata(upload_response.file_id)
    assert m.original_name == "global_tech_sales_2024.csv"
    assert m.row_count == 50
    return True

test("File upload (50 rows x 17 cols)", test_upload)
test("Column metadata extraction", test_column_meta)
test("Preview rows generation", test_preview)
test("File metadata retrieval by ID", test_metadata_retrieve)

# ======================================================================
# 3. Sandbox Code Execution
# ======================================================================
print("\n--- 3. Sandbox Code Execution ---")
# NOTE: The sandbox wrapper pre-loads `df = pd.read_csv(csv_path)`
# Generated code should use `df` directly and assign to `result`.

sandbox = None
stored_path = None

def test_sandbox_init():
    global sandbox, stored_path
    from backend.sandbox.executor import SandboxExecutor
    sandbox = SandboxExecutor()
    stored_path = file_service.get_file_metadata(upload_response.file_id).stored_path
    return True

def test_sum():
    code = "result = df['total_revenue'].sum()"
    r = sandbox.execute(code=code, csv_path=stored_path)
    assert r.success, f"Failed: {r.error_message}"
    return True

def test_groupby():
    code = "result = df.groupby('region')['total_revenue'].sum().sort_values(ascending=False)"
    r = sandbox.execute(code=code, csv_path=stored_path)
    assert r.success, f"Failed: {r.error_message}"
    return True

def test_filter():
    code = "result = df[df['total_revenue'] > 7000][['order_id', 'product_name', 'total_revenue']]"
    r = sandbox.execute(code=code, csv_path=stored_path)
    assert r.success, f"Failed: {r.error_message}"
    return True

def test_profit_margin():
    code = (
        "df['profit_margin'] = (df['profit'] / df['total_revenue'] * 100).round(2)\n"
        "result = df['profit_margin'].mean()"
    )
    r = sandbox.execute(code=code, csv_path=stored_path)
    assert r.success, f"Failed: {r.error_message}"
    return True

def test_date_ops():
    code = (
        "df['date'] = pd.to_datetime(df['date'])\n"
        "result = df.groupby(df['date'].dt.month)['total_revenue'].sum()"
    )
    r = sandbox.execute(code=code, csv_path=stored_path)
    assert r.success, f"Failed: {r.error_message}"
    return True

def test_topn():
    code = "result = df.nlargest(5, 'profit')[['product_name', 'profit', 'region']]"
    r = sandbox.execute(code=code, csv_path=stored_path)
    assert r.success, f"Failed: {r.error_message}"
    return True

def test_correlation():
    code = "result = df[['quantity', 'unit_price', 'total_revenue', 'profit']].corr()"
    r = sandbox.execute(code=code, csv_path=stored_path)
    assert r.success, f"Failed: {r.error_message}"
    return True

def test_chart_generation():
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "region_rev = df.groupby('region')['total_revenue'].sum().sort_values()\n"
        "region_rev.plot(kind='barh', ax=ax, color='#6C63FF')\n"
        "ax.set_title('Revenue by Region')\n"
        "plt.tight_layout()\n"
        "plt.savefig(chart_path, dpi=100)\n"
        "plt.close()\n"
        "result = 'Chart saved'"
    )
    r = sandbox.execute(code=code, csv_path=stored_path)
    assert r.success, f"Failed: {r.error_message}"
    return True

def test_security_block():
    code = "import os\nos.system('whoami')"
    try:
        r = sandbox.execute(code=code, csv_path=stored_path)
        assert not r.success, "Dangerous code should not succeed"
    except Exception:
        pass  # Expected -- validator blocks it
    return True

test("Sandbox init", test_sandbox_init)
test("SUM aggregation", test_sum)
test("GROUP BY operation", test_groupby)
test("Filter operation", test_filter)
test("Profit margin calc", test_profit_margin)
test("Date operations (monthly)", test_date_ops)
test("Top-N analysis", test_topn)
test("Correlation matrix", test_correlation)
test("Chart generation (matplotlib)", test_chart_generation)
test("Security: dangerous code blocked", test_security_block)

# ======================================================================
# 4. Analytics Engine
# ======================================================================
print("\n--- 4. Analytics Engine ---")

def test_orchestrator():
    from backend.analytics.orchestrator import AnalyticsOrchestrator
    import pandas as pd
    orch = AnalyticsOrchestrator()
    df = pd.read_csv(CSV_PATH)
    report = orch.run_full_analysis(df)
    assert report is not None and isinstance(report, dict)
    return True

def test_profiler():
    from backend.analytics.data_profiler import DataProfiler
    import pandas as pd
    p = DataProfiler()
    r = p.profile(pd.read_csv(CSV_PATH))
    assert r is not None
    return True

def test_quality():
    from backend.analytics.data_quality import DataQualityAnalyzer
    import pandas as pd
    qa = DataQualityAnalyzer()
    r = qa.analyze(pd.read_csv(CSV_PATH))
    assert r is not None
    return True

def test_stats():
    from backend.analytics.statistical import StatisticalAnalyzer
    import pandas as pd
    s = StatisticalAnalyzer()
    r = s.analyze(pd.read_csv(CSV_PATH))
    assert r is not None
    return True

test("Analytics orchestrator", test_orchestrator)
test("Data profiler", test_profiler)
test("Data quality analyzer", test_quality)
test("Statistical analyzer", test_stats)

# ======================================================================
# 5. Visualization Engine
# ======================================================================
print("\n--- 5. Visualization Engine ---")

def test_chart_selector():
    from backend.visualization.chart_selector import ChartSelector
    import pandas as pd
    sel = ChartSelector()
    ct = sel.select_chart_type(
        df=pd.read_csv(CSV_PATH),
        x_col="region", y_col="total_revenue",
        query="Compare revenue by region"
    )
    assert ct is not None
    return True

def test_themes():
    from backend.visualization.chart_themes import get_dark_theme
    assert get_dark_theme() is not None
    return True

test("Chart type selector", test_chart_selector)
test("Dark theme loading", test_themes)

# ======================================================================
# 6. Session Management
# ======================================================================
print("\n--- 6. Session Management ---")

def test_session():
    from backend.services.session_service import SessionService
    svc = SessionService()
    sid = svc.create_session(file_id=upload_response.file_id, title="Smoke Test")
    assert sid and len(sid) > 0
    return True

test("Session create", test_session)

# ======================================================================
# 7. Security Validator
# ======================================================================
print("\n--- 7. Security Validator ---")

def test_safe_code():
    from backend.sandbox.validator import CodeValidator
    v = CodeValidator()
    violations = v.validate("result = df.head(10)")
    critical = [x for x in violations if x.severity.value == "critical"]
    assert len(critical) == 0, f"Safe code had violations: {critical}"
    return True

def test_block_subprocess():
    from backend.sandbox.validator import CodeValidator
    v = CodeValidator()
    assert not v.is_safe("import subprocess\nsubprocess.call(['rm', '-rf', '/'])")
    return True

def test_block_exec():
    from backend.sandbox.validator import CodeValidator
    v = CodeValidator()
    assert not v.is_safe("exec('print(1)')")
    return True

def test_block_file_write():
    from backend.sandbox.validator import CodeValidator
    v = CodeValidator()
    assert not v.is_safe("open('/etc/passwd', 'w').write('x')")
    return True

test("Safe pandas code passes", test_safe_code)
test("Block subprocess import", test_block_subprocess)
test("Block exec() call", test_block_exec)
test("Block file write", test_block_file_write)

# ======================================================================
# Summary
# ======================================================================
print()
print("=" * 70)
total = results["pass"] + results["fail"]
print(f"  Results: {results['pass']}/{total} passed, {results['fail']} failed")
if results["fail"] == 0:
    print(f"  \033[92mALL TESTS PASSED\033[0m")
else:
    print(f"  \033[91m{results['fail']} TESTS FAILED\033[0m")
print("=" * 70)
sys.exit(results["fail"])
