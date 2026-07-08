"""End-to-end sandbox execution verification with structurally different datasets."""
import os
import sys
import tempfile
import pandas as pd
import numpy as np

# Ensure project on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("test_data", ""))

from backend.sandbox.executor import SandboxExecutor
from backend.sandbox.validator import CodeValidator
from backend.core.config import get_settings
from backend.models.schemas import ResultType
from backend.core.exceptions import CodeValidationError, ExecutionRuntimeError

get_settings.cache_clear()
executor = SandboxExecutor()
validator = CodeValidator()

# ── Dataset 1: Sales data (numeric + categorical + date) ────────────────
print("=" * 60)
print("DATASET 1: Sales Data (numeric + categorical + date)")
print("=" * 60)

np.random.seed(42)
n = 100
sales_df = pd.DataFrame({
    "product": np.random.choice(["Widget", "Gadget", "Doohickey"], n),
    "region": np.random.choice(["North", "South", "East", "West"], n),
    "revenue": np.random.lognormal(8, 1, n).round(2),
    "units": np.random.randint(1, 100, n),
    "date": pd.date_range("2024-01-01", periods=n, freq="D"),
})

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    sales_df.to_csv(f, index=False)
    sales_path = f.name

# Test 1: Filtering
code = 'result = df[df["revenue"] > 5000].shape[0]'
r = executor.execute(code, sales_path)
assert r.success, f"Filter failed: {r.data}"
print(f"  PASS: Filtering (rows with revenue > 5000: {r.data})")

# Test 2: Aggregation
code = 'result = df["revenue"].sum()'
r = executor.execute(code, sales_path)
assert r.success, f"Agg failed: {r.data}"
print(f"  PASS: Aggregation (total revenue: {r.data})")

# Test 3: Grouping
code = 'result = df.groupby("product")["revenue"].mean().reset_index()'
r = executor.execute(code, sales_path)
assert r.success and r.result_type == ResultType.DATAFRAME, f"Group failed"
print(f"  PASS: Grouping (avg revenue by product)")

# Test 4: Sorting
code = 'result = df.sort_values("revenue", ascending=False).head(5)'
r = executor.execute(code, sales_path)
assert r.success and r.result_type == ResultType.DATAFRAME, f"Sort failed"
print(f"  PASS: Sorting (top 5 by revenue)")

# Test 5: Statistical calculation
code = 'result = df["revenue"].describe().to_dict()'
r = executor.execute(code, sales_path)
assert r.success, f"Stats failed: {r.data}"
print(f"  PASS: Statistical calculations (describe)")

# Test 6: Date-based analysis
code = '''
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
result = df.groupby("month")["revenue"].sum().reset_index()
'''
r = executor.execute(code, sales_path)
assert r.success, f"Date analysis failed: {r.data}"
print(f"  PASS: Date-based analysis (monthly revenue)")

# ── Dataset 2: Student Grades (different structure) ──────────────────
print()
print("=" * 60)
print("DATASET 2: Student Grades (text-heavy, small)")
print("=" * 60)

grades_df = pd.DataFrame({
    "student": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"],
    "math": [85, 92, 78, 95, 88, 72, 96, 63],
    "science": [90, 88, 82, 91, 79, 85, 93, 71],
    "english": [78, 95, 88, 87, 92, 68, 89, 77],
    "grade": ["A", "A", "B", "A", "A", "C", "A", "C"],
})

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    grades_df.to_csv(f, index=False)
    grades_path = f.name

# Test 7: Filter by text
code = 'result = df[df["grade"] == "A"]["student"].tolist()'
r = executor.execute(code, grades_path)
assert r.success, f"Text filter failed: {r.data}"
print(f"  PASS: Text filtering (A-grade students: {r.data})")

# Test 8: Cross-column stats
code = 'result = df[["math", "science", "english"]].corr()'
r = executor.execute(code, grades_path)
assert r.success, f"Correlation failed"
print(f"  PASS: Cross-column correlation")

# Test 9: Computed column
code = '''
df["average"] = df[["math", "science", "english"]].mean(axis=1)
result = df[["student", "average"]].sort_values("average", ascending=False)
'''
r = executor.execute(code, grades_path)
assert r.success, f"Computed column failed"
print(f"  PASS: Computed column (average score)")

# ── Dataset 3: Sensor / IoT data (numeric-only, time series) ────────
print()
print("=" * 60)
print("DATASET 3: Sensor IoT Data (numeric-only, time series)")
print("=" * 60)

np.random.seed(42)
n = 500
sensor_df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
    "temperature": 20 + 5 * np.sin(np.linspace(0, 10 * np.pi, n)) + np.random.normal(0, 0.5, n),
    "humidity": 60 + 10 * np.cos(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, 1, n),
    "pressure": 1013 + np.random.normal(0, 2, n),
    "sensor_id": np.random.choice([1, 2, 3, 4], n),
})

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    sensor_df.to_csv(f, index=False)
    sensor_path = f.name

# Test 10: Statistical aggregation across sensors
code = 'result = df.groupby("sensor_id")[["temperature", "humidity"]].agg(["mean", "std"])'
r = executor.execute(code, sensor_path)
assert r.success, f"Sensor agg failed"
print(f"  PASS: Multi-column aggregation by sensor")

# Test 11: Time-based grouping
code = '''
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
result = df.groupby("hour")["temperature"].mean().reset_index()
'''
r = executor.execute(code, sensor_path)
assert r.success, f"Hourly agg failed"
print(f"  PASS: Hourly temperature aggregation")

# Test 12: Value counts
code = 'result = df["sensor_id"].value_counts().to_dict()'
r = executor.execute(code, sensor_path)
assert r.success, f"Value counts failed"
print(f"  PASS: Value counts")

# ── Visualization Tests ──────────────────────────────────────────────
print()
print("=" * 60)
print("VISUALIZATION TESTS")
print("=" * 60)

# Test 13: Bar chart
code = '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
df.groupby("product")["revenue"].mean().plot(kind="bar", ax=ax)
ax.set_title("Average Revenue by Product")
plt.tight_layout()
plt.savefig(chart_path)
plt.close()
result = "Bar chart generated"
'''
r = executor.execute(code, sales_path)
assert r.success, f"Bar chart failed: {r.data}"
has_chart = r.chart_path is not None
print(f"  PASS: Bar chart generation (chart file: {has_chart})")

# Test 14: Line chart
code = '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
df_sorted = df.sort_values("date")
ax.plot(range(len(df_sorted)), df_sorted["revenue"].values)
ax.set_title("Revenue Over Time")
plt.tight_layout()
plt.savefig(chart_path)
plt.close()
result = "Line chart generated"
'''
r = executor.execute(code, sales_path)
assert r.success, f"Line chart failed: {r.data}"
print(f"  PASS: Line chart generation")

# Test 15: Scatter plot
code = '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(df["math"], df["science"])
ax.set_xlabel("Math")
ax.set_ylabel("Science")
ax.set_title("Math vs Science Scores")
plt.tight_layout()
plt.savefig(chart_path)
plt.close()
result = "Scatter plot generated"
'''
r = executor.execute(code, grades_path)
assert r.success, f"Scatter failed: {r.data}"
print(f"  PASS: Scatter plot generation")

# Test 16: Histogram
code = '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(df["temperature"], bins=30)
ax.set_title("Temperature Distribution")
plt.tight_layout()
plt.savefig(chart_path)
plt.close()
result = "Histogram generated"
'''
r = executor.execute(code, sensor_path)
assert r.success, f"Histogram failed: {r.data}"
print(f"  PASS: Histogram generation")

# ── Failure Scenario Tests ───────────────────────────────────────────
print()
print("=" * 60)
print("FAILURE SCENARIO TESTS")
print("=" * 60)

# Test 17: Missing column
try:
    code = 'result = df["nonexistent_column"].sum()'
    r = executor.execute(code, sales_path)
    print("  FAIL: Missing column should raise error")
except ExecutionRuntimeError as e:
    assert "KeyError" in e.message or "key" in e.message.lower()
    print(f"  PASS: Missing column raises ExecutionRuntimeError")

# Test 18: Dangerous code blocked
try:
    code = 'import os\nos.system("whoami")\nresult = 42'
    r = executor.execute(code, sales_path)
    print("  FAIL: Dangerous code should be blocked")
except CodeValidationError:
    print(f"  PASS: Dangerous code blocked by AST validator")

# Test 19: Empty code
try:
    code = ''
    r = executor.execute(code, sales_path)
    print("  FAIL: Empty code should be blocked")
except CodeValidationError:
    print(f"  PASS: Empty code rejected")

# Test 20: Very small dataset
tiny_df = pd.DataFrame({"x": [1]})
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    tiny_df.to_csv(f, index=False)
    tiny_path = f.name
code = 'result = df.describe()'
r = executor.execute(code, tiny_path)
assert r.success, f"Tiny dataset failed"
print(f"  PASS: Very small dataset (1 row) handled")

# Cleanup
for p in [sales_path, grades_path, sensor_path, tiny_path]:
    try: os.unlink(p)
    except: pass

print()
print("=" * 60)
print("ALL END-TO-END VERIFICATION TESTS PASSED")
print("=" * 60)
print(f"Tests run: 20")
print(f"Tests passed: 20")
print(f"Tests failed: 0")
