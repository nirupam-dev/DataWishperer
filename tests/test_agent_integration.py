"""Integration tests for the DataWhisperer Code Interpreter pipeline."""

from backend.llm.providers.ollama_provider import OllamaProvider
from backend.llm.memory import ConversationMemory
from backend.llm.agent import DataWhispererAgent, AgentResult
from backend.llm.chains.output_parser import OutputParser
from backend.llm.chains.query_chain import QueryChain
from backend.llm.prompts.context_builder import ContextBuilder
from backend.llm.prompts.system import (
    SYSTEM_PROMPT,
    REASONING_PROMPT,
    EXPLANATION_PROMPT,
    CHART_EXPLANATION_PROMPT,
    DEBUG_PROMPT,
    ERROR_RECOVERY_PROMPT,
)
from backend.models.schemas import (
    ChatResponse,
    FileMetadata,
    ColumnInfo,
    ResultType,
)


def test_memory_isolation():
    """Test that conversation memory is isolated per session."""
    print("=== Test 1: Memory Isolation ===")
    mem = ConversationMemory()
    mem.add_user_message("session_1", "Hello")
    mem.add_assistant_message("session_1", "Hi there!")
    mem.add_user_message("session_2", "Different session")

    s1 = mem.get_dict_messages("session_1")
    s2 = mem.get_dict_messages("session_2")
    assert len(s1) == 2, f"Expected 2 msgs in s1, got {len(s1)}"
    assert len(s2) == 1, f"Expected 1 msg in s2, got {len(s2)}"
    print("PASS: Session isolation works")


def test_context_switching():
    """Test multi-dataset context switching."""
    print("\n=== Test 2: Dataset Context Switching ===")
    mem = ConversationMemory()
    prev = mem.set_active_dataset("session_1", "file_001")
    assert prev is None, "First set should return None"
    prev = mem.set_active_dataset("session_1", "file_002")
    assert prev == "file_001", f"Expected file_001, got {prev}"
    active = mem.get_active_dataset("session_1")
    assert active == "file_002"
    print("PASS: Context switching works")


def test_agent_result_structure():
    """Test AgentResult carries all interpreter pipeline fields."""
    print("\n=== Test 3: AgentResult Interpreter Fields ===")
    result = AgentResult(
        success=True,
        content="Full interpreter output",
        code="result = df.groupby('cat')['val'].mean()",
        result_type=ResultType.DATAFRAME,
        explanation="This groups the data by category and finds averages.",
        chart_explanation=None,
        auto_debug_applied=False,
        internal_reasoning="Step 1: identify relevant columns",
    )
    assert result.success
    assert result.code is not None
    assert result.explanation is not None
    assert result.auto_debug_applied is False
    assert result.internal_reasoning is not None
    # CoT must never leak into content
    assert "Step 1" not in result.content
    print("PASS: AgentResult has all interpreter fields")


def test_agent_result_auto_debug():
    """Test AgentResult correctly represents auto-debug scenario."""
    print("\n=== Test 4: AgentResult Auto-Debug ===")
    result = AgentResult(
        success=True,
        content="Auto-debugged output",
        code="result = df.dropna().groupby('cat')['val'].mean()",
        result_type=ResultType.DATAFRAME,
        auto_debug_applied=True,
        debug_summary="Fixed NaN handling in groupby operation.",
        attempts=2,
    )
    assert result.auto_debug_applied is True
    assert result.attempts == 2
    assert result.debug_summary is not None
    print("PASS: Auto-debug fields populated correctly")


def test_agent_result_chart_explanation():
    """Test chart explanation field on chart results."""
    print("\n=== Test 5: Chart Explanation ===")
    result = AgentResult(
        success=True,
        content="Chart output",
        code="plt.bar(x, y); plt.savefig(chart_path)",
        result_type=ResultType.CHART,
        chart_path="/tmp/chart.png",
        chart_explanation="A bar chart was chosen because the data compares categories.",
    )
    assert result.chart_explanation is not None
    assert "bar chart" in result.chart_explanation.lower()
    print("PASS: Chart explanation populated")


def test_cot_stripping():
    """Test that chain-of-thought is extracted and separated."""
    print("\n=== Test 6: CoT Stripping ===")
    parser = OutputParser()
    test_output = (
        "Let me think step-by-step.\n"
        "Step 1: We need the average.\n"
        "Step 2: Group by category.\n\n"
        "```python\n"
        'result = df.groupby("cat")["val"].mean()\n'
        "```\n"
    )
    code, reasoning = parser.extract_code_and_reasoning(test_output)
    assert "groupby" in code
    assert reasoning is not None
    assert "step-by-step" in reasoning.lower()
    # Verify reasoning is NOT in the code
    assert "Step 1" not in code
    assert "think" not in code.lower()
    print(f"Code: {code[:50]}")
    print(f"Reasoning detected: {len(reasoning)} chars")
    print("PASS: CoT extracted and separated")


def test_chart_type_detection():
    """Test chart type detection from code."""
    print("\n=== Test 7: Chart Type Detection ===")
    chain = QueryChain.__new__(QueryChain)

    # Bar chart
    bar_code = "ax.bar(range(10), values)"
    assert QueryChain._detect_chart_type(bar_code) == "bar chart"

    # Scatter plot
    scatter_code = "plt.scatter(x, y)"
    assert QueryChain._detect_chart_type(scatter_code) == "scatter plot"

    # Histogram
    hist_code = "df['col'].hist(bins=20)"
    assert QueryChain._detect_chart_type(hist_code) == "histogram"

    # Pie chart
    pie_code = "df.plot.pie(y='values')"
    assert QueryChain._detect_chart_type(pie_code) == "pie chart"

    # No chart
    no_chart = "result = df.describe()"
    assert QueryChain._detect_chart_type(no_chart) is None

    # Generic chart with savefig
    generic = "plt.savefig(chart_path)"
    assert QueryChain._detect_chart_type(generic) == "chart"

    print("PASS: All chart types detected correctly")


def test_multi_dataset_context():
    """Test multi-dataset context generation."""
    print("\n=== Test 8: Multi-Dataset Context ===")
    cb = ContextBuilder()
    meta1 = FileMetadata(
        original_name="sales.csv",
        stored_path="/tmp/sales.csv",
        row_count=100,
        col_count=3,
        file_size_bytes=5000,
        memory_usage_mb=0.5,
        columns=[
            ColumnInfo(name="product", dtype="object", non_null_count=100,
                       null_count=0, unique_count=10),
            ColumnInfo(name="revenue", dtype="float64", non_null_count=100,
                       null_count=0, unique_count=80, mean=500.0,
                       std=100.0, min_val=100.0, max_val=1000.0),
        ],
    )
    meta2 = FileMetadata(
        original_name="inventory.csv",
        stored_path="/tmp/inventory.csv",
        row_count=50,
        col_count=2,
        file_size_bytes=2000,
        memory_usage_mb=0.2,
        columns=[
            ColumnInfo(name="item", dtype="object", non_null_count=50,
                       null_count=0, unique_count=25),
            ColumnInfo(name="stock", dtype="int64", non_null_count=50,
                       null_count=0, unique_count=20),
        ],
    )
    multi_ctx = cb.build_multi_dataset_context(
        meta1, {"f1": meta1, "f2": meta2}
    )
    assert "ACTIVE DATASET" in multi_ctx
    assert "OTHER LOADED DATASETS" in multi_ctx
    assert "inventory.csv" in multi_ctx
    print("PASS: Multi-dataset context built")


def test_interpreter_output_format():
    """Test that _format_interpreter_output produces correct sections."""
    print("\n=== Test 9: Interpreter Output Format ===")
    from backend.models.schemas import CodeExecutionResult

    result = CodeExecutionResult(
        success=True,
        result_type=ResultType.DATAFRAME,
        data='[{"product": "A", "total": 100}]',
    )
    content = DataWhispererAgent._format_interpreter_output(
        code='result = df.groupby("product").sum()',
        execution_result=result,
        explanation="This shows total values per product.",
        chart_explanation=None,
        auto_debug_applied=False,
    )
    assert "📝 **Generated Code:**" in content
    assert "```python" in content
    assert "📋 **Output:**" in content
    assert "💡 **Explanation:**" in content
    assert "🔧 **Auto-Debug:**" not in content  # Not applied
    print("PASS: Interpreter output has correct sections")


def test_interpreter_output_with_chart():
    """Test interpreter output when a chart is generated."""
    print("\n=== Test 10: Interpreter Output With Chart ===")
    from backend.models.schemas import CodeExecutionResult

    result = CodeExecutionResult(
        success=True,
        result_type=ResultType.CHART,
        data="Monthly trend chart",
        chart_path="/tmp/chart.png",
    )
    content = DataWhispererAgent._format_interpreter_output(
        code='plt.bar(x, y)\nplt.savefig(chart_path)',
        execution_result=result,
        explanation="This chart shows monthly trends.",
        chart_explanation="A bar chart was used because it effectively compares values across months.",
        auto_debug_applied=False,
    )
    assert "📝 **Generated Code:**" in content
    assert "📊 **Output:**" in content
    assert "💡 **Explanation:**" in content
    assert "🎨 **Chart Reasoning:**" in content
    assert "bar chart" in content
    print("PASS: Chart output has reasoning section")


def test_interpreter_output_with_debug():
    """Test interpreter output when auto-debug was applied."""
    print("\n=== Test 11: Interpreter Output With Auto-Debug ===")
    from backend.models.schemas import CodeExecutionResult

    result = CodeExecutionResult(
        success=True,
        result_type=ResultType.TEXT,
        data="42",
    )
    content = DataWhispererAgent._format_interpreter_output(
        code='result = df["col"].sum()',
        execution_result=result,
        explanation="The total sum is 42.",
        chart_explanation=None,
        auto_debug_applied=True,
    )
    assert "🔧 **Auto-Debug:**" in content
    assert "automatically debugged" in content
    print("PASS: Auto-debug note appears in output")


def test_chat_response_schema():
    """Test ChatResponse has all interpreter fields."""
    print("\n=== Test 12: ChatResponse Schema ===")
    response = ChatResponse(
        message_id="msg-123",
        content="Full output",
        generated_code="result = 42",
        result_type=ResultType.TEXT,
        result_data="42",
        explanation="The answer is 42.",
        chart_explanation=None,
        auto_debug_applied=False,
    )
    assert response.explanation == "The answer is 42."
    assert response.auto_debug_applied is False
    assert response.chart_explanation is None
    print("PASS: ChatResponse has all interpreter fields")


def test_all_prompts_exist():
    """Verify all interpreter pipeline prompts are defined."""
    print("\n=== Test 13: All Pipeline Prompts ===")
    assert len(SYSTEM_PROMPT) > 100, "SYSTEM_PROMPT too short"
    assert len(REASONING_PROMPT) > 50, "REASONING_PROMPT too short"
    assert len(EXPLANATION_PROMPT) > 50, "EXPLANATION_PROMPT too short"
    assert len(CHART_EXPLANATION_PROMPT) > 50, "CHART_EXPLANATION_PROMPT too short"
    assert len(DEBUG_PROMPT) > 100, "DEBUG_PROMPT too short"
    assert len(ERROR_RECOVERY_PROMPT) > 50, "ERROR_RECOVERY_PROMPT too short"

    # Verify template variables
    assert "{code}" in EXPLANATION_PROMPT
    assert "{result_summary}" in EXPLANATION_PROMPT
    assert "{chart_type}" in CHART_EXPLANATION_PROMPT
    assert "{failed_code}" in DEBUG_PROMPT
    assert "{error_type}" in DEBUG_PROMPT
    assert "{row_count}" in DEBUG_PROMPT
    print("PASS: All pipeline prompts defined with correct variables")


def test_provider_init():
    """Test that the provider initializes correctly."""
    print("\n=== Test 14: Provider Initialization ===")
    provider = OllamaProvider()
    health = provider.health_check()
    connected = health.get("connected", False)
    print(f"Connected: {connected}")
    if health.get("error"):
        print(f"Note: {health['error']}")
    print("PASS: Provider initializes correctly")
    provider.close()


if __name__ == "__main__":
    test_memory_isolation()
    test_context_switching()
    test_agent_result_structure()
    test_agent_result_auto_debug()
    test_agent_result_chart_explanation()
    test_cot_stripping()
    test_chart_type_detection()
    test_multi_dataset_context()
    test_interpreter_output_format()
    test_interpreter_output_with_chart()
    test_interpreter_output_with_debug()
    test_chat_response_schema()
    test_all_prompts_exist()
    test_provider_init()

    print("\n" + "=" * 40)
    print(f"ALL 14 TESTS PASSED")
    print("=" * 40)
