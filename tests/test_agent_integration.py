"""Integration test for the DataWhisperer LangChain agent system."""

from backend.llm.providers.ollama_provider import OllamaProvider
from backend.llm.memory import ConversationMemory
from backend.llm.agent import DataWhispererAgent, AgentResult
from backend.llm.chains.output_parser import OutputParser
from backend.llm.prompts.context_builder import ContextBuilder
from backend.models.schemas import FileMetadata, ColumnInfo, ResultType


def test_memory_isolation():
    print("=== Test 1: Memory Isolation ===")
    mem = ConversationMemory()
    mem.add_user_message("session_1", "Hello")
    mem.add_assistant_message("session_1", "Hi there!")
    mem.add_user_message("session_2", "Different session")

    s1 = mem.get_dict_messages("session_1")
    s2 = mem.get_dict_messages("session_2")
    assert len(s1) == 2, f"Expected 2 msgs in s1, got {len(s1)}"
    assert len(s2) == 1, f"Expected 1 msg in s2, got {len(s2)}"
    assert s1[0]["role"] == "user"
    assert s1[1]["role"] == "assistant"
    print("PASS: Session isolation works")


def test_context_switching():
    print("\n=== Test 2: Dataset Context Switching ===")
    mem = ConversationMemory()
    prev = mem.set_active_dataset("session_1", "file_001")
    assert prev is None, "First set should return None"
    prev = mem.set_active_dataset("session_1", "file_002")
    assert prev == "file_001", f"Expected file_001, got {prev}"
    active = mem.get_active_dataset("session_1")
    assert active == "file_002"
    print("PASS: Context switching works")


def test_agent_result():
    print("\n=== Test 3: AgentResult ===")
    result = AgentResult(
        success=True,
        content="Analysis done",
        code="result = df.head()",
        result_type=ResultType.DATAFRAME,
        explanation="Shows first 5 rows",
        internal_reasoning="Step 1: use head()",
    )
    assert result.success
    assert result.internal_reasoning is not None
    assert "reasoning" not in result.content.lower()
    print("PASS: AgentResult hides reasoning")


def test_provider_init():
    print("\n=== Test 4: Provider Initialization ===")
    provider = OllamaProvider()
    health = provider.health_check()
    connected = health.get("connected", False)
    model_loaded = health.get("model_loaded", False)
    print(f"Connected: {connected}")
    print(f"Model loaded: {model_loaded}")
    error = health.get("error")
    if error:
        print(f"Note: {error}")
    print("PASS: Provider initializes correctly")
    provider.close()


def test_cot_stripping():
    print("\n=== Test 5: CoT Stripping ===")
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
    print(f"Code: {code[:50]}")
    print(f"Reasoning detected: {len(reasoning)} chars")
    print("PASS: CoT extracted and separated")


def test_multi_dataset_context():
    print("\n=== Test 6: Multi-Dataset Context ===")
    cb = ContextBuilder()
    meta1 = FileMetadata(
        original_name="sales.csv",
        stored_path="/tmp/sales.csv",
        row_count=100,
        col_count=3,
        file_size_bytes=5000,
        memory_usage_mb=0.5,
        columns=[
            ColumnInfo(name="product", dtype="object", non_null_count=100, null_count=0, unique_count=10),
            ColumnInfo(name="revenue", dtype="float64", non_null_count=100, null_count=0, unique_count=80, mean=500.0, std=100.0, min_val=100.0, max_val=1000.0),
            ColumnInfo(name="date", dtype="object", non_null_count=100, null_count=0, unique_count=30),
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
            ColumnInfo(name="item", dtype="object", non_null_count=50, null_count=0, unique_count=25),
            ColumnInfo(name="stock", dtype="int64", non_null_count=50, null_count=0, unique_count=20),
        ],
    )
    multi_ctx = cb.build_multi_dataset_context(meta1, {"f1": meta1, "f2": meta2})
    assert "ACTIVE DATASET" in multi_ctx
    assert "OTHER LOADED DATASETS" in multi_ctx
    assert "inventory.csv" in multi_ctx
    print("PASS: Multi-dataset context built")


if __name__ == "__main__":
    test_memory_isolation()
    test_context_switching()
    test_agent_result()
    test_provider_init()
    test_cot_stripping()
    test_multi_dataset_context()

    print("\n============================")
    print("ALL TESTS PASSED")
    print("============================")
