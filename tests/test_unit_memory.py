"""
Unit tests for ConversationMemory — the per-session LangChain memory manager.

Tests:
    - Session creation and retrieval
    - Message adding (user + assistant)
    - Sliding window enforcement
    - Dict-format message conversion
    - Database message loading
    - Active dataset tracking (multi-dataset support)
    - Session clearing
    - Session counting
    - Trimming behavior at max capacity
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.core.config import ChatSettings
from backend.llm.memory import ConversationMemory


@pytest.fixture
def memory():
    """Create memory with a small window for deterministic testing."""
    settings = ChatSettings(
        max_history_messages=10,
        history_window_size=4,
        max_query_length=2000,
        suggested_questions_count=4,
    )
    return ConversationMemory(chat_settings=settings)


class TestSessionCreation:
    """Verify session lifecycle management."""

    def test_get_or_create_new_session(self, memory):
        history = memory.get_or_create("sess-1")
        assert history is not None
        assert len(history.messages) == 0

    def test_get_or_create_returns_same_instance(self, memory):
        h1 = memory.get_or_create("sess-1")
        h2 = memory.get_or_create("sess-1")
        assert h1 is h2

    def test_different_sessions_are_isolated(self, memory):
        memory.add_user_message("sess-1", "Hello from session 1")
        memory.add_user_message("sess-2", "Hello from session 2")
        msgs1 = memory.get_langchain_messages("sess-1")
        msgs2 = memory.get_langchain_messages("sess-2")
        assert len(msgs1) == 1
        assert len(msgs2) == 1
        assert msgs1[0].content != msgs2[0].content


class TestMessageAdding:
    """Verify message insertion for both roles."""

    def test_add_user_message(self, memory):
        memory.add_user_message("s1", "What is the average revenue?")
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "What is the average revenue?"

    def test_add_assistant_message(self, memory):
        memory.add_assistant_message("s1", "The average is $500.")
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)

    def test_message_ordering_preserved(self, memory):
        memory.add_user_message("s1", "Q1")
        memory.add_assistant_message("s1", "A1")
        memory.add_user_message("s1", "Q2")
        memory.add_assistant_message("s1", "A2")
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 4
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert isinstance(msgs[2], HumanMessage)
        assert isinstance(msgs[3], AIMessage)


class TestSlidingWindow:
    """Verify window_size limits messages returned for LLM context."""

    def test_window_truncates_old_messages(self, memory):
        for i in range(10):
            memory.add_user_message("s1", f"msg-{i}")
        # Window size is 4 — should only return the last 4
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 4
        assert msgs[0].content == "msg-6"

    def test_custom_window_size_override(self, memory):
        for i in range(10):
            memory.add_user_message("s1", f"msg-{i}")
        msgs = memory.get_langchain_messages("s1", window_size=2)
        assert len(msgs) == 2

    def test_window_when_fewer_messages_than_limit(self, memory):
        memory.add_user_message("s1", "only-one")
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 1


class TestDictMessages:
    """Verify dict-format message conversion for legacy components."""

    def test_user_message_to_dict(self, memory):
        memory.add_user_message("s1", "Hello")
        dicts = memory.get_dict_messages("s1")
        assert dicts == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_to_dict(self, memory):
        memory.add_assistant_message("s1", "Hi!")
        dicts = memory.get_dict_messages("s1")
        assert dicts == [{"role": "assistant", "content": "Hi!"}]

    def test_mixed_messages_to_dict(self, memory):
        memory.add_user_message("s1", "Q")
        memory.add_assistant_message("s1", "A")
        dicts = memory.get_dict_messages("s1")
        assert len(dicts) == 2
        assert dicts[0]["role"] == "user"
        assert dicts[1]["role"] == "assistant"

    def test_dict_window_size_respected(self, memory):
        for i in range(10):
            memory.add_user_message("s1", f"msg-{i}")
        dicts = memory.get_dict_messages("s1", window_size=2)
        assert len(dicts) == 2


class TestDatabaseLoading:
    """Verify session restore from database records."""

    def test_load_empty_messages(self, memory):
        memory.load_from_db_messages("s1", [])
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 0

    def test_load_user_and_assistant_messages(self, memory):
        db_msgs = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        memory.load_from_db_messages("s1", db_msgs)
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)

    def test_load_clears_existing_messages(self, memory):
        memory.add_user_message("s1", "old-message")
        memory.load_from_db_messages("s1", [{"role": "user", "content": "new"}])
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 1
        assert msgs[0].content == "new"


class TestActiveDataset:
    """Verify multi-dataset context tracking."""

    def test_set_first_dataset_returns_none(self, memory):
        result = memory.set_active_dataset("s1", "file-A")
        assert result is None

    def test_same_dataset_returns_none(self, memory):
        memory.set_active_dataset("s1", "file-A")
        result = memory.set_active_dataset("s1", "file-A")
        assert result is None

    def test_switch_dataset_returns_previous(self, memory):
        memory.set_active_dataset("s1", "file-A")
        result = memory.set_active_dataset("s1", "file-B")
        assert result == "file-A"

    def test_get_active_dataset(self, memory):
        assert memory.get_active_dataset("s1") is None
        memory.set_active_dataset("s1", "file-X")
        assert memory.get_active_dataset("s1") == "file-X"


class TestSessionClearing:
    """Verify session cleanup."""

    def test_clear_session_removes_messages(self, memory):
        memory.add_user_message("s1", "hello")
        memory.set_active_dataset("s1", "file-1")
        memory.clear_session("s1")
        msgs = memory.get_langchain_messages("s1")
        assert len(msgs) == 0
        assert memory.get_active_dataset("s1") is None

    def test_clear_nonexistent_session_is_safe(self, memory):
        memory.clear_session("nonexistent")  # Should not raise

    def test_session_count(self, memory):
        assert memory.get_session_count() == 0
        memory.add_user_message("s1", "hi")
        memory.add_user_message("s2", "hi")
        assert memory.get_session_count() == 2
        memory.clear_session("s1")
        assert memory.get_session_count() == 1


class TestTrimming:
    """Verify automatic trimming at max_history_messages."""

    def test_trim_at_max_capacity(self, memory):
        """Insert more than max_history_messages and verify trimming."""
        # max_history_messages=10
        for i in range(15):
            memory.add_user_message("s1", f"msg-{i}")
        history = memory.get_or_create("s1")
        assert len(history.messages) <= 10
