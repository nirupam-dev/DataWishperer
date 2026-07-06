"""
Agent factory — Dependency injection container for the DataWhisperer agent.

Provides a single-call factory that wires up all dependencies:
    OllamaProvider → QueryChain → Agent → ChatService

This is the composition root of the application. All dependencies
are assembled here and injected downward. No service creates its
own dependencies — they all receive them from the factory.

Usage::

    from backend.llm.factory import create_agent, create_chat_service

    # Create the agent (singleton per app lifecycle)
    agent = create_agent()

    # Create the chat service with the agent
    chat_service = create_chat_service(agent)
"""

from __future__ import annotations

from typing import Optional

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.llm.agent import DataWhispererAgent
from backend.llm.chains.output_parser import OutputParser
from backend.llm.chains.query_chain import QueryChain
from backend.llm.memory import ConversationMemory
from backend.llm.prompts.context_builder import ContextBuilder
from backend.llm.providers.ollama_provider import OllamaProvider
from backend.sandbox.executor import SandboxExecutor
from backend.services.chat_service import ChatService
from backend.storage.repositories.message_repo import MessageRepository
from backend.storage.repositories.session_repo import SessionRepository

logger = get_logger(__name__)


def create_provider() -> OllamaProvider:
    """
    Create and return an OllamaProvider instance.

    Returns:
        Configured ``OllamaProvider`` using application settings.
    """
    settings = get_settings()
    provider = OllamaProvider(settings=settings.ollama)
    logger.info("Created OllamaProvider: model=%s", settings.ollama.model)
    return provider


def create_agent(
    provider: Optional[OllamaProvider] = None,
) -> DataWhispererAgent:
    """
    Create and return a fully-wired DataWhispererAgent.

    All dependencies are assembled here via dependency injection.
    The agent is the primary intelligence component of the application.

    Args:
        provider: Optional pre-created provider. If None, creates a new one.

    Returns:
        A fully configured ``DataWhispererAgent``.
    """
    settings = get_settings()

    # 1. LLM Provider
    llm_provider = provider or create_provider()

    # 2. Shared Memory
    memory = ConversationMemory(chat_settings=settings.chat)

    # 3. Chain Components
    output_parser = OutputParser()
    context_builder = ContextBuilder()

    # 4. Query Chain
    query_chain = QueryChain(
        provider=llm_provider,
        output_parser=output_parser,
        context_builder=context_builder,
        memory=memory,
        chat_settings=settings.chat,
    )

    # 5. Sandbox Executor
    sandbox = SandboxExecutor(
        sandbox_settings=settings.sandbox,
        storage_settings=settings.storage,
    )

    # 6. Agent
    agent = DataWhispererAgent(
        provider=llm_provider,
        query_chain=query_chain,
        sandbox=sandbox,
        memory=memory,
        chat_settings=settings.chat,
        sandbox_settings=settings.sandbox,
    )

    logger.info(
        "Created DataWhispererAgent: model=%s, ctx=%d, retries=%d",
        settings.ollama.model,
        settings.ollama.num_ctx,
        settings.sandbox.max_retries,
    )

    return agent


def create_chat_service(
    agent: Optional[DataWhispererAgent] = None,
) -> ChatService:
    """
    Create a ChatService with all dependencies wired.

    Args:
        agent: Optional pre-created agent. If None, creates a new one.

    Returns:
        A fully configured ``ChatService``.
    """
    settings = get_settings()
    actual_agent = agent or create_agent()

    service = ChatService(
        agent=actual_agent,
        session_repo=SessionRepository(),
        message_repo=MessageRepository(),
        chat_settings=settings.chat,
    )

    logger.info("Created ChatService with DataWhispererAgent")
    return service
