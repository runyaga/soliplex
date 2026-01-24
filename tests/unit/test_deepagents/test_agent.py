"""Tests for soliplex.deepagents.agent module."""

import sys
from unittest import mock

import pytest

from soliplex.deepagents import config as deep_config


# Mock pydantic_deep module before importing agent
@pytest.fixture(autouse=True)
def mock_pydantic_deep():
    """Mock pydantic_deep module for all tests."""
    mock_backend = mock.MagicMock()
    mock_deps = mock.MagicMock()
    mock_deps.todos = []
    mock_deps.files = {}

    mock_state_backend = mock.MagicMock(return_value=mock_backend)
    mock_deep_agent_deps = mock.MagicMock(return_value=mock_deps)

    mock_module = mock.MagicMock()
    mock_module.DeepAgentDeps = mock_deep_agent_deps
    mock_module.StateBackend = mock_state_backend

    with mock.patch.dict(sys.modules, {"pydantic_deep": mock_module}):
        # Need to reimport to pick up mocked module
        import importlib

        from soliplex.deepagents import agent as agent_module

        importlib.reload(agent_module)

        yield mock_module, mock_deps


class TestCheckPydanticDeepAvailable:
    """Tests for _check_pydantic_deep_available function."""

    def test_available(self, mock_pydantic_deep):
        from soliplex.deepagents import agent

        # Should not raise when available
        agent._check_pydantic_deep_available()

    def test_not_available(self):
        from soliplex.deepagents import agent

        original = agent.PYDANTIC_DEEP_AVAILABLE
        agent.PYDANTIC_DEEP_AVAILABLE = False

        try:
            with pytest.raises(ImportError, match="pydantic-deep is required"):
                agent._check_pydantic_deep_available()
        finally:
            agent.PYDANTIC_DEEP_AVAILABLE = original


class TestSoliplexDeepAgent:
    """Tests for SoliplexDeepAgent wrapper class."""

    @pytest.fixture
    def agent_config(self):
        """Create a mock deep agent config."""
        config = mock.create_autospec(deep_config.DeepAgentConfig)
        config.id = "test-deep-agent"
        return config

    @pytest.fixture
    def deep_agent(self, mock_pydantic_deep, agent_config):
        """Create a SoliplexDeepAgent instance."""
        from soliplex.deepagents.agent import SoliplexDeepAgent

        mock_inner_agent = mock.MagicMock()

        return SoliplexDeepAgent(
            _deep_agent=mock_inner_agent,
            agent_config=agent_config,
        )

    def test_output_type_is_none(self, deep_agent):
        assert deep_agent.output_type is None

    def test_todos_property(self, deep_agent, mock_pydantic_deep):
        mock_module, mock_deps = mock_pydantic_deep
        mock_deps.todos = ["todo1", "todo2"]
        deep_agent._deep_deps = mock_deps

        assert deep_agent.todos == ["todo1", "todo2"]

    def test_todos_property_no_deps(self, deep_agent):
        deep_agent._deep_deps = None
        assert deep_agent.todos == []

    def test_files_property(self, deep_agent, mock_pydantic_deep):
        mock_module, mock_deps = mock_pydantic_deep
        mock_deps.files = {"/test.py": {"content": "print('hi')"}}
        deep_agent._deep_deps = mock_deps

        assert deep_agent.files == {"/test.py": {"content": "print('hi')"}}

    def test_files_property_no_deps(self, deep_agent):
        deep_agent._deep_deps = None
        assert deep_agent.files == {}

    def test_convert_deps(self, deep_agent, mock_pydantic_deep):
        mock_module, mock_deps = mock_pydantic_deep
        deep_agent._deep_deps = mock_deps

        soliplex_deps = mock.MagicMock()
        result = deep_agent._convert_deps(soliplex_deps)

        # Should return the stored deps
        assert result is mock_deps

    @pytest.mark.asyncio
    async def test_run(self, deep_agent, mock_pydantic_deep):
        mock_module, mock_deps = mock_pydantic_deep
        deep_agent._deep_deps = mock_deps

        expected_result = mock.MagicMock()
        deep_agent._deep_agent.run = mock.AsyncMock(
            return_value=expected_result
        )

        soliplex_deps = mock.MagicMock()
        result = await deep_agent.run(
            "test prompt",
            message_history=["msg1"],
            deps=soliplex_deps,
        )

        assert result is expected_result
        deep_agent._deep_agent.run.assert_called_once_with(
            "test prompt",
            message_history=["msg1"],
            deps=mock_deps,
        )

    @pytest.mark.asyncio
    async def test_run_stream(self, deep_agent, mock_pydantic_deep):
        mock_module, mock_deps = mock_pydantic_deep
        deep_agent._deep_deps = mock_deps

        expected_stream = mock.MagicMock()
        context_manager = mock.AsyncMock()
        context_manager.__aenter__.return_value = expected_stream
        context_manager.__aexit__.return_value = None
        deep_agent._deep_agent.run_stream = mock.MagicMock(
            return_value=context_manager
        )

        soliplex_deps = mock.MagicMock()

        async with deep_agent.run_stream(
            "test prompt",
            message_history=["msg1"],
            deps=soliplex_deps,
        ) as stream:
            assert stream is expected_stream

        deep_agent._deep_agent.run_stream.assert_called_once_with(
            "test prompt",
            message_history=["msg1"],
            deps=mock_deps,
        )

    @pytest.mark.asyncio
    async def test_run_stream_events(self, deep_agent, mock_pydantic_deep):
        mock_module, mock_deps = mock_pydantic_deep
        deep_agent._deep_deps = mock_deps

        events = [mock.MagicMock(), mock.MagicMock()]

        async def mock_events(*args, **kwargs):
            for event in events:
                yield event

        deep_agent._deep_agent.run_stream_events = mock_events

        soliplex_deps = mock.MagicMock()
        collected = []

        async for event in deep_agent.run_stream_events(
            output_type=None,
            message_history=["msg1"],
            deps=soliplex_deps,
        ):
            collected.append(event)

        assert collected == events
