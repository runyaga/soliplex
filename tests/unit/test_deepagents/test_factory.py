"""Tests for soliplex.deepagents.factory module."""

import sys
from unittest import mock

import pytest

from soliplex import config
from soliplex.deepagents import config as deep_config

MODEL_NAME = "gpt-oss:latest"
BASE_URL = "https://example.com:12345"
SYSTEM_PROMPT = "You are a test assistant."


@pytest.fixture
def mock_pydantic_deep():
    """Mock pydantic_deep and pydantic_ai modules."""
    mock_deep_agent = mock.MagicMock()
    mock_create_deep_agent = mock.MagicMock(return_value=mock_deep_agent)
    mock_state_backend = mock.MagicMock()
    mock_deep_agent_deps = mock.MagicMock()

    mock_pydantic_deep_module = mock.MagicMock()
    mock_pydantic_deep_module.create_deep_agent = mock_create_deep_agent
    mock_pydantic_deep_module.DeepAgentDeps = mock_deep_agent_deps
    mock_pydantic_deep_module.StateBackend = mock_state_backend

    mock_ollama_provider = mock.MagicMock()
    mock_openai_provider = mock.MagicMock()
    mock_openai_chat_model = mock.MagicMock()

    with mock.patch.dict(
        sys.modules,
        {
            "pydantic_deep": mock_pydantic_deep_module,
            "pydantic_ai.providers.ollama": mock.MagicMock(
                OllamaProvider=mock_ollama_provider
            ),
            "pydantic_ai.providers.openai": mock.MagicMock(
                OpenAIProvider=mock_openai_provider
            ),
            "pydantic_ai.models.openai": mock.MagicMock(
                OpenAIChatModel=mock_openai_chat_model
            ),
        },
    ):
        yield {
            "create_deep_agent": mock_create_deep_agent,
            "deep_agent": mock_deep_agent,
            "ollama_provider": mock_ollama_provider,
            "openai_provider": mock_openai_provider,
            "openai_model": mock_openai_chat_model,
            "state_backend": mock_state_backend,
            "deep_deps": mock_deep_agent_deps,
        }


@pytest.fixture
def installation_config():
    """Create a mock installation config."""
    ic = mock.create_autospec(config.InstallationConfig)
    ic.get_environment.return_value = BASE_URL
    ic.get_secret.return_value = "test-api-key"
    return ic


@pytest.fixture
def agent_config_ollama(installation_config, tmp_path):
    """Create a DeepAgentConfig with Ollama provider."""
    return deep_config.DeepAgentConfig(
        id="test-deep-agent",
        model_name=MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        provider_type=config.LLMProviderType.OLLAMA,
        include_todo=True,
        include_filesystem=True,
        include_subagents=True,
        include_skills=False,
        _installation_config=installation_config,
        _config_path=tmp_path,
    )


@pytest.fixture
def agent_config_openai(installation_config, tmp_path):
    """Create a DeepAgentConfig with OpenAI provider."""
    return deep_config.DeepAgentConfig(
        id="test-deep-agent",
        model_name="gpt-4.1",
        system_prompt=SYSTEM_PROMPT,
        provider_type=config.LLMProviderType.OPENAI,
        provider_key="secret:API_KEY",
        include_todo=True,
        include_filesystem=True,
        include_subagents=True,
        include_skills=False,
        _installation_config=installation_config,
        _config_path=tmp_path,
    )


@pytest.fixture
def agent_config_with_subagents(installation_config, tmp_path):
    """Create a DeepAgentConfig with subagents."""
    return deep_config.DeepAgentConfig(
        id="test-deep-agent",
        model_name=MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        provider_type=config.LLMProviderType.OLLAMA,
        subagents=[
            deep_config.SubAgentConfig(
                name="reviewer",
                description="Reviews code",
                instructions="Review carefully.",
            ),
        ],
        interrupt_on={"execute": True},
        _installation_config=installation_config,
        _config_path=tmp_path,
    )


class TestCreateDeepAgentFromConfig:
    """Tests for create_deep_agent_from_config factory function."""

    def test_pydantic_deep_not_available(self, agent_config_ollama):
        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        original = agent_module.PYDANTIC_DEEP_AVAILABLE
        agent_module.PYDANTIC_DEEP_AVAILABLE = False

        try:
            with pytest.raises(ImportError, match="pydantic-deep is required"):
                factory.create_deep_agent_from_config(agent_config_ollama)
        finally:
            agent_module.PYDANTIC_DEEP_AVAILABLE = original

    def test_ollama_provider(self, mock_pydantic_deep, agent_config_ollama):
        import importlib

        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True
        importlib.reload(factory)

        result = factory.create_deep_agent_from_config(agent_config_ollama)

        # Verify the factory was called correctly
        mock_pydantic_deep["create_deep_agent"].assert_called_once()
        call_kwargs = mock_pydantic_deep["create_deep_agent"].call_args.kwargs

        expected_model = mock_pydantic_deep["openai_model"].return_value
        assert call_kwargs["model"] == expected_model
        assert call_kwargs["instructions"] == SYSTEM_PROMPT
        assert call_kwargs["include_todo"] is True
        assert call_kwargs["include_filesystem"] is True
        assert call_kwargs["include_subagents"] is True
        assert call_kwargs["include_skills"] is False
        assert call_kwargs["subagents"] is None
        assert call_kwargs["interrupt_on"] is None

        # Verify Ollama provider was used
        mock_pydantic_deep["ollama_provider"].assert_called_once()
        mock_pydantic_deep["openai_provider"].assert_not_called()

        # Verify result is a SoliplexDeepAgent
        from soliplex.deepagents.agent import SoliplexDeepAgent

        assert isinstance(result, SoliplexDeepAgent)

    def test_openai_provider(self, mock_pydantic_deep, agent_config_openai):
        import importlib

        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True
        importlib.reload(factory)

        result = factory.create_deep_agent_from_config(agent_config_openai)

        # Verify OpenAI provider was used
        mock_pydantic_deep["openai_provider"].assert_called_once()
        mock_pydantic_deep["ollama_provider"].assert_not_called()

        from soliplex.deepagents.agent import SoliplexDeepAgent

        assert isinstance(result, SoliplexDeepAgent)

    def test_with_subagents(
        self, mock_pydantic_deep, agent_config_with_subagents
    ):
        import importlib

        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True
        importlib.reload(factory)

        result = factory.create_deep_agent_from_config(
            agent_config_with_subagents
        )

        call_kwargs = mock_pydantic_deep["create_deep_agent"].call_args.kwargs

        assert call_kwargs["subagents"] == [
            {
                "name": "reviewer",
                "description": "Reviews code",
                "instructions": "Review carefully.",
            }
        ]
        assert call_kwargs["interrupt_on"] == {"execute": True}

        from soliplex.deepagents.agent import SoliplexDeepAgent

        assert isinstance(result, SoliplexDeepAgent)

    def test_with_tool_configs(self, mock_pydantic_deep, agent_config_ollama):
        import importlib

        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True
        importlib.reload(factory)

        tool_configs = {"test_tool": mock.MagicMock()}
        mcp_configs = {"test_mcp": mock.MagicMock()}

        result = factory.create_deep_agent_from_config(
            agent_config_ollama,
            tool_configs=tool_configs,
            mcp_client_toolset_configs=mcp_configs,
        )

        # For MVP, tool_configs are accepted but not used
        from soliplex.deepagents.agent import SoliplexDeepAgent

        assert isinstance(result, SoliplexDeepAgent)
