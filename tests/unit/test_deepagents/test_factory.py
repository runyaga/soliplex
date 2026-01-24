"""Tests for soliplex.deepagents.factory module."""

from unittest import mock

import pytest

from soliplex import config
from soliplex.deepagents import config as deep_config

MODEL_NAME = "gpt-oss:latest"
BASE_URL = "https://example.com:12345"
SYSTEM_PROMPT = "You are a test assistant."


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
            with pytest.raises(
                ImportError, match="pydantic-deep is required"
            ):
                factory.create_deep_agent_from_config(agent_config_ollama)
        finally:
            agent_module.PYDANTIC_DEEP_AVAILABLE = original

    def test_ollama_provider(self, agent_config_ollama):
        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True

        mock_deep_agent = mock.MagicMock()
        mock_ollama_provider = mock.MagicMock()
        mock_openai_provider = mock.MagicMock()
        mock_openai_model = mock.MagicMock()

        with (
            mock.patch(
                "soliplex.deepagents._pydantic_deep_patch.create_deep_agent",
                return_value=mock_deep_agent,
            ) as mock_create,
            mock.patch(
                "pydantic_ai.providers.ollama.OllamaProvider",
                mock_ollama_provider,
            ),
            mock.patch(
                "pydantic_ai.providers.openai.OpenAIProvider",
                mock_openai_provider,
            ),
            mock.patch(
                "pydantic_ai.models.openai.OpenAIChatModel",
                mock_openai_model,
            ),
        ):
            result = factory.create_deep_agent_from_config(agent_config_ollama)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs

            assert call_kwargs["model"] == mock_openai_model.return_value
            assert call_kwargs["instructions"] == SYSTEM_PROMPT
            assert call_kwargs["include_todo"] is True
            assert call_kwargs["include_filesystem"] is True
            assert call_kwargs["include_subagents"] is True
            assert call_kwargs["include_skills"] is False
            assert call_kwargs["subagents"] is None
            assert call_kwargs["interrupt_on"] is None

            mock_ollama_provider.assert_called_once()
            mock_openai_provider.assert_not_called()

            assert type(result).__name__ == "SoliplexDeepAgent"

    def test_openai_provider(self, agent_config_openai):
        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True

        mock_deep_agent = mock.MagicMock()
        mock_ollama_provider = mock.MagicMock()
        mock_openai_provider = mock.MagicMock()
        mock_openai_model = mock.MagicMock()

        with (
            mock.patch(
                "soliplex.deepagents._pydantic_deep_patch.create_deep_agent",
                return_value=mock_deep_agent,
            ),
            mock.patch(
                "pydantic_ai.providers.ollama.OllamaProvider",
                mock_ollama_provider,
            ),
            mock.patch(
                "pydantic_ai.providers.openai.OpenAIProvider",
                mock_openai_provider,
            ),
            mock.patch(
                "pydantic_ai.models.openai.OpenAIChatModel",
                mock_openai_model,
            ),
        ):
            result = factory.create_deep_agent_from_config(agent_config_openai)

            mock_openai_provider.assert_called_once()
            mock_ollama_provider.assert_not_called()

            assert type(result).__name__ == "SoliplexDeepAgent"

    def test_with_subagents(self, agent_config_with_subagents):
        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True

        mock_deep_agent = mock.MagicMock()
        mock_ollama_provider = mock.MagicMock()
        mock_openai_model = mock.MagicMock()

        with (
            mock.patch(
                "soliplex.deepagents._pydantic_deep_patch.create_deep_agent",
                return_value=mock_deep_agent,
            ) as mock_create,
            mock.patch(
                "pydantic_ai.providers.ollama.OllamaProvider",
                mock_ollama_provider,
            ),
            mock.patch(
                "pydantic_ai.models.openai.OpenAIChatModel",
                mock_openai_model,
            ),
        ):
            result = factory.create_deep_agent_from_config(
                agent_config_with_subagents
            )

            call_kwargs = mock_create.call_args.kwargs

            assert call_kwargs["subagents"] == [
                {
                    "name": "reviewer",
                    "description": "Reviews code",
                    "instructions": "Review carefully.",
                }
            ]
            assert call_kwargs["interrupt_on"] == {"execute": True}

            assert type(result).__name__ == "SoliplexDeepAgent"

    def test_with_tool_configs(self, agent_config_ollama):
        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True

        mock_deep_agent = mock.MagicMock()
        mock_ollama_provider = mock.MagicMock()
        mock_openai_model = mock.MagicMock()

        with (
            mock.patch(
                "soliplex.deepagents._pydantic_deep_patch.create_deep_agent",
                return_value=mock_deep_agent,
            ),
            mock.patch(
                "pydantic_ai.providers.ollama.OllamaProvider",
                mock_ollama_provider,
            ),
            mock.patch(
                "pydantic_ai.models.openai.OpenAIChatModel",
                mock_openai_model,
            ),
        ):
            tool_configs = {"test_tool": mock.MagicMock()}
            mcp_configs = {"test_mcp": mock.MagicMock()}

            result = factory.create_deep_agent_from_config(
                agent_config_ollama,
                tool_configs=tool_configs,
                mcp_client_toolset_configs=mcp_configs,
            )

            # For MVP, tool_configs are accepted but not used
            assert type(result).__name__ == "SoliplexDeepAgent"

    def test_max_nesting_depth_passed(self, installation_config, tmp_path):
        """Test that max_nesting_depth is passed to create_deep_agent."""
        from soliplex.deepagents import agent as agent_module
        from soliplex.deepagents import factory

        agent_module.PYDANTIC_DEEP_AVAILABLE = True

        config_with_nesting = deep_config.DeepAgentConfig(
            id="test-deep-agent",
            model_name=MODEL_NAME,
            system_prompt=SYSTEM_PROMPT,
            provider_type=config.LLMProviderType.OLLAMA,
            max_nesting_depth=2,
            _installation_config=installation_config,
            _config_path=tmp_path,
        )

        mock_deep_agent = mock.MagicMock()
        mock_ollama_provider = mock.MagicMock()
        mock_openai_model = mock.MagicMock()

        with (
            mock.patch(
                "soliplex.deepagents._pydantic_deep_patch.create_deep_agent",
                return_value=mock_deep_agent,
            ) as mock_create,
            mock.patch(
                "pydantic_ai.providers.ollama.OllamaProvider",
                mock_ollama_provider,
            ),
            mock.patch(
                "pydantic_ai.models.openai.OpenAIChatModel",
                mock_openai_model,
            ),
        ):
            factory.create_deep_agent_from_config(config_with_nesting)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_nesting_depth"] == 2
