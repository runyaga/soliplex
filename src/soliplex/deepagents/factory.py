"""Factory function for creating deep agents from configuration."""

from __future__ import annotations

import typing

from soliplex import config
from soliplex.deepagents.agent import SoliplexDeepAgent
from soliplex.deepagents.agent import _check_pydantic_deep_available
from soliplex.deepagents.config import DeepAgentConfig

if typing.TYPE_CHECKING:
    from soliplex.agents import ToolConfigMap


MCP_ToolsetConfigMap = config.MCP_ClientToolsetConfigMap


def _create_backend(agent_config: DeepAgentConfig):
    """Create the appropriate backend based on configuration.

    Args:
        agent_config: Deep agent configuration.

    Returns:
        Backend instance (StateBackend, FilesystemBackend, or DockerSandbox).
    """
    from pydantic_ai_backends import StateBackend

    backend_kind = agent_config.backend_kind

    if backend_kind == "docker":
        from pydantic_ai_backends import DockerSandbox

        # DockerSandbox uses docker.from_env() which respects DOCKER_HOST
        # Set DOCKER_HOST=ssh://hostname for remote Docker execution
        docker_config = agent_config.docker_config or {}
        return DockerSandbox(
            image=docker_config.get("image", "python:3.12-slim"),
            work_dir=docker_config.get("work_dir", "/workspace"),
            sandbox_id=agent_config.id,
        )

    elif backend_kind == "filesystem":
        from pydantic_ai_backends import FilesystemBackend

        if agent_config.backend_root:
            return FilesystemBackend(root=agent_config.backend_root)
        return FilesystemBackend()

    else:  # "state" or default
        return StateBackend()


def create_deep_agent_from_config(
    agent_config: DeepAgentConfig,
    tool_configs: ToolConfigMap | None = None,
    mcp_client_toolset_configs: MCP_ToolsetConfigMap | None = None,
) -> SoliplexDeepAgent:
    """Factory function called by agents.get_agent_from_configs().

    Creates a SoliplexDeepAgent wrapper around a pydantic-deep agent
    configured according to the DeepAgentConfig.

    Args:
        agent_config: Deep agent configuration from YAML.
        tool_configs: Additional Soliplex tool configurations.
        mcp_client_toolset_configs: MCP client toolset configurations.

    Returns:
        A SoliplexDeepAgent instance wrapping the pydantic-deep agent.

    Raises:
        ImportError: If pydantic-deep is not installed.
    """
    _check_pydantic_deep_available()

    # Import here to avoid import errors when pydantic-deep not installed
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    # Use patched create_deep_agent with max_nesting_depth support
    from soliplex.deepagents._pydantic_deep_patch import create_deep_agent

    # Create the model with provider configuration
    provider_kw = agent_config.llm_provider_kw

    if agent_config.provider_type == config.LLMProviderType.OLLAMA:
        provider_kw["api_key"] = provider_kw.get("api_key", "dummy")
        provider = OllamaProvider(**provider_kw)
    else:
        provider = OpenAIProvider(**provider_kw)

    model = OpenAIChatModel(
        model_name=agent_config.model_name,
        provider=provider,
    )

    # Get subagent configurations
    subagents = agent_config.get_subagents_for_deep() or None

    # Get skill directories if configured
    skill_directories = None
    if agent_config.include_skills and agent_config.skill_directories:
        from pydantic_deep import SkillDirectory

        skill_directories = [
            SkillDirectory(path) for path in agent_config.skill_directories
        ]

    # Create backend based on configuration
    backend = _create_backend(agent_config)

    # Create the deep agent
    deep_agent = create_deep_agent(
        model=model,
        instructions=agent_config.get_system_prompt(),
        backend=backend,
        include_todo=agent_config.include_todo,
        include_filesystem=agent_config.include_filesystem,
        include_subagents=agent_config.include_subagents,
        include_skills=agent_config.include_skills,
        include_execute=agent_config.include_execute,
        subagents=subagents,
        skill_directories=skill_directories,
        interrupt_on=agent_config.interrupt_on or None,
        max_nesting_depth=agent_config.max_nesting_depth,
    )

    return SoliplexDeepAgent(
        _deep_agent=deep_agent,
        agent_config=agent_config,
    )
