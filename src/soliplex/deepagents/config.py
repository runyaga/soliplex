"""Configuration for pydantic-deepagents based agents."""

from __future__ import annotations

import dataclasses
import pathlib
import typing

from soliplex.config import FromYamlException
from soliplex.config import InstallationConfig
from soliplex.config import LLMProviderType
from soliplex.config import _no_repr_none


@dataclasses.dataclass
class SubAgentConfig:
    """Configuration for a subagent.

    Mirrors pydantic_deep.SubAgentConfig TypedDict but as a dataclass
    for YAML parsing.
    """

    name: str
    description: str
    instructions: str
    model: str = None
    tools: list = dataclasses.field(default_factory=list)

    @classmethod
    def from_yaml(cls, config: dict) -> SubAgentConfig:
        return cls(**config)

    def to_dict(self) -> dict:
        """Convert to dict for pydantic-deep SubAgentConfig."""
        result = {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
        }
        if self.model is not None:
            result["model"] = self.model
        if self.tools:
            result["tools"] = self.tools
        return result


@dataclasses.dataclass
class DeepAgentConfig:
    """Configuration for pydantic-deepagents based agents.

    This config class enables integration with pydantic-deep's autonomous
    agent capabilities including:
    - Planning (todos)
    - Filesystem operations
    - Subagent delegation
    - Skills

    Registered via `meta.agent_configs` in installation.yaml with
    `kind: "deep"`.
    """

    kind: typing.ClassVar[str] = "deep"

    id: str
    model_name: str = None
    retries: int = 3
    model_settings: typing.Any = None  # pydantic_ai.settings.ModelSettings

    # System prompt (inline or file path starting with './')
    system_prompt: dataclasses.InitVar[str] = None
    _system_prompt_text: str = None
    _system_prompt_path: pathlib.Path = None

    # Deep agent feature toggles
    include_todo: bool = True
    include_filesystem: bool = True
    include_subagents: bool = True
    include_skills: bool = False
    include_execute: bool = None  # None = auto-determine based on backend

    # Subagent nesting configuration
    max_nesting_depth: int = 0  # 0 = subagents can't spawn sub-subagents

    # Backend configuration
    backend_kind: str = "state"  # "state", "filesystem", "docker"
    backend_root: str = None

    # Docker sandbox configuration (when backend_kind="docker")
    # Set DOCKER_HOST=ssh://hostname for remote execution
    docker_config: dict = dataclasses.field(default_factory=dict)

    # Subagent definitions
    subagents: list[SubAgentConfig] = dataclasses.field(default_factory=list)

    # Skill directories
    skill_directories: list[str] = dataclasses.field(default_factory=list)

    # Tool approval requirements
    interrupt_on: dict[str, bool] = dataclasses.field(default_factory=dict)

    # Provider settings (inherited from installation or explicit)
    provider_type: LLMProviderType = LLMProviderType.OLLAMA
    provider_base_url: str = None
    provider_key: str = None

    # Set by `from_yaml` factory
    _installation_config: InstallationConfig = _no_repr_none()
    _config_path: pathlib.Path = None

    def __post_init__(self, system_prompt):
        if self.model_name is None:
            if self._installation_config is not None:
                self.model_name = self._installation_config.get_environment(
                    "DEFAULT_AGENT_MODEL",
                )

        if system_prompt is not None:
            self._system_prompt_text = system_prompt

    @classmethod
    def from_yaml(
        cls,
        installation_config: InstallationConfig,
        config_path: pathlib.Path,
        config: dict,
    ) -> DeepAgentConfig:
        """Create a DeepAgentConfig from YAML configuration.

        Args:
            installation_config: The parent installation configuration.
            config_path: Path to the YAML file containing this config.
            config: Dictionary parsed from YAML.

        Returns:
            A configured DeepAgentConfig instance.
        """
        try:
            config["_installation_config"] = installation_config
            config["_config_path"] = config_path

            # Get installation-level deep_agents config
            deep_agents_config = (
                getattr(installation_config, "deep_agents", None) or {}
            )

            # Apply installation-level defaults for backend if not set in room
            if "backend_kind" not in config:
                config["backend_kind"] = deep_agents_config.get(
                    "default_backend_kind", "state"
                )

            # Set backend_root from installation config if not explicitly set
            if "backend_root" not in config:
                state_root = deep_agents_config.get("state_root")
                if state_root:
                    # Each room gets its own subdirectory under state_root
                    # Agent id is "room-{room_id}", strip the prefix
                    agent_id = config.get("id", "unknown")
                    if agent_id.startswith("room-"):
                        room_id = agent_id[5:]  # Strip "room-" prefix
                    else:
                        room_id = agent_id

                    # Resolve state_root relative to installation config path
                    install_dir = installation_config._config_path.parent
                    resolved_root = (install_dir / state_root).resolve()
                    config["backend_root"] = str(resolved_root / room_id)

            # Handle system_prompt as inline text or file path
            if "system_prompt" in config:
                system_prompt = config.pop("system_prompt")

                if system_prompt.startswith("./"):
                    config["_system_prompt_path"] = system_prompt
                else:
                    config["system_prompt"] = system_prompt

            # Parse subagent configurations
            if "subagents" in config:
                config["subagents"] = [
                    SubAgentConfig.from_yaml(sa)
                    for sa in config.get("subagents", [])
                ]

            # Parse skill_directories as list
            if "skill_directories" in config:
                config["skill_directories"] = list(config["skill_directories"])

            # Parse interrupt_on as dict
            if "interrupt_on" in config:
                config["interrupt_on"] = dict(config["interrupt_on"])

            # Parse docker_config as dict
            if "docker_config" in config:
                config["docker_config"] = dict(config["docker_config"])

            return cls(**config)

        except Exception as exc:
            raise FromYamlException(config_path, "deep_agent", config) from exc

    def get_system_prompt(self) -> str | None:
        """Get the system prompt text.

        Returns:
            The system prompt text, either inline or loaded from file.
        """
        if self._system_prompt_text is not None:
            return self._system_prompt_text

        if self._system_prompt_path is not None:
            if self._config_path is None:  # pragma: NO COVER
                msg = "_config_path not set, cannot resolve prompt"
                raise ValueError(msg)

            system_prompt_file = (
                self._config_path.parent / self._system_prompt_path
            )
            return system_prompt_file.read_text()

        return None

    @property
    def llm_provider_kw(self) -> dict:
        """Get keyword arguments for the LLM provider."""
        if self.provider_base_url is None:
            provider_base_url = self._installation_config.get_environment(
                "OLLAMA_BASE_URL"
            )
        else:
            provider_base_url = self.provider_base_url

        provider_kw = {
            "base_url": f"{provider_base_url}/v1",
        }

        if self.provider_key is not None:
            provider_kw["api_key"] = self._installation_config.get_secret(
                self.provider_key
            )

        return provider_kw

    @property
    def as_yaml(self) -> dict:
        """Convert config back to YAML-compatible dict."""
        prompt = (
            self._system_prompt_path
            if self._system_prompt_text is None
            else self._system_prompt_text
        )
        if self.provider_base_url is None:
            provider_base_url = self._installation_config.get_environment(
                "OLLAMA_BASE_URL"
            )
        else:
            provider_base_url = self.provider_base_url

        return {
            "id": self.id,
            "kind": self.kind,
            "model_name": self.model_name,
            "system_prompt": prompt,
            "include_todo": self.include_todo,
            "include_filesystem": self.include_filesystem,
            "include_subagents": self.include_subagents,
            "include_skills": self.include_skills,
            "include_execute": self.include_execute,
            "backend_kind": self.backend_kind,
            "backend_root": self.backend_root,
            "subagents": [sa.to_dict() for sa in self.subagents],
            "skill_directories": self.skill_directories,
            "interrupt_on": self.interrupt_on,
            "provider_type": str(self.provider_type),
            "provider_base_url": provider_base_url,
            "provider_key": self.provider_key,
        }

    def get_model_string(self) -> str:
        """Get the model string for pydantic-deep.

        Pydantic-deep expects model strings like 'openai:gpt-4.1'.
        """
        if self.provider_type == LLMProviderType.OPENAI:
            return f"openai:{self.model_name}"
        else:
            # Ollama models are accessed via OpenAI-compatible API
            return f"openai:{self.model_name}"

    def get_subagents_for_deep(self) -> list[dict]:
        """Get subagent configs as dicts for pydantic-deep."""
        return [sa.to_dict() for sa in self.subagents]
