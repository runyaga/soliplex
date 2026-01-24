"""Tests for soliplex.deepagents.config module."""

from unittest import mock

import pytest

from soliplex import config
from soliplex.deepagents import config as deep_config

ROOM_ID = "test-deep-room"
MODEL_NAME = "gpt-oss:latest"
SYSTEM_PROMPT = "You are a helpful assistant."
BASE_URL = "https://example.com:12345"


@pytest.fixture
def installation_config():
    """Create a mock installation config."""
    ic = mock.create_autospec(config.InstallationConfig)
    ic.get_environment.return_value = BASE_URL
    ic.get_secret.return_value = "test-api-key"
    return ic


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory with a prompt file.

    Returns the path to a mock config file (room_config.yaml), not
    just the directory. This is because _config_path.parent is used
    to resolve relative system_prompt paths.
    """
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(SYSTEM_PROMPT)
    # Return path to a mock config file in the directory
    return tmp_path / "room_config.yaml"


class TestSubAgentConfig:
    """Tests for SubAgentConfig dataclass."""

    def test_basic_creation(self):
        sa = deep_config.SubAgentConfig(
            name="test-subagent",
            description="A test subagent",
            instructions="Do your best.",
        )
        assert sa.name == "test-subagent"
        assert sa.description == "A test subagent"
        assert sa.instructions == "Do your best."
        assert sa.model is None
        assert sa.tools == []

    def test_creation_with_model(self):
        sa = deep_config.SubAgentConfig(
            name="test-subagent",
            description="A test subagent",
            instructions="Do your best.",
            model="openai:gpt-4.1",
        )
        assert sa.model == "openai:gpt-4.1"

    def test_from_yaml(self):
        yaml_dict = {
            "name": "reviewer",
            "description": "Reviews code",
            "instructions": "Review carefully.",
            "model": "openai:gpt-4",
        }
        sa = deep_config.SubAgentConfig.from_yaml(yaml_dict)
        assert sa.name == "reviewer"
        assert sa.description == "Reviews code"
        assert sa.instructions == "Review carefully."
        assert sa.model == "openai:gpt-4"

    def test_to_dict_minimal(self):
        sa = deep_config.SubAgentConfig(
            name="test",
            description="desc",
            instructions="instr",
        )
        result = sa.to_dict()
        assert result == {
            "name": "test",
            "description": "desc",
            "instructions": "instr",
        }
        assert "model" not in result
        assert "tools" not in result

    def test_to_dict_full(self):
        sa = deep_config.SubAgentConfig(
            name="test",
            description="desc",
            instructions="instr",
            model="openai:gpt-4",
            tools=["tool1"],
        )
        result = sa.to_dict()
        assert result == {
            "name": "test",
            "description": "desc",
            "instructions": "instr",
            "model": "openai:gpt-4",
            "tools": ["tool1"],
        }


class TestDeepAgentConfig:
    """Tests for DeepAgentConfig dataclass."""

    def test_kind_is_deep(self):
        assert deep_config.DeepAgentConfig.kind == "deep"

    def test_basic_creation(self, installation_config, temp_dir):
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            system_prompt=SYSTEM_PROMPT,
            _installation_config=installation_config,
            _config_path=temp_dir,
        )
        assert dac.id == ROOM_ID
        assert dac.model_name == MODEL_NAME
        assert dac.get_system_prompt() == SYSTEM_PROMPT
        assert dac.include_todo is True
        assert dac.include_filesystem is True
        assert dac.include_subagents is True
        assert dac.include_skills is False
        assert dac.backend_kind == "state"
        assert dac.subagents == []
        assert dac.skill_directories == []
        assert dac.interrupt_on == {}

    def test_default_model_from_installation(self, temp_dir):
        ic = mock.create_autospec(config.InstallationConfig)
        ic.get_environment.return_value = "default-model"

        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            _installation_config=ic,
            _config_path=temp_dir,
        )

        assert dac.model_name == "default-model"
        ic.get_environment.assert_called_with("DEFAULT_AGENT_MODEL")

    def test_from_yaml_inline_prompt(self, installation_config, temp_dir):
        yaml_config = {
            "id": ROOM_ID,
            "model_name": MODEL_NAME,
            "system_prompt": SYSTEM_PROMPT,
            "include_todo": False,
            "include_filesystem": False,
        }

        dac = deep_config.DeepAgentConfig.from_yaml(
            installation_config,
            temp_dir,
            yaml_config,
        )

        assert dac.id == ROOM_ID
        assert dac.model_name == MODEL_NAME
        assert dac.get_system_prompt() == SYSTEM_PROMPT
        assert dac.include_todo is False
        assert dac.include_filesystem is False

    def test_from_yaml_file_prompt(self, installation_config, temp_dir):
        yaml_config = {
            "id": ROOM_ID,
            "model_name": MODEL_NAME,
            "system_prompt": "./prompt.txt",
        }

        dac = deep_config.DeepAgentConfig.from_yaml(
            installation_config,
            temp_dir,
            yaml_config,
        )

        assert dac.get_system_prompt() == SYSTEM_PROMPT

    def test_from_yaml_with_subagents(self, installation_config, temp_dir):
        yaml_config = {
            "id": ROOM_ID,
            "model_name": MODEL_NAME,
            "subagents": [
                {
                    "name": "reviewer",
                    "description": "Reviews code",
                    "instructions": "Review carefully.",
                },
                {
                    "name": "tester",
                    "description": "Writes tests",
                    "instructions": "Write tests.",
                },
            ],
        }

        dac = deep_config.DeepAgentConfig.from_yaml(
            installation_config,
            temp_dir,
            yaml_config,
        )

        assert len(dac.subagents) == 2
        assert dac.subagents[0].name == "reviewer"
        assert dac.subagents[1].name == "tester"

    def test_from_yaml_with_skill_directories(
        self, installation_config, temp_dir
    ):
        yaml_config = {
            "id": ROOM_ID,
            "model_name": MODEL_NAME,
            "skill_directories": [
                "/path/to/skills",
                "~/.pydantic-deep/skills",
            ],
        }

        dac = deep_config.DeepAgentConfig.from_yaml(
            installation_config,
            temp_dir,
            yaml_config,
        )

        assert dac.skill_directories == [
            "/path/to/skills",
            "~/.pydantic-deep/skills",
        ]

    def test_from_yaml_with_interrupt_on(self, installation_config, temp_dir):
        yaml_config = {
            "id": ROOM_ID,
            "model_name": MODEL_NAME,
            "interrupt_on": {"execute": True, "write": False},
        }

        dac = deep_config.DeepAgentConfig.from_yaml(
            installation_config,
            temp_dir,
            yaml_config,
        )

        assert dac.interrupt_on == {"execute": True, "write": False}

    def test_from_yaml_error_handling(self, installation_config, temp_dir):
        yaml_config = {
            # Missing required 'id' field
            "model_name": MODEL_NAME,
        }

        with pytest.raises(config.FromYamlException):
            deep_config.DeepAgentConfig.from_yaml(
                installation_config,
                temp_dir,
                yaml_config,
            )

    def test_get_system_prompt_none(self, installation_config, temp_dir):
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            _installation_config=installation_config,
            _config_path=temp_dir,
        )
        assert dac.get_system_prompt() is None

    def test_llm_provider_kw_ollama(self, temp_dir):
        ic = mock.create_autospec(config.InstallationConfig)
        ic.get_environment.return_value = BASE_URL

        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            provider_type=config.LLMProviderType.OLLAMA,
            _installation_config=ic,
            _config_path=temp_dir,
        )

        kw = dac.llm_provider_kw
        assert kw["base_url"] == f"{BASE_URL}/v1"
        assert "api_key" not in kw

    def test_llm_provider_kw_openai(self, temp_dir):
        ic = mock.create_autospec(config.InstallationConfig)
        ic.get_environment.return_value = BASE_URL
        ic.get_secret.return_value = "test-key"

        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            provider_type=config.LLMProviderType.OPENAI,
            provider_key="secret:API_KEY",
            _installation_config=ic,
            _config_path=temp_dir,
        )

        kw = dac.llm_provider_kw
        assert kw["api_key"] == "test-key"

    def test_llm_provider_kw_explicit_base_url(self, temp_dir):
        ic = mock.create_autospec(config.InstallationConfig)

        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            provider_base_url="http://custom:8080",
            _installation_config=ic,
            _config_path=temp_dir,
        )

        kw = dac.llm_provider_kw
        assert kw["base_url"] == "http://custom:8080/v1"
        ic.get_environment.assert_not_called()

    def test_as_yaml(self, installation_config, temp_dir):
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            system_prompt=SYSTEM_PROMPT,
            include_todo=True,
            include_filesystem=False,
            subagents=[
                deep_config.SubAgentConfig(
                    name="test",
                    description="desc",
                    instructions="instr",
                )
            ],
            _installation_config=installation_config,
            _config_path=temp_dir,
        )

        yaml_dict = dac.as_yaml

        assert yaml_dict["id"] == ROOM_ID
        assert yaml_dict["kind"] == "deep"
        assert yaml_dict["model_name"] == MODEL_NAME
        assert yaml_dict["system_prompt"] == SYSTEM_PROMPT
        assert yaml_dict["include_todo"] is True
        assert yaml_dict["include_filesystem"] is False
        assert len(yaml_dict["subagents"]) == 1
        assert yaml_dict["subagents"][0]["name"] == "test"

    def test_get_model_string_openai(self, installation_config, temp_dir):
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name="gpt-4.1",
            provider_type=config.LLMProviderType.OPENAI,
            _installation_config=installation_config,
            _config_path=temp_dir,
        )

        assert dac.get_model_string() == "openai:gpt-4.1"

    def test_get_model_string_ollama(self, installation_config, temp_dir):
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name="gpt-oss:latest",
            provider_type=config.LLMProviderType.OLLAMA,
            _installation_config=installation_config,
            _config_path=temp_dir,
        )

        # Ollama is OpenAI-compatible, so returns same format
        assert dac.get_model_string() == "openai:gpt-oss:latest"

    def test_get_subagents_for_deep_empty(self, installation_config, temp_dir):
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            _installation_config=installation_config,
            _config_path=temp_dir,
        )

        assert dac.get_subagents_for_deep() == []

    def test_get_subagents_for_deep_with_subagents(
        self, installation_config, temp_dir
    ):
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            subagents=[
                deep_config.SubAgentConfig(
                    name="test",
                    description="desc",
                    instructions="instr",
                    model="openai:gpt-4",
                )
            ],
            _installation_config=installation_config,
            _config_path=temp_dir,
        )

        result = dac.get_subagents_for_deep()
        assert len(result) == 1
        assert result[0] == {
            "name": "test",
            "description": "desc",
            "instructions": "instr",
            "model": "openai:gpt-4",
        }

    def test_no_model_with_no_installation_config(self, temp_dir):
        """Test __post_init__ when model_name is None and no ic."""
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            # model_name is None, _installation_config is None
            _config_path=temp_dir,
        )
        # Should not raise, model_name remains None
        assert dac.model_name is None

    def test_as_yaml_with_explicit_provider_base_url(
        self, installation_config, temp_dir
    ):
        """Test as_yaml branch when provider_base_url is not None."""
        dac = deep_config.DeepAgentConfig(
            id=ROOM_ID,
            model_name=MODEL_NAME,
            provider_base_url="http://custom:8080",
            _installation_config=installation_config,
            _config_path=temp_dir,
        )

        yaml_dict = dac.as_yaml

        # Should use the explicit provider_base_url
        assert yaml_dict["provider_base_url"] == "http://custom:8080"
