"""pydantic-deepagents integration for Soliplex.

This package provides autonomous AI agent capabilities including:
- Planning (todos)
- Filesystem operations
- Subagent delegation
- Skills

Usage requires the 'deepagents' dependency group:
    uv sync --group deepagents
"""

from soliplex.deepagents.agent import SoliplexDeepAgent
from soliplex.deepagents.config import DeepAgentConfig
from soliplex.deepagents.factory import create_deep_agent_from_config

__all__ = [
    "DeepAgentConfig",
    "SoliplexDeepAgent",
    "create_deep_agent_from_config",
]
