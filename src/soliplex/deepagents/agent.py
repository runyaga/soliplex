"""SoliplexDeepAgent wrapper for pydantic-deep agents."""

from __future__ import annotations

import contextlib
import dataclasses
import typing
from collections import abc

import pydantic_ai
from pydantic_ai import messages as ai_messages
from pydantic_ai import output as ai_output
from pydantic_ai import run as ai_run

if typing.TYPE_CHECKING:
    from soliplex.agents import AgentDependencies
    from soliplex.deepagents.config import DeepAgentConfig

# Conditional import for pydantic-deep
try:
    from pydantic_deep import DeepAgentDeps
    from pydantic_deep import LocalBackend
    from pydantic_deep import StateBackend

    PYDANTIC_DEEP_AVAILABLE = True
except ImportError:
    PYDANTIC_DEEP_AVAILABLE = False
    DeepAgentDeps = None
    LocalBackend = None
    StateBackend = None


NativeEvent = (
    ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]
)
MessageHistory = typing.Sequence[ai_messages.ModelMessage]

MISSING_PYDANTIC_DEEP_MSG = (
    "pydantic-deep is required for DeepAgent support. "
    "Install with: uv sync --group deepagents"
)


def _check_pydantic_deep_available():
    """Check if pydantic-deep is available, raise if not."""
    if not PYDANTIC_DEEP_AVAILABLE:
        raise ImportError(MISSING_PYDANTIC_DEEP_MSG)


@dataclasses.dataclass
class SoliplexDeepAgent:
    """Wrapper implementing AbstractAgent protocol for deep agents.

    This wrapper bridges Soliplex's AgentDependencies with pydantic-deep's
    DeepAgentDeps, allowing deep agents to work within the Soliplex
    framework.

    The underlying pydantic-deep agent handles:
    - Planning via todo list
    - Filesystem operations
    - Subagent delegation
    - Skills execution

    State is currently in-memory (StateBackend) and resets each run.
    """

    _deep_agent: pydantic_ai.Agent
    agent_config: DeepAgentConfig
    _deep_deps: typing.Any = None  # DeepAgentDeps instance

    # Matches AbstractAgent protocol
    output_type = None

    def __post_init__(self):
        _check_pydantic_deep_available()
        if self._deep_deps is None:
            backend = self._create_backend()
            self._deep_deps = DeepAgentDeps(backend=backend)

    def _create_backend(self):
        """Create the appropriate backend based on agent config."""
        backend_kind = getattr(self.agent_config, "backend_kind", "state")
        backend_root = getattr(self.agent_config, "backend_root", None)

        if backend_kind == "docker":
            try:
                from pydantic_ai_backends import DockerSandbox
            except ImportError as e:
                msg = "Docker backend requires: uv sync --group sandbox"
                raise ImportError(msg) from e

            docker_config = (
                getattr(self.agent_config, "docker_config", {}) or {}
            )
            return DockerSandbox(
                image=docker_config.get("image", "python:3.12-slim"),
                work_dir=docker_config.get("work_dir", "/workspace"),
                sandbox_id=self.agent_config.id,
            )
        elif backend_kind == "filesystem" and backend_root:
            return LocalBackend(root_dir=backend_root)
        return StateBackend()

    def _convert_deps(
        self,
        soliplex_deps: AgentDependencies | None,
    ) -> DeepAgentDeps:
        """Convert Soliplex AgentDependencies to DeepAgentDeps.

        Currently creates a fresh DeepAgentDeps each call. Future phases
        will sync state with AG-UI.
        """
        # In MVP, we use in-memory state that persists for the agent
        # instance lifetime but resets between server restarts
        return self._deep_deps

    async def run(
        self,
        prompt: str,
        message_history: MessageHistory | None = None,
        deps: AgentDependencies | None = None,
    ):
        """Run the agent to completion.

        Args:
            prompt: User message to process.
            message_history: Previous conversation messages.
            deps: Soliplex agent dependencies.

        Returns:
            Agent run result.
        """
        deep_deps = self._convert_deps(deps)

        return await self._deep_agent.run(
            prompt,
            message_history=message_history,
            deps=deep_deps,
        )

    @contextlib.asynccontextmanager
    async def run_stream(
        self,
        prompt: str,
        message_history: MessageHistory | None = None,
        deps: AgentDependencies | None = None,
    ):
        """Run the agent with streaming output.

        Args:
            prompt: User message to process.
            message_history: Previous conversation messages.
            deps: Soliplex agent dependencies.

        Yields:
            Streaming run context.
        """
        deep_deps = self._convert_deps(deps)

        async with self._deep_agent.run_stream(
            prompt,
            message_history=message_history,
            deps=deep_deps,
        ) as stream:
            yield stream

    async def run_stream_events(
        self,
        output_type: ai_output.OutputSpec[typing.Any] | None = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: pydantic_ai.DeferredToolResults | None = None,
        deps: AgentDependencies | None = None,
        **kwargs,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Run the agent and stream events.

        This is the primary method used by Soliplex for AG-UI streaming.

        Args:
            output_type: Optional output type specification.
            message_history: Previous conversation messages.
            deferred_tool_results: Deferred tool results to inject.
            deps: Soliplex agent dependencies.
            **kwargs: Additional arguments passed to the deep agent.

        Yields:
            Agent stream events (PartStartEvent, PartDeltaEvent, etc.)
        """
        deep_deps = self._convert_deps(deps)

        async for event in self._deep_agent.run_stream_events(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            deps=deep_deps,
            **kwargs,
        ):
            yield event

    @property
    def todos(self) -> list:
        """Access the current todo list state."""
        if self._deep_deps is not None:
            return self._deep_deps.todos
        return []

    @property
    def files(self) -> dict:
        """Access the current files cache."""
        if self._deep_deps is not None:
            return self._deep_deps.files
        return {}

    def save_state(
        self,
        output_dir: str | None = None,
        run_output: str | None = None,
        prompt: str | None = None,
    ) -> str | None:
        """Save agent state (files, todos, output) to disk for auditability.

        This is useful for Docker sandbox backends where files exist only
        inside the container. Call this after agent.run() to persist the
        code and results.

        Args:
            output_dir: Directory to save state. If None, uses backend_root
                       from agent config, or returns None if not configured.
            run_output: The agent's output text to save (contains code/results).
            prompt: The user prompt that triggered this run.

        Returns:
            Path to the output directory, or None if no state to save.
        """
        from datetime import datetime
        from pathlib import Path
        import json

        # Determine output directory
        if output_dir is None:
            output_dir = getattr(self.agent_config, "backend_root", None)
            if output_dir and not Path(output_dir).is_absolute():
                config_dir = Path(self.agent_config._config_path).parent
                output_dir = str(config_dir / output_dir)

        if not output_dir:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save files from in-memory state (filesystem/state backends)
        files = self.files
        if files:
            for filename, content in files.items():
                file_path = output_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)

        # Save todos as JSON
        todos = self.todos
        if todos:
            todos_path = output_path / "_todos.json"
            todos_path.write_text(json.dumps(todos, indent=2, default=str))

        # Save run output for auditability (especially for Docker sandbox)
        if run_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_path = output_path / f"_run_{timestamp}.md"
            content_parts = []
            if prompt:
                content_parts.append(f"# Prompt\n\n{prompt}\n")
            content_parts.append(f"# Output\n\n{run_output}\n")
            run_path.write_text("\n".join(content_parts))

        return str(output_path)

    def cleanup(self):
        """Clean up resources, especially Docker containers.

        Should be called when the agent is no longer needed to ensure
        Docker containers are properly stopped and removed.
        """
        if self._deep_deps is None:
            return
        if hasattr(self._deep_deps.backend, "stop"):
            try:
                self._deep_deps.backend.stop()
            except Exception:
                pass  # Ignore cleanup errors
