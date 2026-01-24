"""Patched create_deep_agent with max_nesting_depth support.

This file reimplements create_deep_agent from pydantic-deep to expose
the max_nesting_depth parameter which allows subagents to spawn
sub-subagents.

All other types/classes are imported from the pip-installed pydantic-deep.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import overload

from pydantic_ai import Agent
from pydantic_ai._agent_graph import HistoryProcessor
from pydantic_ai.models import Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.tools import Tool
from pydantic_ai_backends import BackendProtocol
from pydantic_ai_backends import SandboxProtocol
from pydantic_ai_backends import StateBackend
from pydantic_ai_backends import create_console_toolset
from pydantic_ai_backends import get_console_system_prompt
from pydantic_ai_todo import create_todo_toolset
from pydantic_ai_todo import get_todo_system_prompt
from pydantic_deep.deps import DeepAgentDeps
from pydantic_deep.toolsets.skills import create_skills_toolset
from pydantic_deep.toolsets.skills import discover_skills
from pydantic_deep.toolsets.skills import get_skills_system_prompt
from pydantic_deep.types import Skill
from pydantic_deep.types import SkillDirectory
from pydantic_deep.types import SubAgentConfig
from subagents_pydantic_ai import create_subagent_toolset
from subagents_pydantic_ai import get_subagent_system_prompt

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

OutputDataT = TypeVar("OutputDataT")

DEFAULT_MODEL = "openai:gpt-4.1"

DEFAULT_INSTRUCTIONS = """
You are a helpful AI assistant with access to planning, filesystem, \
subagent, and skills tools.

## Capabilities
- **Planning**: Use the todo list to break down complex tasks and track \
progress
- **Filesystem**: Read, write, and search files
- **Subagents**: Delegate specialized tasks to subagents
- **Skills**: Load and use modular skill packages for specialized tasks

## Best Practices
1. Plan before acting - use the todo list for complex tasks
2. Read files before editing them
3. Mark tasks as in_progress when starting, completed when done
4. Delegate specialized work to appropriate subagents
5. Check available skills for specialized tasks - load skill instructions \
when needed
6. Be thorough but efficient
"""


@overload
def create_deep_agent(
    model: str | Model | None = None,
    instructions: str | None = None,
    tools: Sequence[Tool[DeepAgentDeps] | Any] | None = None,
    toolsets: Sequence[AbstractToolset[DeepAgentDeps]] | None = None,
    subagents: list[SubAgentConfig] | None = None,
    skills: list[Skill] | None = None,
    skill_directories: list[SkillDirectory] | None = None,
    backend: BackendProtocol | None = None,
    include_todo: bool = True,
    include_filesystem: bool = True,
    include_subagents: bool = True,
    include_skills: bool = True,
    include_general_purpose_subagent: bool = True,
    include_execute: bool | None = None,
    interrupt_on: dict[str, bool] | None = None,
    output_type: None = None,
    history_processors: Sequence[HistoryProcessor[DeepAgentDeps]] | None = (
        None
    ),
    max_nesting_depth: int = 0,
    **agent_kwargs: Any,
) -> Agent[DeepAgentDeps, str]: ...


@overload
def create_deep_agent(
    model: str | Model | None = None,
    instructions: str | None = None,
    tools: Sequence[Tool[DeepAgentDeps] | Any] | None = None,
    toolsets: Sequence[AbstractToolset[DeepAgentDeps]] | None = None,
    subagents: list[SubAgentConfig] | None = None,
    skills: list[Skill] | None = None,
    skill_directories: list[SkillDirectory] | None = None,
    backend: BackendProtocol | None = None,
    include_todo: bool = True,
    include_filesystem: bool = True,
    include_subagents: bool = True,
    include_skills: bool = True,
    include_general_purpose_subagent: bool = True,
    include_execute: bool | None = None,
    interrupt_on: dict[str, bool] | None = None,
    *,
    output_type: OutputSpec[OutputDataT],
    history_processors: Sequence[HistoryProcessor[DeepAgentDeps]] | None = (
        None
    ),
    max_nesting_depth: int = 0,
    **agent_kwargs: Any,
) -> Agent[DeepAgentDeps, OutputDataT]: ...


def create_deep_agent(  # noqa: C901
    model: str | Model | None = None,
    instructions: str | None = None,
    tools: Sequence[Tool[DeepAgentDeps] | Any] | None = None,
    toolsets: Sequence[AbstractToolset[DeepAgentDeps]] | None = None,
    subagents: list[SubAgentConfig] | None = None,
    skills: list[Skill] | None = None,
    skill_directories: list[SkillDirectory] | None = None,
    backend: BackendProtocol | None = None,
    include_todo: bool = True,
    include_filesystem: bool = True,
    include_subagents: bool = True,
    include_skills: bool = True,
    include_general_purpose_subagent: bool = True,
    include_execute: bool | None = None,
    interrupt_on: dict[str, bool] | None = None,
    output_type: OutputSpec[OutputDataT] | None = None,
    history_processors: Sequence[HistoryProcessor[DeepAgentDeps]] | None = (
        None
    ),
    max_nesting_depth: int = 0,
    **agent_kwargs: Any,
) -> Agent[DeepAgentDeps, OutputDataT] | Agent[DeepAgentDeps, str]:
    """Create deep agent with max_nesting_depth support.

    This is a patched version of pydantic_deep.create_deep_agent that
    exposes the max_nesting_depth parameter for subagent nesting control.

    Args:
        model: Model to use (default: openai:gpt-4.1).
        instructions: Custom instructions for the agent.
        tools: Additional tools to register.
        toolsets: Additional toolsets to register.
        subagents: Subagent configurations for the task tool.
        skills: Pre-loaded skills to make available.
        skill_directories: Directories to discover skills from.
        backend: File storage backend (default: StateBackend).
        include_todo: Whether to include the todo toolset.
        include_filesystem: Whether to include the filesystem toolset.
        include_subagents: Whether to include the subagent toolset.
        include_skills: Whether to include the skills toolset.
        include_general_purpose_subagent: Include general-purpose subagent.
        include_execute: Whether to include execute tool.
        interrupt_on: Map of tool names to approval requirements.
        output_type: Structured output type.
        history_processors: History processors for summarization etc.
        max_nesting_depth: Maximum depth for nested subagents. 0 means
            subagents cannot spawn their own subagents.
        **agent_kwargs: Additional arguments passed to Agent constructor.

    Returns:
        Configured Agent instance.
    """
    model = model or DEFAULT_MODEL
    backend = backend or StateBackend()
    interrupt_on = interrupt_on or {}

    all_toolsets: list[AbstractToolset[DeepAgentDeps]] = []

    if include_todo:
        todo_toolset = create_todo_toolset(id="deep-todo")
        all_toolsets.append(todo_toolset)

    if include_filesystem:
        require_write_approval = interrupt_on.get(
            "write_file", False
        ) or interrupt_on.get("edit_file", False)
        require_execute_approval = interrupt_on.get("execute", True)

        should_include_execute = (
            include_execute
            if include_execute is not None
            else isinstance(backend, SandboxProtocol)
        )

        console_toolset = create_console_toolset(
            id="deep-console",
            include_execute=should_include_execute,
            require_write_approval=require_write_approval,
            require_execute_approval=require_execute_approval,
        )
        all_toolsets.append(console_toolset)

    if include_subagents:
        subagent_model = model if isinstance(model, str) else DEFAULT_MODEL

        def subagent_toolsets_factory(
            deps: DeepAgentDeps,
        ) -> list[Any]:  # pragma: no cover
            """Provide console and todo toolsets for subagents."""
            return [
                create_console_toolset(
                    include_execute=True,
                    require_write_approval=False,
                    require_execute_approval=False,
                ),
                create_todo_toolset(),
            ]

        subagent_toolset = create_subagent_toolset(
            id="deep-subagents",
            subagents=subagents,
            default_model=subagent_model,
            include_general_purpose=include_general_purpose_subagent,
            toolsets_factory=subagent_toolsets_factory,
            max_nesting_depth=max_nesting_depth,
        )
        all_toolsets.append(subagent_toolset)

    loaded_skills: list[Skill] = []
    if include_skills:
        skills_toolset = create_skills_toolset(
            id="deep-skills",
            directories=skill_directories,
            skills=skills,
        )
        all_toolsets.append(skills_toolset)
        if skills:
            loaded_skills = skills
        elif skill_directories:
            loaded_skills = discover_skills(skill_directories)

    if toolsets:
        all_toolsets.extend(toolsets)

    base_instructions = instructions or DEFAULT_INSTRUCTIONS

    agent_create_kwargs: dict[str, Any] = {
        "deps_type": DeepAgentDeps,
        "toolsets": all_toolsets,
        "instructions": base_instructions,
    }

    has_interrupt_tools = any(interrupt_on.values())

    if output_type is not None:
        if has_interrupt_tools:
            agent_create_kwargs["output_type"] = [
                output_type,
                DeferredToolRequests,
            ]
        else:
            agent_create_kwargs["output_type"] = output_type
    elif has_interrupt_tools:
        agent_create_kwargs["output_type"] = [str, DeferredToolRequests]

    if history_processors is not None:
        agent_create_kwargs["history_processors"] = list(history_processors)

    agent_create_kwargs.update(agent_kwargs)

    agent: Agent[DeepAgentDeps, Any] = Agent(
        model,
        **agent_create_kwargs,
    )

    @agent.instructions
    def dynamic_instructions(ctx: Any) -> str:  # pragma: no cover
        """Generate dynamic instructions based on current state."""
        parts = []

        uploads_prompt = ctx.deps.get_uploads_summary()
        if uploads_prompt:
            parts.append(uploads_prompt)

        if include_todo:
            todo_prompt = get_todo_system_prompt(ctx.deps)
            if todo_prompt:
                parts.append(todo_prompt)

        if include_filesystem:
            console_prompt = get_console_system_prompt()
            if console_prompt:
                parts.append(console_prompt)

        if include_subagents:
            prompt_configs: list[SubAgentConfig] = list(subagents or [])
            if include_general_purpose_subagent:
                from subagents_pydantic_ai import (
                    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
                )

                prompt_configs.append(
                    SubAgentConfig(
                        name="general-purpose",
                        description=DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
                        instructions="",
                    )
                )
            if prompt_configs:
                subagent_prompt = get_subagent_system_prompt(prompt_configs)
                if subagent_prompt:
                    parts.append(subagent_prompt)

        if include_skills and loaded_skills:
            skills_prompt = get_skills_system_prompt(ctx.deps, loaded_skills)
            if skills_prompt:
                parts.append(skills_prompt)

        return "\n\n".join(parts) if parts else ""

    if tools:
        for tool in tools:
            if isinstance(tool, Tool):
                agent.tool(tool.function)  # pragma: no cover
            else:
                agent.tool(tool)

    return agent
