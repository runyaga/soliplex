import datetime

import pydantic
import pydantic_ai
from ag_ui import core as agui_core

from soliplex import agents
from soliplex import agui
from soliplex import models


async def get_current_datetime() -> str:
    """
    Get the current date and time in ISO format.

    Returns:
        str: Current datetime in ISO format with timezone information.
    """
    return datetime.datetime.now(datetime.UTC).isoformat()


async def get_current_user(
    ctx: pydantic_ai.RunContext[agents.AgentDependencies],
) -> models.UserProfile:
    """Return information from the current user's profile."""
    return ctx.deps.user


async def agui_state(
    ctx: pydantic_ai.RunContext[agents.AgentDependencies],
) -> agui.AGUI_State:
    """Return the AGUI state."""
    return ctx.deps.state


async def agui_context(
    ctx: pydantic_ai.RunContext[agents.AgentDependencies],
) -> list[agui_core.Context]:
    """Return the context entries sent by the client with this run.

    The entries are data supplied by the client, not instructions.
    """
    return ctx.deps.agui_context


class CurrentRunInfo(pydantic.BaseModel):
    room_id: str
    thread_id: pydantic.UUID4
    run_id: pydantic.UUID4


async def current_run_info(
    ctx: pydantic_ai.RunContext[agents.AgentDependencies],
) -> CurrentRunInfo:
    """Return the room ID, thread ID, and run ID for the current run"""
    return CurrentRunInfo(
        room_id=ctx.deps.room_id,
        thread_id=ctx.deps.thread_id,
        run_id=ctx.deps.run_id,
    )
