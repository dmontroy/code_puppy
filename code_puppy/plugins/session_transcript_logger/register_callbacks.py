"""Register structured session transcript logging callbacks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from code_puppy.callbacks import register_callback
from code_puppy.messaging.bus import get_session_context

from . import logger


async def _on_user_prompt_submit(prompt: str, session_id: str | None = None) -> None:
    logger.log_user_prompt(prompt, session_id)


async def _on_agent_run_start(
    agent_name: str,
    model_name: str,
    session_id: str | None = None,
) -> None:
    logger.log_run_start(agent_name, model_name, session_id)


async def _on_stream_event(
    event_type: str, event_data: Any, agent_session_id: Optional[str] = None
) -> None:
    logger.log_stream_event(event_type, event_data, agent_session_id)


async def _on_pre_tool_call(
    tool_name: str, tool_args: Dict[str, Any], context: Any = None
) -> None:
    logger.log_tool_start(tool_name, tool_args, get_session_context())


async def _on_post_tool_call(
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
    duration_ms: float,
    context: Any = None,
) -> None:
    logger.log_tool_end(
        tool_name,
        tool_args,
        result,
        duration_ms,
        get_session_context(),
    )


async def _on_agent_run_end(
    agent_name: str,
    model_name: str,
    session_id: str | None = None,
    success: bool = True,
    error: Exception | None = None,
    response_text: str | None = None,
    metadata: dict | None = None,
) -> None:
    logger.log_run_end(
        agent_name,
        model_name,
        session_id,
        success,
        error,
        response_text,
        metadata,
    )


def _on_session_end() -> None:
    logger.reset_state()


register_callback("user_prompt_submit", _on_user_prompt_submit)
register_callback("agent_run_start", _on_agent_run_start)
register_callback("stream_event", _on_stream_event)
register_callback("pre_tool_call", _on_pre_tool_call)
register_callback("post_tool_call", _on_post_tool_call)
register_callback("agent_run_end", _on_agent_run_end)
register_callback("session_end", _on_session_end)
