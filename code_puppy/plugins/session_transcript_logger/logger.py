"""Write structured per-session transcript logs as JSONL."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from code_puppy.config import STATE_DIR
from code_puppy.tools.common import get_working_directory

logger = logging.getLogger(__name__)

_TRANSCRIPT_DIRNAME = "session_transcripts"
_MAX_TEXT_PREVIEW = 4000

_sequence_lock = threading.Lock()
_session_sequences: dict[str, int] = {}
_session_project_names: dict[str, str] = {}


def reset_state() -> None:
    """Reset in-memory state used to sequence log records."""
    with _sequence_lock:
        _session_sequences.clear()
        _session_project_names.clear()


def get_transcript_logs_dir() -> Path:
    """Return the directory containing structured session transcript logs."""
    path = Path(STATE_DIR) / "logs" / _TRANSCRIPT_DIRNAME
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def get_transcript_log_path(session_id: str) -> Path:
    """Return the JSONL log path for a session."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in session_id)
    return get_transcript_logs_dir() / f"{safe}.jsonl"


def log_user_prompt(prompt: str, session_id: Optional[str]) -> None:
    if not session_id or not prompt:
        return
    _append_event(
        session_id,
        "user_prompt",
        {
            "role": "user",
            "text": prompt,
        },
    )


def log_run_start(agent_name: str, model_name: str, session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _sequence_lock:
        _session_project_names[session_id] = os.path.basename(get_working_directory())
    _append_event(
        session_id,
        "run_start",
        {
            "agent_name": agent_name,
            "model_name": model_name,
            "project_name": _get_project_name(session_id),
        },
    )


def _get_project_name(session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    with _sequence_lock:
        return _session_project_names.get(session_id)


def log_stream_event(
    event_type: str, event_data: Any, agent_session_id: Optional[str]
) -> None:
    if not agent_session_id:
        return

    normalized = _normalize_stream_event(event_type, event_data)
    _append_event(agent_session_id, "stream_event", normalized)


def log_tool_start(
    tool_name: str, tool_args: Dict[str, Any], session_id: Optional[str]
) -> None:
    if not session_id:
        return
    _append_event(
        session_id,
        "tool_start",
        {
            "tool_name": tool_name,
            "tool_args": _sanitize_value(tool_args),
            "project_name": _get_project_name(session_id),
        },
    )


def log_tool_end(
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
    duration_ms: float,
    session_id: Optional[str],
) -> None:
    if not session_id:
        return
    _append_event(
        session_id,
        "tool_end",
        {
            "tool_name": tool_name,
            "tool_args": _sanitize_value(tool_args),
            "duration_ms": round(float(duration_ms), 3),
            "success": _is_successful_result(result),
            "result_summary": _summarize_result(result),
            "project_name": _get_project_name(session_id),
        },
    )


def log_run_end(
    agent_name: str,
    model_name: str,
    session_id: Optional[str],
    success: bool,
    error: Optional[BaseException],
    response_text: Optional[str],
    metadata: Optional[dict],
) -> None:
    if not session_id:
        return
    _append_event(
        session_id,
        "run_end",
        {
            "agent_name": agent_name,
            "model_name": model_name,
            "success": success,
            "error": str(error) if error else None,
            "response_text": response_text,
            "metadata": _sanitize_value(metadata or {}),
            "project_name": _get_project_name(session_id),
        },
    )


def _append_event(session_id: str, event: str, payload: Dict[str, Any]) -> None:
    try:
        record = {
            "version": 1,
            "session_id": session_id,
            "sequence": _next_sequence(session_id),
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        path = get_transcript_log_path(session_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    except Exception:
        logger.debug("Session transcript logging failed", exc_info=True)


def _next_sequence(session_id: str) -> int:
    with _sequence_lock:
        next_value = _session_sequences.get(session_id, 0) + 1
        _session_sequences[session_id] = next_value
        return next_value


def _normalize_stream_event(event_type: str, event_data: Any) -> Dict[str, Any]:
    data = event_data if isinstance(event_data, dict) else {}
    normalized: Dict[str, Any] = {"stream_event": event_type}

    if event_type == "part_start":
        part = data.get("part")
        content = getattr(part, "content", None)
        if isinstance(content, str) and content:
            normalized["channel"] = _part_channel(data.get("part_type"), part)
            normalized["text"] = content
            return normalized
    elif event_type == "part_delta":
        delta = data.get("delta")
        content = getattr(delta, "content_delta", None) if delta is not None else None
        if isinstance(content, str) and content:
            normalized["channel"] = _delta_channel(data.get("delta_type"), delta)
            normalized["text"] = content
            return normalized

        args_delta = getattr(delta, "args_delta", None) if delta is not None else None
        if isinstance(args_delta, str) and args_delta:
            normalized["channel"] = "tool_call"
            normalized["args_delta"] = args_delta
            normalized["tool_call_id"] = getattr(delta, "tool_call_id", None)
            normalized["tool_name"] = getattr(delta, "tool_name", None)
            return normalized

    normalized["event_data"] = _sanitize_value(event_data)
    return normalized


def _part_channel(part_type: Any, part: Any) -> str:
    if "Thinking" in str(part_type or "") or "Thinking" in type(part).__name__:
        return "thinking"
    return "assistant"


def _delta_channel(delta_type: Any, delta: Any) -> str:
    if "Thinking" in str(delta_type or "") or "Thinking" in type(delta).__name__:
        return "thinking"
    return "assistant"


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(v) for v in value]
    if hasattr(value, "content_delta"):
        return {
            "type": type(value).__name__,
            "content_delta": _truncate_text(getattr(value, "content_delta", None)),
        }
    if hasattr(value, "args_delta"):
        return {
            "type": type(value).__name__,
            "args_delta": _truncate_text(getattr(value, "args_delta", None)),
            "tool_call_id": getattr(value, "tool_call_id", None),
            "tool_name": getattr(value, "tool_name", None),
        }
    if hasattr(value, "content"):
        return {
            "type": type(value).__name__,
            "content": _truncate_text(getattr(value, "content", None)),
        }
    if hasattr(value, "__dict__"):
        try:
            return {
                "type": type(value).__name__,
                "attrs": _sanitize_value(vars(value)),
            }
        except Exception:
            pass
    return _truncate_text(repr(value))


def _truncate_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= _MAX_TEXT_PREVIEW:
        return value
    return value[: _MAX_TEXT_PREVIEW - 3] + "..."


def _summarize_result(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, str):
        return _truncate_text(result)
    if isinstance(result, dict):
        summary = dict(_sanitize_value(result))
        text = summary.get("text")
        if isinstance(text, str):
            summary["text"] = _truncate_text(text)
        return summary
    return _truncate_text(repr(result))


def _is_successful_result(result: Any) -> bool:
    if isinstance(result, dict):
        if "success" in result:
            return bool(result.get("success"))
        if result.get("blocked") is True:
            return False
        if result.get("error"):
            return False
    return True
