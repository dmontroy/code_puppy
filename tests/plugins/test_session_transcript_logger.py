import json

from code_puppy.plugins.session_transcript_logger import logger as transcript_logger


class _TextPart:
    def __init__(self, content: str) -> None:
        self.content = content


class _ThinkingDelta:
    def __init__(self, content_delta: str) -> None:
        self.content_delta = content_delta


class _ToolArgsDelta:
    def __init__(
        self,
        args_delta: str,
        tool_call_id: str = "call-1",
        tool_name: str = "edit_file",
    ) -> None:
        self.args_delta = args_delta
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_structured_session_log_writes_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_logger, "STATE_DIR", str(tmp_path))
    transcript_logger.reset_state()

    transcript_logger.log_user_prompt("Summarize this repo", "sess-123")
    transcript_logger.log_run_start("main", "gpt-4.1", "sess-123")
    transcript_logger.log_stream_event(
        "part_start",
        {"part": _TextPart("Hello"), "part_type": "TextPart"},
        "sess-123",
    )
    transcript_logger.log_tool_start("read_file", {"path": "README.md"}, "sess-123")
    transcript_logger.log_tool_end(
        "read_file",
        {"path": "README.md"},
        {"success": True, "text": "done"},
        12.34,
        "sess-123",
    )
    transcript_logger.log_run_end(
        "main",
        "gpt-4.1",
        "sess-123",
        True,
        None,
        "Final answer",
        {"usage_total_tokens": 42},
    )

    path = transcript_logger.get_transcript_log_path("sess-123")
    records = _read_records(path)

    assert [record["event"] for record in records] == [
        "user_prompt",
        "run_start",
        "stream_event",
        "tool_start",
        "tool_end",
        "run_end",
    ]
    assert [record["sequence"] for record in records] == [1, 2, 3, 4, 5, 6]
    assert records[0]["text"] == "Summarize this repo"
    assert records[2]["channel"] == "assistant"
    assert records[2]["text"] == "Hello"
    assert records[4]["success"] is True
    assert records[5]["response_text"] == "Final answer"
    assert records[5]["metadata"]["usage_total_tokens"] == 42


def test_stream_event_normalizes_thinking_and_tool_deltas(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_logger, "STATE_DIR", str(tmp_path))
    transcript_logger.reset_state()

    transcript_logger.log_stream_event(
        "part_delta",
        {
            "delta": _ThinkingDelta("step by step"),
            "delta_type": "ThinkingPartDelta",
        },
        "sess-abc",
    )
    transcript_logger.log_stream_event(
        "part_delta",
        {
            "delta": _ToolArgsDelta('{\"path\":\"app.py\"}'),
            "delta_type": "ToolCallPartDelta",
        },
        "sess-abc",
    )

    records = _read_records(transcript_logger.get_transcript_log_path("sess-abc"))

    assert records[0]["channel"] == "thinking"
    assert records[0]["text"] == "step by step"
    assert records[1]["channel"] == "tool_call"
    assert records[1]["args_delta"] == '{"path":"app.py"}'
    assert records[1]["tool_name"] == "edit_file"


def test_logger_skips_missing_session_id(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_logger, "STATE_DIR", str(tmp_path))
    transcript_logger.reset_state()

    transcript_logger.log_user_prompt("hello", None)
    transcript_logger.log_run_start("main", "gpt-4.1", None)
    transcript_logger.log_stream_event("part_start", {"part": _TextPart("x")}, None)
    transcript_logger.log_tool_start("read_file", {"path": "x"}, None)
    transcript_logger.log_run_end("main", "gpt-4.1", None, True, None, "done", {})

    assert list((tmp_path / "logs").glob("**/*")) == []
