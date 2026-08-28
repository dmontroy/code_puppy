# Prompt-Path & Agent-Loop Performance Analysis

**Status:** analysis + measured proposals.
**Finding #1 is FIXED** on branch `config-optimization` — see
[Finding 1 → Fix](#fix). Findings #2-#6 remain open.
**Scope:** prompt assembly, the pydantic-ai agent loop, the streaming event
handler, callback dispatch, and the ACP (`session/prompt`) turn.

---

## TL;DR

The agent loop is not slow because of the model, the network, or the history
size. It is slow because **`puppy.cfg` is re-read and re-parsed from disk on
essentially every hot-path decision** — thousands of times per turn.

A single `get_value()` call costs **~0.11 ms** (open + read + `configparser`
parse). Nothing caches it. That one fact is the root cause of every finding
below.

Measured on a realistic ACP turn (4 000 streamed deltas, 4 plugin-owned
`stream_event` callbacks — the stock ACP set):

| | per turn | per delta |
|---|---|---|
| Today | **2 565 ms** | 641 µs |
| With an mtime-cached config | **63 ms** | 16 µs |
| | **40.5× faster** | **2.5 s saved per turn** |

That 2.5 s is pure CPU + syscall burn **on the event loop**, so it directly
delays streamed tokens reaching the terminal and `session/update`
notifications reaching the ACP client.

---

## Method

All numbers from `.venv/bin/python` on this machine, median of 20–300
iterations. Synthetic history = 180 messages (60 turns of
prompt → response+tool-call → 8 KB tool return), which is an ordinary
mid-session Code Puppy conversation.

Benchmarks are reproducible; see [Appendix A](#appendix-a--reproducing) for
the harness.

---

## Finding 1 — `get_value()` is uncached, and it is everywhere 

`config.get_value()` → `config._load_config()` → `config_file.load_config()` →
`atomic_io.read_bounded_bytes()` + `ConfigParser.read_string()`.

**Every single call hits the filesystem.** There is no cache, no memoization,
no mtime check.

```
config.get_value('output_level')      median = 0.1154 ms
config.get_output_level()             median = 0.1099 ms
ModelFactory.load_config()            median = 0.1443 ms
config.get_model_context_length()     median = 0.1411 ms
_builder.load_puppy_rules()           median = 0.1459 ms
```

0.11 ms sounds like nothing. It is nothing — *once*. The problem is the call
sites.

### 1a. Per streamed delta

`event_stream_handler`'s inner `async for event` loop calls
`_suppress_thinking_stream()` / `_suppress_tool_progress()` on **every single
delta**. Both funnel into `get_output_level()` → `get_value()` → disk.

```
_suppress_thinking_stream() x4000  =  932.4 ms
_suppress_tool_progress()   x4000  =  462.0 ms
```

`_suppress_thinking_stream()` is doubly bad: in the default `medium` output
level it evaluates `level == "low" or get_suppress_thinking_messages()`, so it
pays for **two** config reads per delta.

> This is a per-token disk read. Rendering density cannot change mid-token.

### 1b. Per callback, per hook fire 

This one is the real killer.

```python
# callbacks.py
def get_callbacks(phase, *, include_disabled=False):
    all_cbs = _callbacks.get(phase, []).copy()
    return [cb for cb in all_cbs
            if is_callback_owner_enabled(_callback_owners.get(cb))]
#            ^^^^^^^^^^^^^^^^^^^^^^^^^ → _get_disabled_plugins()
#                                      → get_disabled_plugins()
#                                      → get_value("disabled_plugins")  ← DISK
```

`get_callbacks()` performs **one full `puppy.cfg` read+parse per registered,
plugin-owned callback, on every hook dispatch.**

```
get_callbacks('stream_event')  [0 registered]      median = 0.0001 ms
get_callbacks('stream_event')  [5 owned callbacks] median = 0.5545 ms
   -> ownership filter cost: 554 us per dispatch
```

Now recall that `_fire_stream_event()` fires the `stream_event` phase **per
streamed delta**, and a stock ACP session has four plugin-owned subscribers
(`acp`, `emoji_filter`, `frontend_emitter`, `subagent_panel`):

```
_fire_stream_event x1000: create=1.2 ms  drain=623.4 ms
```

**0.62 ms per delta**, ~99 % of which is asking the config file "is this
plugin still enabled?" — four times, for the same answer, forty thousand times
a session.

### 1c. Per message, per token estimate

`BaseAgent.estimate_tokens_for_message()` resolves the model name *per
message*:

```python
def estimate_tokens_for_message(self, message):
    return estimate_tokens_for_message(message, self.get_model_name())
    #                                            ^ get_agent_pinned_model() → DISK
```

```
agent.get_model_name()  [x1]                             =  0.108 ms
estimate_tokens_for_message x180 (model resolved ONCE)   =  0.131 ms
agent.estimate_tokens_for_message x180 (BaseAgent path)  = 20.677 ms   ← 158×
```

**158× slower than the work it wraps.** Call sites that pay this per turn or
per save:

- `session_storage.save_session()` → `sum(token_estimator(m) for m in history)`
  — every ACP turn persists, so every ACP turn does 180+ config reads
  (thankfully in `asyncio.to_thread`, but it still contends on the FS).
- `model_switching._refresh_context_status()`
- `cli_runner`, `session_commands` (`/compact`, `/truncate`, session listing).

`_compaction.compact()` gets this right — it resolves `model_name` once and
passes it down. That is the pattern; everything else should copy it.

### Fix

**SHIPPED** on branch `config-optimization`.

`config_file.load_config_cached()` keys a cache on the file's identity
`(st_mtime_ns, st_size, st_ino)`. The inode is the load-bearing third element:
writes go through `atomic_write_bytes` (temp file + `os.replace`), so every
write produces a **new inode** — meaning even a same-size rewrite inside a
single timestamp tick is detected. `mutate_config()` additionally evicts the
entry explicitly while it still holds the write lock, so in-process writes
never depend on filesystem timestamp resolution at all.

Two safety properties worth naming:

- **`load_config()` keeps its original contract** — a private, freshly-parsed
  parser. Only the new `load_config_cached()` shares one.
  `ensure_config_exists()`, the single caller that *mutates* the parser it is
  handed, was moved onto the uncached path so it cannot poison the shared
  instance. The other four `_load_config()` callers are strictly read-only
  (`.get`, `.keys()`, `.items()`, `in`).
- **The identity is re-stat'd after the read.** If the file changes while we
  are parsing, we decline to cache rather than pin a stale parse under a
  current-looking key.

Measured against the real patched code:

| | before | after | |
|---|---|---|---|
| `get_value('output_level')` | 0.1154 ms | **0.0016 ms** | 72× |
| `get_callbacks('stream_event')`, 4 plugins | 0.44 ms | **0.010 ms** | 44× |
| `estimate_tokens_for_message` ×180 | 20.677 ms | **0.490 ms** | 42× |
| `_estimate_context_overhead()` | 0.468 ms | **0.030 ms** | 15× |
| **full turn, 4000 deltas** | **2564.9 ms** | **65.8 ms** | **39×** |

Tests: `tests/test_config_cache.py` — 17 cases covering cache hits, all five
invalidation routes (in-process write, external rewrite, same-size rewrite,
atomic replace, deletion), unstattable files, corruption passthrough, per-path
keying, and a write-during-parse race. Full suite: **7613 passed, 0 failed.**

**This single change was worth more than every other item in this document
combined.**

---

## Finding 2 — `_estimate_context_overhead()` recomputes the system prompt per model request 

Called from the history processor, i.e. **before every model request**:

```python
def _estimate_context_overhead(self) -> int:
    system_prompt = self.get_full_system_prompt()   # ← fires on_load_prompt()
    ...
    return estimate_context_overhead(resolved, tools, ...)
```

`get_full_system_prompt()` fires the `load_prompt` hook, whose stock
subscribers do real I/O:

| plugin | work done per model request |
|---|---|
| `puppy_kennel` | `packer.pack()` → **SQLite connect + queries** |
| `file_permission_handler` | permission-rule assembly |
| `namespace_skill_search` | skill catalogue scan |
| `computer_use` | capability probe |

And `estimate_context_overhead()` then `json.dumps()` **every registered tool's
JSON schema** plus every cached MCP tool schema — again, per request.

```
agent._estimate_context_overhead()  =  0.468 ms   (with no kennel DB present)
```

The measured cost here is modest on a bare install, but it scales with plugin
count, tool count and MCP server count — and it's *unbounded*, because plugins
own the work.

The deeper problem is that this is **provably redundant**: the instructions the
model actually receives were frozen at agent-build time by
`_builder._assemble_instructions()`. Recomputing the prompt per request to
estimate its token count means (a) doing the expensive work twice and (b)
risking the estimate drifting from the reality — the badge can lie.

### Fix

Compute the overhead **once per agent build** (where the instructions are
already in hand) and cache it on the agent, invalidated by the same events that
rebuild the pydantic agent (model change, tool-set change, MCP attach).
Zen: *"There should be one — and preferably only one — obvious way to do it."*
The system prompt already has one owner; the estimator should read it, not
re-derive it.

---

## Finding 3 — `ModelFactory.load_config()` is uncached and parses `models.json` twice 

Per call it: reads bundled `models.json`, then up to five overlay files
(`extra_models.json`, ChatGPT/Claude/Gemini/Copilot OAuth models), fires two
callback phases, and then **re-opens and re-parses bundled `models.json` a
second time** just to build a description overlay map.

```python
# pass 1
bundled_models = pathlib.Path(__file__).parent / "models.json"
with open(bundled_models) as f:
    config = json.load(f)
...
# pass 2 — same file, again
bundled_models = pathlib.Path(__file__).parent / "models.json"
with open(bundled_models) as f:
    bundled_config = json.load(f)
```

21 call sites, including `BaseAgent._get_model_context_length()` — which the
history processor calls **per model request**.

Measures at only 0.144 ms here because this checkout's `models.json` is 4
bytes. On a real install with OAuth model files it is materially worse, and
`models_dev_api.json` is 535 KB sitting right next door.

### Fix

Same mtime-keyed cache treatment. At minimum, hoist the duplicated
`models.json` read — that one is free.

---

## Finding 4 — the history processor re-derives everything, every request 

`_compaction.make_history_processor()` runs before **every** model request and
performs a full-history pass for each of:

1. `on_message_history_processor_start(message_history=list(history), incoming_messages=list(messages))` — two list copies + hook dispatch
2. `{hash_message(m) for m in history}` — SHA-256 over every message's full stringified content
3. `agent._get_model_context_length()` — Finding 3
4. `agent._estimate_context_overhead()` — Finding 2
5. `compact()` → another full-history token sum
6. `_strip_empty_thinking_parts()` — rebuild
7. `sanitize_tool_call_ids()` — rebuild
8. `on_message_history_processor_end(message_history=list(cleaned))` — another copy + dispatch

```
hash_message x180             = 0.292 ms
sanitize_tool_call_ids(180)   = 0.022 ms
prune_interrupted_tool_calls  = 0.021 ms
```

Individually cheap, but note the shape: **with S steps in a turn and N messages
in history, this is O(S·N) — quadratic in a long turn.** Every completed step
appends to history, so step 40 re-hashes all 39 previous steps' messages.

`hash_message()` and `estimate_tokens_for_message()` both call
`stringify_part()`, building a full string copy of every message — including
multi-megabyte tool returns — twice per request.

### Fix

Memoize per message. Messages are immutable once appended, so a hash/token
value can be computed once at append time and reused. A `WeakKeyDictionary`
(or an id-keyed cache cleared per run) keyed on the message object turns the
O(S·N) sweep into O(S·Δ).

DRY note: `stringify_part()` is walked twice for the same message to produce
two derived values. Walk once, return both.

---

## Finding 5 — ACP emits one JSON-RPC notification per token 

`acp/bridge.py::_on_stream_event` sends a `session/update` for **every**
`part_delta`:

```python
if not content:
    return
...
await self._send(session_id, update)      # one JSON-RPC frame per token
```

Code Puppy's own terminal renderer coalesces — `SmoothTermflowWriter` and
line-buffering in `event_stream_handler` mean the console gets whole lines. The
ACP bridge has **no coalescing at all**: every token becomes a Pydantic model
construction, a JSON serialization, and a framed stdout write.

There is also an **ordering hazard**. `_fire_stream_event()` does:

```python
asyncio.create_task(callbacks.on_stream_event(event_type, event_data, sid))
```

Fire-and-forget, one task per delta. Each task runs to its first `await`
(inside `connection.session_update`) and then yields — so interleaved delta
tasks can race at the transport boundary. Today it appears to hold, but it
holds by accident of scheduling, not by construction. Scrambled agent text is
a nasty failure mode to debug.

Secondary: those tasks are created and never awaited or tracked, so exceptions
surface only as "task exception was never retrieved", and nothing bounds how
many are in flight.

### Fix

1. **Coalesce.** Buffer deltas and flush on a short timer (~16–33 ms) or on
   part-end/tool-call boundary. Editors render at frame rate; they do not need
   per-token frames. This is strictly fewer, larger writes — better latency
   *and* less CPU.
2. **Serialize.** Feed bridge updates through a single `asyncio.Queue` drained
   by one long-lived writer task. Ordering becomes structural, task churn
   drops from N-per-turn to one, and exceptions get a real owner.

---

## Finding 6 — smaller stuff 

**Per-delta imports.** The pause gate re-executes
`from code_puppy.messaging.pause_controller import get_pause_controller`
inside the `async for event` loop, plus `from code_puppy.config import
get_value` inside the pause `while`. `sys.modules` hits are cheap but not free,
and this is the single hottest loop in the application. Hoist them.

**Eager log formatting.** `_trigger_callbacks_sync()` / `_trigger_callbacks()`
build `f"No callbacks registered for phase '{phase}'"` and
`f"Successfully executed callback {callback.__name__}"` unconditionally, at
`DEBUG`, per callback, per dispatch. Use `logger.debug("...%s", phase)` and let
logging decide. YAGNI: the happy-path success log earns nothing.

**Double agent build.** `build_pydantic_agent()` constructs the pydantic agent
and runs `register_tools_for_agent()` **twice** — a throwaway probe pass with
`toolsets=[]` purely to discover tool names for MCP collision filtering, then
the real build.

```
build_pydantic_agent(agent)  [2-pass build]  = 15.704 ms
```

~8 ms of that is the probe. It's on the critical path of ACP `new_session`,
`load_session`, `resume_session`, `fork_session`, and every model rebind — i.e.
editor time-to-first-interaction. The tool-name set is derivable from
`agent.get_available_tools()` + the tool registry without instantiating a
`PydanticAgent`; the probe should be a lookup, not a construction.

**Duplicated first-turn prompt work.** `_runtime._should_prepend_system_prompt()`
calls `get_full_system_prompt()` + `load_puppy_rules()` again on turn one —
work `_assemble_instructions()` just finished during the build a few lines
earlier. Pass the assembled instructions through instead of re-deriving them
(and re-running every `load_prompt` plugin).

---

## Recommended order of work

| # | Change | Effort | Win |
|---|---|---|---|
| ~~1~~ | ~~mtime-cache `config._load_config()`~~ **DONE** | ~120 lines | **39× on the streaming path; 2.5 s/turn** |
| 2 | Resolve `model_name` once in `estimate_tokens_for_message` callers | small | 158× on token sums |
| 3 | Coalesce + serialize ACP `session/update` writes | medium | Big ACP latency + ordering safety |
| 4 | Cache `_estimate_context_overhead()` per agent build | small | Kills per-request SQLite/plugin I/O |
| 5 | mtime-cache `ModelFactory.load_config()`, drop the double read | small | Per-request JSON parsing |
| 6 | Memoize `hash_message` / token estimates per message | medium | Removes O(S·N) sweep |
| 7 | Hoist per-delta imports, lazy log formatting | trivial | Tidy the hottest loop |
| 8 | Remove the probe pass in `build_pydantic_agent` | medium | ~8 ms off every session open |

Items 1 and 2 are, between them, roughly a 40× improvement on the hot path for
about thirty lines of code. Everything else is gravy.

---

## A note on the pattern

Every finding here is the same mistake wearing a different hat: **a cheap
question answered by an expensive lookup, repeated in a loop.**

`get_output_level()`, `get_disabled_plugins()`, `get_model_name()`,
`load_config()`, `get_full_system_prompt()` — all of them read like free
property accesses at the call site. None of them are. The call sites are
innocent; the accessors are lying about their cost.

The fix is not to sprinkle caching at call sites (that's the DRY violation
waiting to happen — twenty call sites each with their own stale-cache bug).
It is to make the accessors honest: cache once, at the source, invalidated by
the one thing that can change the answer.

> *Special cases aren't special enough to break the rules.*

---

## Appendix A — Reproducing

```bash
uv sync
.venv/bin/python /tmp/bench_cp.py    # config + hashing costs
.venv/bin/python /tmp/bench_cp2.py   # callback dispatch + simulated turn
.venv/bin/python /tmp/bench_cp3.py   # agent build + per-request overhead
.venv/bin/python /tmp/bench_cp4.py   # isolating the 20 ms
.venv/bin/python /tmp/bench_cp5.py   # before/after with prototype cache
```

The prototype cache used for the "after" numbers:

```python
import configparser, os
from code_puppy import config as cfg

_state = {"key": None, "parser": None}
_real = cfg._load_config

def _cached() -> configparser.ConfigParser:
    try:
        st = os.stat(cfg.CONFIG_FILE)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if _state["parser"] is not None and _state["key"] == key:
        return _state["parser"]
    parser = _real()
    _state.update(key=key, parser=parser)
    return parser

cfg._load_config = _cached
```

Production version should additionally bust the cache explicitly from
`set_config_value()` / `mutate_config()` (sub-second `st_mtime_ns` resolution
is fine on APFS/ext4, but an explicit invalidation is free insurance), and
should be process-local — ACP runs concurrent sessions on one process and they
all share the same `puppy.cfg`.
