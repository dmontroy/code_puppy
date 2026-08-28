"""Bounded, recoverable I/O for Code Puppy's INI configuration file.

The public helpers in this module keep config-file safety concerns out of the
already-large :mod:`code_puppy.config` module. Built on the generic
primitives in :mod:`code_puppy.atomic_io`:

* reads are size-bounded before parsing, preventing pathological lines from
  exhausting memory;
* only confirmed content corruption is quarantined -- ordinary I/O failures
  still propagate;
* recovery and writes share a cross-process lock;
* quarantine names are collision-resistant and never overwrite a backup; and
* writes use a same-directory temporary file followed by atomic replacement.
"""

from __future__ import annotations

import configparser
import io
import locale
import logging
import os
import threading
from collections.abc import Callable
from typing import Optional, Tuple

from code_puppy import atomic_io

logger = logging.getLogger(__name__)

MAX_CONFIG_BYTES = atomic_io.DEFAULT_MAX_BYTES
_LOCK_TIMEOUT_SECONDS = atomic_io.DEFAULT_LOCK_TIMEOUT_SECONDS

# Re-exported for backwards compatibility -- callers (and this module's own
# tests) historically imported the lock timeout error from here.
ConfigLockTimeout = atomic_io.LockTimeout

# ---------------------------------------------------------------------------
# Parsed-config cache
# ---------------------------------------------------------------------------
#
# ``get_value()`` -- and therefore every ``get_*`` accessor in ``config.py`` --
# used to reach disk on EVERY call: open, bounded read, full ``configparser``
# parse, ~119us a pop. Harmless once; catastrophic in the places that actually
# call it. ``event_stream_handler`` asks for the output level once per streamed
# token, and ``callbacks.get_callbacks()`` asks whether each plugin is disabled
# once per registered callback per hook dispatch -- so a single streaming turn
# burned ~2.5 SECONDS of pure config re-parsing, on the event loop, delaying
# both terminal output and ACP ``session/update`` notifications.
#
# The answer can only change when the file changes, so key a cache on the
# file's identity: (mtime_ns, size, inode). Writes go through
# ``atomic_write_bytes`` (temp file + ``os.replace``), which yields a brand new
# inode every time, so even a same-size same-tick rewrite is detected. A
# ``stat`` costs 0.75us against the 119us it saves.
#
# In-process writers do not rely on that at all -- ``mutate_config`` evicts the
# entry explicitly while it still holds the write lock.

_CacheKey = Tuple[int, int, int]

# path -> (identity key, parsed config). Guarded by ``_CACHE_LOCK``.
_cache: dict[str, Tuple[_CacheKey, configparser.ConfigParser]] = {}
_CACHE_LOCK = threading.Lock()


def _stat_key(path: str) -> Optional[_CacheKey]:
    """Identity of the file at ``path``, or ``None`` if it can't be stat'd.

    ``None`` means "never serve this from cache" -- a missing or unreadable
    file must keep falling through to the real read so its error handling
    (and first-run creation) behaves exactly as before.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def invalidate_config_cache(path: Optional[str] = None) -> None:
    """Drop cached parse results for ``path`` (or all paths when ``None``).

    Callers that write config through this module never need this -- it is
    already wired into :func:`mutate_config`. It is exported for tests and for
    any exotic caller that replaces a config file behind our back.
    """
    with _CACHE_LOCK:
        if path is None:
            _cache.clear()
        else:
            _cache.pop(path, None)


class ConfigFileCorrupt(Exception):
    """The file was read successfully but its contents are not safe INI."""


def _config_lock(path: str):
    """Serialize recovery and read-modify-write operations across processes."""
    return atomic_io.path_lock(path, timeout=_LOCK_TIMEOUT_SECONDS)


def _decode_config_text(raw: bytes) -> str:
    """Decode config text as UTF-8, with a narrow Windows locale fallback.

    Canonical on-disk encoding is UTF-8 (with or without BOM). For Windows
    upgrades from older builds that may have locale-encoded config bytes,
    make one fallback attempt using the active locale encoding before treating
    the file as corrupt.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as utf8_error:
        if os.name != "nt":
            raise

        fallback_encoding = locale.getpreferredencoding(False)
        if not fallback_encoding:
            raise

        try:
            decoded = raw.decode(fallback_encoding)
        except (LookupError, UnicodeDecodeError):
            raise utf8_error

        if "\x00" in decoded or "[" not in decoded:
            raise utf8_error

        logger.warning(
            "Loaded config with Windows locale fallback encoding (%s); "
            "rewrite as UTF-8 on next save",
            fallback_encoding,
        )
        return decoded


def _read_unlocked(path: str) -> configparser.ConfigParser:
    """Read and parse a bounded snapshot; propagate filesystem failures."""
    parser = configparser.ConfigParser()
    try:
        raw = atomic_io.read_bounded_bytes(path, max_bytes=MAX_CONFIG_BYTES)
    except atomic_io.ContentTooLarge as exc:
        raise ConfigFileCorrupt(str(exc)) from exc

    try:
        text = _decode_config_text(raw)
        if text:
            parser.read_string(text, source=path)
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise ConfigFileCorrupt(str(exc)) from exc
    return parser


def _quarantine_unlocked(path: str) -> str:
    # Quarantine moves the file aside, so anything cached for it is now a
    # parse of content that no longer lives at this path.
    invalidate_config_cache(path)
    return atomic_io.quarantine_file(path)


def load_config(path: str) -> configparser.ConfigParser:
    """Load config, quarantining only confirmed content corruption.

    A second read under the process lock closes the read/quarantine TOCTOU
    window between cooperating Code Puppy processes. If another process fixed
    or replaced the file first, its healthy version wins and is returned.
    Filesystem errors and genuine ``MemoryError`` conditions propagate; neither
    proves that user data is corrupt.
    """
    try:
        return _read_unlocked(path)
    except ConfigFileCorrupt as first_error:
        with _config_lock(path):
            try:
                return _read_unlocked(path)
            except ConfigFileCorrupt as confirmed_error:
                quarantine_path = _quarantine_unlocked(path)
                logger.warning(
                    "Corrupted config %s was quarantined to %s: %s",
                    path,
                    quarantine_path,
                    confirmed_error or first_error,
                )
                return configparser.ConfigParser()


def load_config_cached(path: str) -> configparser.ConfigParser:
    """Return a **shared, read-only** parse of ``path``, reusing it when valid.

    Same recovery semantics as :func:`load_config` -- corruption handling,
    quarantine and locking all happen there; this only decides whether that
    work needs doing again.

    .. warning::
       The returned parser is **shared between all callers**. Treat it as
       read-only. Mutating it corrupts every other reader's view until the
       next invalidation. A caller that needs to modify a parser wants
       :func:`load_config` (a private, freshly-parsed instance) or, to write,
       :func:`mutate_config`.

    A file that cannot be stat'd is never cached, so missing-file and
    permission-error behaviour is identical to the uncached path.
    """
    key = _stat_key(path)
    if key is not None:
        with _CACHE_LOCK:
            entry = _cache.get(path)
        if entry is not None and entry[0] == key:
            return entry[1]

    parser = load_config(path)

    # Re-stat AFTER the read: if the file changed while we were parsing, the
    # pre-read key would label stale content as current and pin it until the
    # next write. Storing the post-read identity means a concurrent write is
    # either captured or re-read next call -- never silently wrong.
    if _stat_key(path) == key and key is not None:
        with _CACHE_LOCK:
            _cache[path] = (key, parser)
    return parser


def _atomic_write_unlocked(path: str, parser: configparser.ConfigParser) -> None:
    """Durably replace ``path`` with serialized config from a temp file."""
    buffer = io.StringIO()
    parser.write(buffer)
    atomic_io.atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def mutate_config(
    path: str, mutation: Callable[[configparser.ConfigParser], bool | None]
) -> configparser.ConfigParser:
    """Apply one locked read-modify-write transaction.

    ``mutation`` may return ``False`` to skip an unnecessary write. A corrupt
    file is quarantined while the same lock remains held; if quarantine fails,
    the exception propagates and the mutation is never written over user data.
    """
    with _config_lock(path):
        try:
            parser = _read_unlocked(path)
        except ConfigFileCorrupt as exc:
            quarantine_path = _quarantine_unlocked(path)
            logger.warning(
                "Corrupted config %s was quarantined to %s: %s",
                path,
                quarantine_path,
                exc,
            )
            parser = configparser.ConfigParser()
        should_write = mutation(parser)
        if should_write is not False:
            _atomic_write_unlocked(path, parser)
        # Evict before dropping the lock: a reader must never be able to
        # observe the post-write file through a pre-write cache entry. The
        # stat key would catch it anyway; this makes it deterministic rather
        # than dependent on filesystem timestamp resolution.
        invalidate_config_cache(path)
        return parser
