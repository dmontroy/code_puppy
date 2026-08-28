"""Regression tests for ``config_file``'s parsed-config cache.

Background: every ``get_*`` accessor in :mod:`code_puppy.config` funnels into
``_load_config()``, which re-read and re-parsed ``puppy.cfg`` from disk on
EVERY call (~119us). That is fine once, and ruinous in the places that
actually call it -- ``event_stream_handler`` asks for the output level once
per streamed token, and ``callbacks.get_callbacks()`` asks whether each plugin
is disabled once per registered callback per hook dispatch. A single streaming
turn burned ~2.5s of pure config re-parsing on the event loop.

:func:`config_file.load_config_cached` fixes that by keying a cache on the
file's identity ``(mtime_ns, size, inode)``, with explicit eviction on write.
This suite pins the properties that make sharing a parser safe:

* a cache hit returns without touching the disk;
* every way the file can change is noticed (in-process write, external
  rewrite, same-size rewrite, atomic replace, deletion);
* a file that cannot be stat'd is never cached;
* corruption handling is unchanged; and
* the cache is keyed per path, so distinct config files never bleed.
"""

import os
import time
from unittest.mock import patch

import pytest

from code_puppy import config as cp_config
from code_puppy import config_file


@pytest.fixture(autouse=True)
def clean_cache():
    """No test may inherit another's cached parse."""
    config_file.invalidate_config_cache()
    yield
    config_file.invalidate_config_cache()


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "cfgdir"
    cfg_dir.mkdir()
    path = cfg_dir / "puppy.cfg"
    monkeypatch.setattr(cp_config, "CONFIG_FILE", str(path))
    return path


def _write(path, level="low"):
    path.write_text(f"[puppy]\noutput_level = {level}\n")


def _level(path):
    parser = config_file.load_config_cached(str(path))
    return parser.get("puppy", "output_level", fallback=None)


class TestCacheHits:
    def test_repeat_read_returns_same_parser(self, cfg_path):
        _write(cfg_path)
        first = config_file.load_config_cached(str(cfg_path))
        second = config_file.load_config_cached(str(cfg_path))
        assert first is second

    def test_cache_hit_does_not_touch_disk(self, cfg_path):
        _write(cfg_path)
        config_file.load_config_cached(str(cfg_path))  # prime
        with patch.object(
            config_file, "_read_unlocked", side_effect=AssertionError("re-read!")
        ):
            assert _level(cfg_path) == "low"

    def test_value_is_correct_on_first_read(self, cfg_path):
        _write(cfg_path, "high")
        assert _level(cfg_path) == "high"


class TestInvalidation:
    def test_mutate_config_evicts(self, cfg_path):
        _write(cfg_path, "low")
        assert _level(cfg_path) == "low"

        def _apply(parser):
            if "puppy" not in parser:
                parser["puppy"] = {}
            parser["puppy"]["output_level"] = "high"

        config_file.mutate_config(str(cfg_path), _apply)
        assert _level(cfg_path) == "high"

    def test_external_rewrite_is_noticed(self, cfg_path):
        _write(cfg_path, "low")
        assert _level(cfg_path) == "low"
        time.sleep(0.01)
        _write(cfg_path, "high")
        assert _level(cfg_path) == "high"

    def test_same_size_rewrite_is_noticed(self, cfg_path):
        """A value swap that keeps byte-length identical must still invalidate."""
        _write(cfg_path, "low ")
        assert _level(cfg_path) == "low"
        time.sleep(0.01)
        _write(cfg_path, "high")
        assert _level(cfg_path) == "high"

    def test_atomic_replace_is_noticed(self, cfg_path):
        """The real write path is temp-file + os.replace, i.e. a new inode."""
        _write(cfg_path, "low")
        assert _level(cfg_path) == "low"
        replacement = cfg_path.parent / "incoming.cfg"
        replacement.write_text("[puppy]\noutput_level = high\n")
        os.replace(str(replacement), str(cfg_path))
        assert _level(cfg_path) == "high"

    def test_explicit_invalidation_for_one_path(self, cfg_path):
        _write(cfg_path, "low")
        config_file.load_config_cached(str(cfg_path))
        config_file.invalidate_config_cache(str(cfg_path))
        assert _level(cfg_path) == "low"

    def test_global_invalidation(self, cfg_path):
        _write(cfg_path, "low")
        config_file.load_config_cached(str(cfg_path))
        config_file.invalidate_config_cache()
        assert _level(cfg_path) == "low"

    def test_invalidating_unknown_path_is_a_noop(self, cfg_path):
        config_file.invalidate_config_cache("/no/such/config.cfg")


class TestUnstattableFiles:
    def test_missing_file_is_not_cached(self, cfg_path):
        _write(cfg_path, "low")
        assert _level(cfg_path) == "low"
        os.remove(str(cfg_path))
        assert _level(cfg_path) is None

    def test_file_reappearing_is_seen(self, cfg_path):
        assert _level(cfg_path) is None
        _write(cfg_path, "high")
        assert _level(cfg_path) == "high"


class TestUnchangedSemantics:
    def test_corrupt_file_still_recovers(self, cfg_path):
        cfg_path.write_text("this is not [valid ini at all\nnope=nope=nope\n")
        parser = config_file.load_config_cached(str(cfg_path))
        assert parser.sections() == []

    def test_cache_is_keyed_per_path(self, cfg_path):
        other = cfg_path.parent / "other.cfg"
        _write(cfg_path, "low")
        other.write_text("[puppy]\noutput_level = high\n")
        assert _level(cfg_path) == "low"
        assert _level(other) == "high"
        assert _level(cfg_path) == "low"

    def test_load_config_still_returns_private_parsers(self, cfg_path):
        """The uncached entry point must keep its fresh-instance contract.

        ``ensure_config_exists`` mutates what it is handed; if that ever came
        back shared, it would poison every reader.
        """
        _write(cfg_path)
        a = config_file.load_config(str(cfg_path))
        b = config_file.load_config(str(cfg_path))
        assert a is not b

    def test_get_value_reflects_set_config_value(self, cfg_path):
        """End-to-end through the public config API, cache included."""
        cp_config.set_config_value("output_level", "high")
        assert cp_config.get_value("output_level") == "high"
        cp_config.set_config_value("output_level", "low")
        assert cp_config.get_value("output_level") == "low"


class TestConcurrentWriteDuringParse:
    def test_file_changing_mid_parse_is_not_cached_stale(self, cfg_path):
        """If the file moves under us while parsing, don't pin the stale parse.

        The post-read re-stat means we simply decline to cache; the next call
        re-reads and sees the new content.
        """
        _write(cfg_path, "low")
        real_read = config_file._read_unlocked

        def racing_read(path):
            parsed = real_read(path)
            # Someone else writes while we were parsing.
            cfg_path.write_text("[puppy]\noutput_level = high\n")
            return parsed

        with patch.object(config_file, "_read_unlocked", side_effect=racing_read):
            config_file.load_config_cached(str(cfg_path))

        assert _level(cfg_path) == "high"
