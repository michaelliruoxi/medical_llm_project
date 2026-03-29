"""Tests for src/utils.py — config loading, caching, token tracker."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.utils import load_config, load_prompts, _cache_key, TokenTracker


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        cfg = {"backend": "local", "n_examples": 10}
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
        result = load_config(str(cfg_file))
        assert result["backend"] == "local"
        assert result["n_examples"] == 10

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


class TestCacheKey:
    def test_deterministic(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _cache_key("model-a", msgs, 0.5, 100)
        k2 = _cache_key("model-a", msgs, 0.5, 100)
        assert k1 == k2

    def test_different_models_different_keys(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _cache_key("model-a", msgs, 0.5, 100)
        k2 = _cache_key("model-b", msgs, 0.5, 100)
        assert k1 != k2

    def test_different_temps_different_keys(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _cache_key("model-a", msgs, 0.2, 100)
        k2 = _cache_key("model-a", msgs, 0.8, 100)
        assert k1 != k2

    def test_different_api_modes_different_keys(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _cache_key("model-a", msgs, 0.2, 100, backend="openai", api_mode="chat_completions")
        k2 = _cache_key("model-a", msgs, 0.2, 100, backend="openai", api_mode="responses")
        assert k1 != k2


class TestTokenTracker:
    def test_add_and_summary(self):
        t = TokenTracker()
        t.add(100, 50)
        t.add(200, 100)
        s = t.summary()
        assert s["calls"] == 2
        assert s["prompt_tokens"] == 300
        assert s["completion_tokens"] == 150
        assert s["total_tokens"] == 450

    def test_initial_state(self):
        t = TokenTracker()
        assert t.total_tokens == 0
        assert t.summary()["calls"] == 0
