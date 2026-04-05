"""Tests for src/utils.py — config loading, caching, token tracker."""

import json

import pytest
import yaml

from src.utils import CONFIG_ENV_VAR, TokenTracker, _cache_key, load_config, resolve_config_path


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

    def test_load_uses_env_override(self, tmp_path, monkeypatch):
        cfg = {"backend": "openai", "n_examples": 7}
        cfg_file = tmp_path / "env_config.yaml"
        cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
        monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg_file))

        result = load_config()

        assert result["backend"] == "openai"
        assert result["n_examples"] == 7

    def test_resolve_config_prefers_explicit_path(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit.yaml"
        explicit.write_text("backend: local\n", encoding="utf-8")
        env_cfg = tmp_path / "env.yaml"
        env_cfg.write_text("backend: openai\n", encoding="utf-8")
        monkeypatch.setenv(CONFIG_ENV_VAR, str(env_cfg))

        resolved = resolve_config_path(explicit)

        assert resolved == explicit.resolve()


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
