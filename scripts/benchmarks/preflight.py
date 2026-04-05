"""Preflight enabled model configs with a tiny smoke test.

This script is intentionally much lighter than the full comparison pipeline:
it loads each configured local model or makes one tiny API call for OpenAI
configs, records success/failure details, and unloads local models before
moving to the next one.

Usage:
    python scripts/benchmarks/preflight.py
    python scripts/benchmarks/preflight.py --configs configs/models/mistral_7b.yaml
    python scripts/benchmarks/preflight.py --load-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

# Block broken TensorFlow BEFORE anything else imports it
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import importlib.machinery

_tf_stub = types.ModuleType("tensorflow")
_tf_stub.__version__ = "0.0.0"
_tf_stub.__path__ = []
_tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
sys.modules["tensorflow"] = _tf_stub

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils import PROJECT_ROOT as SRC_PROJECT_ROOT
from src.utils import (
    _get_local_model,
    _local_generate,
    call_llm,
    load_config,
    quantization_label,
    set_active_config,
    setup_logging,
    unload_local_models,
)


logger = setup_logging()


def discover_configs(config_dir: Path) -> list[str]:
    configs = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = load_config(str(config_path))
            if cfg.get("enabled", True) is False:
                continue
        except Exception:
            pass
        configs.append(str(config_path))
    return configs


def hf_cache_path(model_name: str) -> Path:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = "models--" + model_name.replace("/", "--")
    return cache_root / repo_dir


def summarize_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    lowered = text.lower()

    if "gatedrepoerror" in lowered or "is a gated repository" in lowered:
        return "gated model access not granted"
    if "401 client error" in lowered or "403 client error" in lowered:
        return "authentication or permission error"
    if "outofmemoryerror" in lowered or "cuda out of memory" in lowered:
        return "GPU out of memory"
    if "couldn't connect" in lowered or "failed to establish a new connection" in lowered:
        return "network error while downloading"
    if "repository not found" in lowered:
        return "model repository not found"
    return text.splitlines()[0][:300]


def unload_models():
    unload_local_models()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def check_one_config(config_path: str, do_generate: bool = True) -> dict:
    set_active_config(config_path)
    cfg = load_config(config_path)
    backend = cfg.get("backend", "local")
    model_name = cfg["answer_model"]
    quantization = cfg.get("quantization", "none")
    quant_label = quantization_label(cfg)
    cache_dir = hf_cache_path(model_name)

    result = {
        "config": Path(config_path).name,
        "config_path": str(Path(config_path).resolve()),
        "backend": backend,
        "model": model_name,
        "quantization": quant_label,
        "cache_present_before": cache_dir.exists(),
        "status": "unknown",
        "load_seconds": None,
        "generate_seconds": None,
        "device": None,
        "gpu_peak_gb": None,
        "error": None,
    }

    unload_models()

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if backend == "local":
            start = time.perf_counter()
            entry = _get_local_model(model_name, quantization=quantization)
            result["load_seconds"] = round(time.perf_counter() - start, 2)
            result["device"] = str(next(entry["model"].parameters()).device)

            if do_generate:
                messages = [
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Reply with the single word: ready"},
                ]
                start = time.perf_counter()
                text = _local_generate(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=8,
                    quantization=quantization,
                )
                result["generate_seconds"] = round(time.perf_counter() - start, 2)
                result["sample_output"] = text[:120]
        else:
            messages = [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with the single word: ready"},
            ]
            smoke_max_tokens = 16 if cfg.get("api_mode", "chat_completions") == "responses" else 8
            start = time.perf_counter()
            text = call_llm(
                messages=messages,
                model=model_name,
                temperature=0.0,
                max_tokens=smoke_max_tokens,
                use_cache=False,
                backend=backend,
                quantization=quantization,
                api_mode=cfg.get("api_mode", "chat_completions"),
                reasoning_effort=cfg.get("reasoning_effort_answer", cfg.get("reasoning_effort")),
            )
            result["generate_seconds"] = round(time.perf_counter() - start, 2)
            result["sample_output"] = text[:120]
            result["device"] = f"api:{cfg.get('api_mode', 'chat_completions')}"

        if torch.cuda.is_available():
            result["gpu_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)

        result["cache_present_after"] = cache_dir.exists()
        result["status"] = "ok"
        return result
    except Exception as exc:  # pragma: no cover - preflight diagnostics
        result["status"] = "failed"
        result["error"] = summarize_error(exc)
        result["error_type"] = exc.__class__.__name__
        result["cache_present_after"] = cache_dir.exists()
        if torch.cuda.is_available():
            result["gpu_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        return result
    finally:
        unload_models()


def run_preflight(config_paths: list[str], do_generate: bool = True) -> list[dict]:
    results = []
    for idx, config_path in enumerate(config_paths, start=1):
        logger.info("[%d/%d] Preflighting %s", idx, len(config_paths), Path(config_path).name)
        result = check_one_config(config_path, do_generate=do_generate)
        results.append(result)

        if result["status"] == "ok":
            logger.info(
                "PASS %s | load=%ss generate=%ss peak=%sGB",
                result["config"],
                result["load_seconds"],
                result["generate_seconds"],
                result["gpu_peak_gb"],
            )
        else:
            logger.error(
                "FAIL %s | %s (%s)",
                result["config"],
                result.get("error"),
                result.get("error_type"),
            )
    return results


def write_report(results: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_root": str(SRC_PROJECT_ROOT),
        "generated_at_epoch": time.time(),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_summary(results: list[dict]):
    print("\n" + "=" * 100)
    print("MODEL PREFLIGHT")
    print("=" * 100)
    for r in results:
        peak = f"{r['gpu_peak_gb']:.2f} GB" if r.get("gpu_peak_gb") is not None else "n/a"
        load_s = f"{r['load_seconds']:.2f}s" if r.get("load_seconds") is not None else "n/a"
        gen_s = f"{r['generate_seconds']:.2f}s" if r.get("generate_seconds") is not None else "n/a"
        if r["status"] == "ok":
            print(
                f"[PASS] {r['config']:<24} "
                f"load={load_s:<8} gen={gen_s:<8} device={r['device']:<10} peak={peak:<10}"
            )
        else:
            print(f"[FAIL] {r['config']:<24} {r.get('error')}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preflight model configs")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific config files to check (default: all in configs/models/)",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load models; skip the tiny generation step",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "outputs" / "comparison" / "model_preflight.json"),
        help="Where to save the JSON report",
    )
    args = parser.parse_args()

    config_paths = args.configs or discover_configs(PROJECT_ROOT / "configs" / "models")
    if not config_paths:
        raise SystemExit("No model config files found.")

    results = run_preflight(config_paths, do_generate=not args.load_only)
    write_report(results, Path(args.output))
    print_summary(results)
