"""Shared utilities: config loading, LLM wrapper with caching/retry, logging, token tracking."""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from threading import Lock

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | None = None) -> dict:
    path = path or str(PROJECT_ROOT / "configs" / "experiment.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: str | None = None) -> dict:
    path = path or str(PROJECT_ROOT / "configs" / "prompts.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("medquad")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Token-usage tracker
# ---------------------------------------------------------------------------

class TokenTracker:
    """Thread-safe accumulator for prompt / completion token counts."""

    def __init__(self):
        self._lock = Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


_global_tracker = TokenTracker()


def get_token_tracker() -> TokenTracker:
    return _global_tracker


# ---------------------------------------------------------------------------
# LLM wrapper with caching + retry
# ---------------------------------------------------------------------------

def _cache_key(model: str, messages: list, temperature: float, max_tokens: int) -> str:
    blob = json.dumps({"model": model, "messages": messages,
                        "temperature": temperature, "max_tokens": max_tokens},
                       sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_dir() -> Path:
    cfg = load_config()
    d = PROJECT_ROOT / cfg["paths"]["cache"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_cache(key: str) -> str | None:
    p = _cache_dir() / f"{key}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)["content"]
    return None


def _write_cache(key: str, content: str):
    p = _cache_dir() / f"{key}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"content": content}, f)


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _api_call(model: str, messages: list, temperature: float, max_tokens: int) -> tuple[str, int, int]:
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content.strip()
    usage = resp.usage
    return content, usage.prompt_tokens, usage.completion_tokens


# ---------------------------------------------------------------------------
# Local model backend (HuggingFace transformers)
# ---------------------------------------------------------------------------

_local_models: dict = {}


def _disable_tensorflow():
    """Replace TensorFlow in sys.modules with a stub before transformers touches it."""
    import types
    import importlib.machinery
    stub = types.ModuleType("tensorflow")
    stub.__version__ = "0.0.0"
    stub.__path__ = []
    stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
    sys.modules["tensorflow"] = stub


def _get_local_model(model_name: str):
    """Lazy-load a HuggingFace model + tokenizer, cached by model name.

    Supports both causal (GPT-style) and seq2seq (T5-style) models.
    When a CUDA GPU is available the model is loaded in float16 with
    automatic device placement; otherwise it falls back to CPU.
    """
    if model_name not in _local_models:
        os.environ["USE_TF"] = "0"
        os.environ["USE_TORCH"] = "1"
        _disable_tensorflow()
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

        logger = logging.getLogger("medquad")
        is_seq2seq = "t5" in model_name.lower()
        kind = "seq2seq" if is_seq2seq else "causal"
        use_gpu = torch.cuda.is_available()
        logger.info("Loading local model: %s (%s, gpu=%s)", model_name, kind, use_gpu)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict = {}
        if use_gpu:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"

        if is_seq2seq:
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
        else:
            model_obj = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        model_obj.eval()

        has_chat_template = getattr(tokenizer, "chat_template", None) is not None

        if use_gpu:
            mem = torch.cuda.memory_allocated() / 1024**3
            logger.info("GPU memory used after load: %.2f GB", mem)

        _local_models[model_name] = {
            "tokenizer": tokenizer,
            "model": model_obj,
            "is_seq2seq": is_seq2seq,
            "has_chat_template": has_chat_template,
        }
    return _local_models[model_name]


def _local_generate(model: str, messages: list, temperature: float, max_tokens: int) -> str:
    """Generate text using a local HuggingFace model (GPU-accelerated when available)."""
    import torch

    entry = _get_local_model(model)
    tokenizer = entry["tokenizer"]
    model_obj = entry["model"]
    is_seq2seq = entry["is_seq2seq"]
    has_chat_template = entry["has_chat_template"]

    # Build input ids — use chat template when the tokenizer supports it
    if has_chat_template and not is_seq2seq:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
    else:
        prompt = "\n\n".join(m["content"] for m in messages)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=2048)["input_ids"]

    device = next(model_obj.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        repetition_penalty=1.2,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-4)
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = model_obj.generate(input_ids, **gen_kwargs)

    if is_seq2seq:
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        new_tokens = output_ids[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return text.strip()


# ---------------------------------------------------------------------------
# Unified call_llm
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 300,
    use_cache: bool = True,
    backend: str = "openai",
) -> str:
    """Route to OpenAI API or local HuggingFace model based on *backend*."""
    key = _cache_key(model, messages, temperature, max_tokens)

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    if backend == "local":
        content = _local_generate(model, messages, temperature, max_tokens)
    else:
        content, pt, ct = _api_call(model, messages, temperature, max_tokens)
        _global_tracker.add(pt, ct)

    if use_cache:
        _write_cache(key, content)

    return content
