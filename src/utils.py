"""Shared utilities: config loading, LLM wrapper with caching/retry, logging, token tracking, voting."""

import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from threading import Lock

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "models" / "gpt54_hybrid_api.yaml"
CONFIG_ENV_VAR = "MEDQUAD_CONFIG"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def resolve_config_path(path: str | os.PathLike | None = None) -> Path:
    if path:
        return Path(path).expanduser().resolve()

    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return DEFAULT_CONFIG_PATH.resolve()


def set_active_config(path: str | os.PathLike | None):
    if path is None:
        os.environ.pop(CONFIG_ENV_VAR, None)
        return
    os.environ[CONFIG_ENV_VAR] = str(Path(path).expanduser().resolve())


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = resolve_config_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quantization_label(cfg: dict) -> str:
    return str(cfg.get("display_quantization", cfg.get("quantization", "none")))


def _safe_cache_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(":", "_")


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

def _cache_key(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    backend: str = "openai",
    api_mode: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    blob = json.dumps({"model": model, "messages": messages,
                        "temperature": temperature, "max_tokens": max_tokens,
                        "backend": backend, "api_mode": api_mode,
                        "reasoning_effort": reasoning_effort},
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
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = json.load(f).get("content", "")
        except Exception:
            return None

        if isinstance(content, str) and content.strip():
            return content

        try:
            p.unlink()
        except OSError:
            pass
    return None


def _write_cache(key: str, content: str):
    p = _cache_dir() / f"{key}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"content": content}, f)


_client_cache: dict[tuple[str, str], OpenAI] = {}


def _get_client() -> OpenAI:
    cfg = load_config()
    base_url = str(cfg.get("base_url", "") or os.getenv("OPENAI_BASE_URL", "") or "").strip()

    api_key = None
    api_key_env = cfg.get("api_key_env")
    if api_key_env:
        api_key = os.getenv(str(api_key_env))
    if not api_key:
        api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key and base_url:
        api_key = "local-placeholder"

    client_key = (base_url, str(api_key or ""))
    client = _client_cache.get(client_key)
    if client is None:
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        _client_cache[client_key] = client
    return client


def _temperature_supported(model: str) -> bool:
    cfg = load_config()
    if "supports_temperature" in cfg:
        return bool(cfg["supports_temperature"])
    return not model.startswith("gpt-5.4")


def _api_messages(messages: list[dict]) -> list[dict]:
    """Normalize messages for current OpenAI APIs.

    GPT-5-family and newer APIs prefer `developer` messages in place of legacy
    `system` messages, so we translate those here while preserving user turns.
    """
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            role = "developer"
        normalized.append({"role": role, "content": msg.get("content", "")})
    return normalized


def _content_as_text_blocks(content) -> list[dict]:
    if isinstance(content, list):
        return content
    if content is None:
        content = ""
    return [{"type": "text", "text": str(content)}]


def _merge_message_content(left, right):
    if isinstance(left, list) or isinstance(right, list):
        merged = list(_content_as_text_blocks(left))
        if merged and right:
            merged.append({"type": "text", "text": "\n\n"})
        merged.extend(_content_as_text_blocks(right))
        return merged
    return f"{left or ''}\n\n{right or ''}".strip()


def _processor_messages(messages: list[dict]) -> list[dict]:
    normalized = []
    for msg in messages:
        normalized.append(
            {
                "role": msg.get("role", "user"),
                "content": _content_as_text_blocks(msg.get("content", "")),
                "tool_calls": msg.get("tool_calls", []),
                "tool_responses": msg.get("tool_responses", []),
            }
        )
    return normalized


def _responses_input(messages: list[dict]) -> list[dict]:
    return [
        {
            "role": msg["role"],
            "content": [{"type": "input_text", "text": msg["content"]}],
        }
        for msg in _api_messages(messages)
    ]


def _extract_response_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                chunks.append(getattr(content, "text", ""))
    return "".join(chunks).strip()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _api_call(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    api_mode: str = "chat_completions",
    reasoning_effort: str | None = None,
) -> tuple[str, int, int]:
    client = _get_client()
    if api_mode == "responses":
        max_output_tokens = max(16, int(max_tokens))
        request = {
            "model": model,
            "input": _responses_input(messages),
            "max_output_tokens": max_output_tokens,
        }
        if _temperature_supported(model):
            request["temperature"] = temperature
        if reasoning_effort:
            request["reasoning"] = {"effort": reasoning_effort}
        resp = client.responses.create(**request)
        content = _extract_response_text(resp)
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        return content, prompt_tokens, completion_tokens

    request = {
        "model": model,
        "messages": _api_messages(messages),
        "max_completion_tokens": max_tokens,
    }
    if _temperature_supported(model):
        request["temperature"] = temperature
    resp = client.chat.completions.create(**request)
    content = (resp.choices[0].message.content or "").strip()
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


def unload_local_models():
    """Drop cached local models and aggressively release CUDA memory."""
    import gc

    _local_models.clear()
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _fold_system_into_user(messages: list[dict]) -> list[dict]:
    """Fallback for chat templates that reject the system role."""
    chat_messages = list(messages)
    if chat_messages and chat_messages[0]["role"] == "system":
        system_content = chat_messages.pop(0)["content"]
        if chat_messages and chat_messages[0]["role"] == "user":
            chat_messages[0] = {
                "role": "user",
                "content": _merge_message_content(system_content, chat_messages[0]["content"]),
            }
        else:
            chat_messages.insert(0, {"role": "user", "content": system_content})
    return chat_messages


def _prefer_processor_chat_template(model_name: str) -> bool:
    cfg = load_config()
    if "use_processor_chat_template" in cfg:
        return bool(cfg["use_processor_chat_template"])
    return "gemma-4" in model_name.lower()


def _tokenizer_fix_dir() -> Path:
    root = PROJECT_ROOT / "data" / "outputs" / "cache" / "tokenizer_fixes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prepare_tokenizer_source(model_name: str) -> str:
    if "gemma-4" not in model_name.lower():
        return model_name

    try:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                repo_id=model_name,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                    "tokenizer.model",
                    "tokenizer.model.v3",
                ],
            )
        )
    except Exception:
        return model_name

    cfg_path = snapshot_path / "tokenizer_config.json"
    if not cfg_path.exists():
        return str(snapshot_path)

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return str(snapshot_path)

    extra_special_tokens = cfg.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return str(snapshot_path)

    fixed_dir = _tokenizer_fix_dir() / _safe_cache_name(model_name)
    fixed_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tokenizer.model",
        "tokenizer.model.v3",
    ]:
        src = snapshot_path / name
        if src.exists():
            shutil.copy2(src, fixed_dir / name)

    cfg["extra_special_tokens"] = {
        ("video_token" if idx == 0 else f"extra_special_token_{idx}"): token
        for idx, token in enumerate(extra_special_tokens)
    }
    (fixed_dir / "tokenizer_config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )
    return str(fixed_dir)


def _get_local_model(model_name: str, quantization: str = "none"):
    """Lazy-load a HuggingFace model + tokenizer, cached by model name.

    Supports both causal (GPT-style) and seq2seq (T5-style) models.
    When a CUDA GPU is available the model is loaded in float16 with
    automatic device placement; otherwise it falls back to CPU.

    Args:
        model_name: HuggingFace model identifier.
        quantization: "none", "4bit", or "8bit". Requires bitsandbytes.
    """
    use_processor_chat_template = _prefer_processor_chat_template(model_name)
    cache_key = f"{model_name}::{quantization}::processor={use_processor_chat_template}"
    if cache_key not in _local_models:
        os.environ["USE_TF"] = "0"
        os.environ["USE_TORCH"] = "1"
        _disable_tensorflow()
        import torch
        from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

        logger = logging.getLogger("medquad")
        is_seq2seq = "t5" in model_name.lower()
        kind = "seq2seq" if is_seq2seq else "causal"
        use_gpu = torch.cuda.is_available()
        logger.info("Loading local model: %s (%s, gpu=%s, quant=%s)",
                     model_name, kind, use_gpu, quantization)

        tokenizer_source = _prepare_tokenizer_source(model_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        processor = None
        if use_processor_chat_template:
            try:
                processor = AutoProcessor.from_pretrained(model_name)
            except Exception:
                processor = None

        cfg = load_config()
        load_kwargs: dict = {}
        if use_gpu:
            load_kwargs["device_map"] = "auto"

        if quantization in ("4bit", "8bit") and use_gpu:
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=bool(cfg.get("cpu_offload", False)),
                )
                if cfg.get("cpu_offload"):
                    load_kwargs["offload_folder"] = str(PROJECT_ROOT / "data" / "outputs" / "cache" / "offload")
        elif use_gpu:
            # Let each model use its preferred precision instead of forcing fp16.
            # This avoids dtype mismatches for repos such as GPT-OSS that default
            # to bf16 or mixed low-precision kernels on newer GPUs.
            load_kwargs["dtype"] = "auto"

        if is_seq2seq:
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
        else:
            model_obj = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        model_obj.eval()

        has_chat_template = getattr(tokenizer, "chat_template", None) is not None

        if use_gpu:
            mem = torch.cuda.memory_allocated() / 1024**3
            logger.info("GPU memory used after load: %.2f GB", mem)

        _local_models[cache_key] = {
            "tokenizer": tokenizer,
            "processor": processor,
            "model": model_obj,
            "is_seq2seq": is_seq2seq,
            "has_chat_template": has_chat_template,
        }
    return _local_models[cache_key]


def _local_generate(model: str, messages: list, temperature: float, max_tokens: int,
                     quantization: str = "none") -> str:
    """Generate text using a local HuggingFace model (GPU-accelerated when available)."""
    import torch

    entry = _get_local_model(model, quantization=quantization)
    tokenizer = entry["tokenizer"]
    processor = entry.get("processor")
    model_obj = entry["model"]
    is_seq2seq = entry["is_seq2seq"]
    has_chat_template = entry["has_chat_template"]

    # Build input ids — use chat template when the tokenizer supports it
    if has_chat_template and not is_seq2seq:
        # Some chat templates reject the system role. Fold it into the first
        # user turn when the template metadata or runtime error says to.
        chat_messages = list(messages)
        _tpl = tokenizer.chat_template or ""
        if chat_messages and chat_messages[0]["role"] == "system" and "system" not in _tpl:
            chat_messages = _fold_system_into_user(chat_messages)

        try:
            input_ids = tokenizer.apply_chat_template(
                chat_messages, return_tensors="pt", add_generation_prompt=True,
            )
        except Exception as exc:
            if "system role not supported" not in str(exc).lower():
                raise
            input_ids = tokenizer.apply_chat_template(
                _fold_system_into_user(messages),
                return_tensors="pt",
                add_generation_prompt=True,
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


def _local_generate(model: str, messages: list, temperature: float, max_tokens: int,
                     quantization: str = "none") -> str:
    """Generate text using a local HuggingFace model, including processor-backed chat models."""
    import torch

    entry = _get_local_model(model, quantization=quantization)
    tokenizer = entry["tokenizer"]
    processor = entry.get("processor")
    model_obj = entry["model"]
    is_seq2seq = entry["is_seq2seq"]
    has_chat_template = entry["has_chat_template"]

    if processor is not None and hasattr(processor, "apply_chat_template") and not is_seq2seq:
        try:
            model_inputs = processor.apply_chat_template(
                _processor_messages(messages),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception as exc:
            if "system role not supported" not in str(exc).lower():
                raise
            model_inputs = processor.apply_chat_template(
                _processor_messages(_fold_system_into_user(messages)),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        device = next(model_obj.parameters()).device
        model_inputs = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in model_inputs.items()
        }
        if "attention_mask" not in model_inputs and "input_ids" in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
        prompt_len = model_inputs["input_ids"].shape[1]
        decoder = getattr(processor, "tokenizer", processor)
    elif has_chat_template and not is_seq2seq:
        chat_messages = list(messages)
        template = tokenizer.chat_template or ""
        if chat_messages and chat_messages[0]["role"] == "system" and "system" not in template:
            chat_messages = _fold_system_into_user(chat_messages)

        try:
            tokenized_chat = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception as exc:
            if "system role not supported" not in str(exc).lower():
                raise
            tokenized_chat = tokenizer.apply_chat_template(
                _fold_system_into_user(messages),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        device = next(model_obj.parameters()).device
        if hasattr(tokenized_chat, "items"):
            model_inputs = {
                key: (value.to(device) if hasattr(value, "to") else value)
                for key, value in tokenized_chat.items()
            }
            input_ids = model_inputs["input_ids"]
        else:
            input_ids = tokenized_chat.to(device)
            model_inputs = {
                "input_ids": input_ids,
            }
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]
        decoder = tokenizer
    else:
        prompt = "\n\n".join(str(m["content"]) for m in messages)
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = next(model_obj.parameters()).device
        model_inputs = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in tokenized.items()
        }
        if "attention_mask" not in model_inputs and "input_ids" in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
        prompt_len = model_inputs["input_ids"].shape[1]
        decoder = tokenizer

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        repetition_penalty=1.2,
    )
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-4)
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = model_obj.generate(**model_inputs, **gen_kwargs)

    if is_seq2seq:
        text = decoder.decode(output_ids[0], skip_special_tokens=True)
    else:
        new_tokens = output_ids[0][prompt_len:]
        text = decoder.decode(new_tokens, skip_special_tokens=True)

    return text.strip()


# ---------------------------------------------------------------------------
# Unified call_llm
# ---------------------------------------------------------------------------

def _pick_most_consistent(candidates: list[str]) -> str:
    """Pick the candidate most similar to all others (self-consistency voting).

    Uses pairwise character-level similarity (SequenceMatcher) to find the
    response that best represents the consensus.  Fast enough for <=5 candidates.
    """
    if len(candidates) == 1:
        return candidates[0]

    from difflib import SequenceMatcher
    best_idx, best_score = 0, -1.0
    for i, a in enumerate(candidates):
        total = sum(SequenceMatcher(None, a, b).ratio() for j, b in enumerate(candidates) if j != i)
        if total > best_score:
            best_score = total
            best_idx = i
    return candidates[best_idx]


def _majority_vote_int(candidates: list[int]) -> int:
    """Return the most common integer; ties broken by lowest value."""
    counts = Counter(candidates)
    max_count = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == max_count)
    return winners[0]


def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 300,
    use_cache: bool = True,
    backend: str = "openai",
    quantization: str = "none",
    n_votes: int = 1,
    api_mode: str = "chat_completions",
    reasoning_effort: str | None = None,
) -> str:
    """Route to OpenAI API or local HuggingFace model based on *backend*.

    When *n_votes* > 1, generate multiple responses and return the most
    consistent one (self-consistency voting).  The winning response is cached
    under the original cache key so subsequent calls are instant.
    """
    key = _cache_key(
        model,
        messages,
        temperature,
        max_tokens,
        backend=backend,
        api_mode=api_mode,
        reasoning_effort=reasoning_effort,
    )

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    if n_votes <= 1:
        # Single-shot (original behavior)
        if backend == "local":
            content = _local_generate(model, messages, temperature, max_tokens,
                                      quantization=quantization)
        else:
            content, pt, ct = _api_call(
                model,
                messages,
                temperature,
                max_tokens,
                api_mode=api_mode,
                reasoning_effort=reasoning_effort,
            )
            _global_tracker.add(pt, ct)
    else:
        # Multi-vote: generate n_votes responses, pick best via self-consistency
        logger = logging.getLogger("medquad")
        candidates = []
        # Use slightly elevated temperature for diversity if temp is near 0
        vote_temp = max(temperature, 0.3)
        for v in range(n_votes):
            if backend == "local":
                resp = _local_generate(model, messages, vote_temp, max_tokens,
                                       quantization=quantization)
            else:
                resp, pt, ct = _api_call(
                    model,
                    messages,
                    vote_temp,
                    max_tokens,
                    api_mode=api_mode,
                    reasoning_effort=reasoning_effort,
                )
                _global_tracker.add(pt, ct)
            candidates.append(resp)
        content = _pick_most_consistent(candidates)
        logger.info("Voting (%d candidates) → picked response (%.0f%% avg similarity)",
                     n_votes,
                     sum(1 for c in candidates if c == content) / n_votes * 100)

    if use_cache and content.strip():
        _write_cache(key, content)

    return content
