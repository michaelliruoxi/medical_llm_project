"""Check Hugging Face auth and gated-model access from the project environment.

Usage:
    & 'C:\\Users\\Owner\\AppData\\Local\\Programs\\Python\\Python311\\python.exe' scripts\\check_hf_access.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, whoami


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODELS = [
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
]


def main() -> None:
    token = os.getenv("HF_TOKEN")
    print("HF_TOKEN present:", bool(token))

    if not token:
        print("No HF_TOKEN found in environment or .env")
        return

    try:
        identity = whoami(token=token)
        print("whoami: ok")
        print(json.dumps(identity, indent=2))
    except Exception as exc:
        print("whoami: failed")
        print(f"{type(exc).__name__}: {exc}")
        return

    for repo_id in MODELS:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                token=token,
            )
            print(f"{repo_id}: ok")
            print(f"  config cached at: {path}")
        except Exception as exc:
            print(f"{repo_id}: failed")
            print(f"  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
