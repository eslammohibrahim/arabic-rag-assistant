"""Minimal client for a local Ollama server (https://ollama.com)."""

import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")


def generate(prompt: str, system: str = "", model: str = MODEL, max_tokens: int = 80) -> str:
    """Greedy decoding (temperature 0) so results are reproducible."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_predict": max_tokens, "num_ctx": 4096},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def is_available(model: str = MODEL) -> bool:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        return any(m["name"] == model or m["name"] == f"{model}:latest" for m in tags.get("models", []))
    except requests.RequestException:
        return False
