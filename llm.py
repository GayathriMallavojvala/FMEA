"""
Provider-agnostic LLM client for any OpenAI-compatible chat endpoint.

Public cloud (Groq, default), OpenAI, or a PRIVATE on-prem server (Ollama, vLLM, LM Studio,
llama.cpp server) so a plant's data never has to leave its network:

    LLM_BASE_URL  default https://api.groq.com/openai/v1
                  e.g. http://localhost:11434/v1  (Ollama),  http://gpu-box:8000/v1  (vLLM)
    LLM_API_KEY   falls back to GROQ_API_KEY; may be any string for local servers
    LLM_MODEL     default openai/gpt-oss-20b   (e.g. llama3.1:8b for Ollama)

Exposes `client.chat.completions.create(...)` with the same call shape the app already uses.
"""
import os
import time
from types import SimpleNamespace

import httpx

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")


class LLMError(RuntimeError):
    pass


class _Completions:
    def __init__(self, base_url, api_key, timeout):
        self.base_url, self.api_key, self.timeout = base_url, api_key, timeout

    def create(self, model, messages, temperature=0.3, max_tokens=1200, retries=0, **extra):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **extra}
        last = None
        for attempt in range(retries + 1):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=body,
                               headers=headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(1.5 * (attempt + 1)); continue
                if r.status_code >= 400:
                    raise LLMError(f"LLM server returned {r.status_code}: {r.text[:300]}")
                text = r.json()["choices"][0]["message"]["content"]
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
            except httpx.HTTPError as e:
                last = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1)); continue
        raise LLMError(f"Could not reach LLM server at {self.base_url}: {last}")


class Client:
    def __init__(self, base_url=BASE_URL, api_key=None, timeout=30.0):
        self.chat = SimpleNamespace(completions=_Completions(base_url, api_key, timeout))


_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        local = BASE_URL.startswith(("http://localhost", "http://127.", "http://192.168.", "http://10."))
        if not key and not local:
            raise LLMError("No API key: set LLM_API_KEY (or GROQ_API_KEY), or point LLM_BASE_URL at a local server.")
        _client = Client(api_key=key)
    return _client


def describe():
    return f"{BASE_URL} / {MODEL}"
