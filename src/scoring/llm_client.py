"""OpenAI LLM client — extraction, scoring, tool-calling. OpenAI only.

Uses a thread-safe singleton client so TCP+TLS connections are reused
across the dozens of API calls a single run makes.
"""

import json
import re
import threading
from typing import Optional

from src.config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_TEMPERATURE,
    EXTRACTION_MODEL, SCORING_MODEL,
)

_client = None
_client_lock = threading.Lock()


def get_openai_client():
    """Return a shared OpenAI client (thread-safe lazy singleton)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from openai import OpenAI
                _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def call_openai(prompt: str, model: str, max_tokens: int = 1000, system: str = "") -> Optional[str]:
    client = get_openai_client()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=msgs,
        max_tokens=max_tokens, temperature=OPENAI_TEMPERATURE,
    )
    return resp.choices[0].message.content


def call_extraction_llm(prompt: str, max_tokens: int = 800, system: str = "") -> Optional[str]:
    if LLM_PROVIDER == "openai":
        return call_openai(prompt, EXTRACTION_MODEL, max_tokens, system)
    return None


def call_scoring_llm(prompt: str, max_tokens: int = 1000, system: str = "") -> Optional[str]:
    if LLM_PROVIDER == "openai":
        return call_openai(prompt, SCORING_MODEL, max_tokens, system)
    return None


def parse_json(raw: Optional[str]) -> Optional[dict]:
    """Strip markdown fences and parse JSON.

    Handles truncated responses by closing open brackets/braces so that
    partially-generated JSON (from hitting max_tokens) still parses.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

    # Happy path
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Truncation recovery: strip trailing partial value, then close open brackets
    # Remove a trailing incomplete string/value (e.g. `"evidence": "some text`)
    truncated = re.sub(r',\s*"[^"]*"?\s*:\s*"?[^"{}[\]]*$', '', cleaned)
    # Remove a trailing comma left over
    truncated = re.sub(r',\s*$', '', truncated)
    # Count open vs close for [] and {}
    open_sq = truncated.count('[') - truncated.count(']')
    open_br = truncated.count('{') - truncated.count('}')
    truncated += ']' * max(0, open_sq) + '}' * max(0, open_br)
    try:
        return json.loads(truncated)
    except Exception:
        return None


def has_llm() -> bool:
    return LLM_PROVIDER == "openai"
