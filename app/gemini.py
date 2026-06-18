"""
Gemini 3 Flash — backup translator.
Called when own model is not loaded or returns confidence < CONFIDENCE_THRESHOLD.
Batches all low-confidence segments into a single API call.
"""

import json
import logging
import asyncio
from typing import Any
import httpx
from app import config

logger = logging.getLogger(__name__)

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={key}"
)


class GeminiError(RuntimeError):
    pass


def _build_prompt(
    segments: list[dict[str, str]],
    source_lang: str,
    target_lang: str,
) -> str:
    return f"""Translate the following segments from {source_lang} to {target_lang}.

Rules:
- Return a JSON array with exactly the same number of objects: [{{"id": "...", "text": "..."}}]
- Preserve all {{N}} placeholders (e.g. {{1}}, {{2}}) exactly as-is
- Translate only the surrounding human-readable text
- Do not add explanations or extra fields

Segments:
{json.dumps(segments, ensure_ascii=False, indent=2)}"""


async def translate_batch(
    segments: list[dict[str, str]],
    source_lang: str,
    target_lang: str,
    *,
    timeout: float = 60.0,
) -> list[dict[str, str]]:
    """
    Translate a batch of segments via Gemini 3 Flash.

    segments: [{"id": "...", "text": "source text"}, ...]
    Returns:  [{"id": "...", "text": "translated text"}, ...]
    """
    if not config.GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is not set")

    if not segments:
        return []

    url = _GEMINI_URL.format(model=config.GEMINI_MODEL, key=config.GEMINI_API_KEY)
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": _build_prompt(segments, source_lang, target_lang)}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code != 200:
        raise GeminiError(f"Gemini API {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"]
    results: list[dict[str, str]] = json.loads(raw)

    # Index by id so callers can look up quickly
    return results


def translate_batch_sync(
    segments: list[dict[str, str]],
    source_lang: str,
    target_lang: str,
) -> list[dict[str, str]]:
    return asyncio.run(translate_batch(segments, source_lang, target_lang))
