"""
Gemini 3 Flash — backup translator.
Uses a shared httpx.AsyncClient (connection pool) injected at startup.
Retries on 429 / 5xx with exponential backoff via tenacity.
"""

import json
import logging
from typing import Any

import httpx
import tenacity

from app.config import settings
from app import metrics

logger = logging.getLogger(__name__)

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={key}"
)


class GeminiError(RuntimeError):
    pass


class GeminiRateLimitError(GeminiError):
    pass


def _build_prompt(segments: list[dict[str, str]], source_lang: str, target_lang: str) -> str:
    return f"""Translate the following segments from {source_lang} to {target_lang}.

Rules:
- Return a JSON array with exactly the same number of objects: [{{"id": "...", "text": "..."}}]
- Preserve all {{N}} placeholders (e.g. {{1}}, {{2}}) exactly as-is
- Translate only the surrounding human-readable text
- Do not add explanations or extra fields

Segments:
{json.dumps(segments, ensure_ascii=False, indent=2)}"""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, GeminiRateLimitError):
        return True
    if isinstance(exc, GeminiError) and "5" in str(exc)[:3]:
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    return False


@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_exponential(multiplier=2, min=2, max=30),
    stop=tenacity.stop_after_attempt(settings.GEMINI_MAX_RETRIES),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def translate_batch(
    http: httpx.AsyncClient,
    segments: list[dict[str, str]],
    source_lang: str,
    target_lang: str,
) -> list[dict[str, str]]:
    """
    Translate a batch of segments via Gemini 3 Flash.
    `http` must be the shared AsyncClient injected from app.state.
    """
    if not settings.GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is not set")

    if not segments:
        return []

    url = _GEMINI_URL.format(model=settings.GEMINI_MODEL, key=settings.GEMINI_API_KEY)
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": _build_prompt(segments, source_lang, target_lang)}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }

    try:
        resp = await http.post(url, json=payload, timeout=settings.GEMINI_TIMEOUT)
    except httpx.TimeoutException as e:
        metrics.gemini_requests_total.labels(outcome="error").inc()
        raise GeminiError(f"Gemini request timed out: {e}") from e

    if resp.status_code == 429:
        metrics.gemini_requests_total.labels(outcome="retry").inc()
        raise GeminiRateLimitError("Gemini rate limited (429)")

    if resp.status_code >= 500:
        metrics.gemini_requests_total.labels(outcome="retry").inc()
        raise GeminiError(f"Gemini server error {resp.status_code}: {resp.text[:300]}")

    if resp.status_code != 200:
        metrics.gemini_requests_total.labels(outcome="error").inc()
        raise GeminiError(f"Gemini API {resp.status_code}: {resp.text[:300]}")

    metrics.gemini_requests_total.labels(outcome="success").inc()

    data = resp.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw)
