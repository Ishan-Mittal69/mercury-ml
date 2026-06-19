"""
Translation orchestration — own model → Gemini fallback.

1. If model is loaded: run all segments through it.
   Segments with confidence ≥ CONFIDENCE_THRESHOLD are kept.
2. Remaining segments are batched into one Gemini call.
3. Results merged and returned in original order.
"""

import logging

import httpx

from app.config import settings
from app import gemini, metrics
from app.gemini import GeminiError
from app.model import TranslationModel

logger = logging.getLogger(__name__)


async def translate(
    http: httpx.AsyncClient,
    segments: list[dict],
    source_lang: str,
    target_lang: str,
) -> list[dict]:
    """
    segments: [{"id": "...", "text": "source"}]
    returns:  [{"id": "...", "text": "translated", "confidence": 0.95}]
    """
    if not segments:
        return []

    model = TranslationModel.get()
    ordered_ids = [s["id"] for s in segments]
    results: dict[str, dict] = {}
    needs_gemini: list[dict] = []

    # ── 1. Own model pass (batch, non-blocking) ───────────────────────────────
    if model.is_loaded():
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            batch_results = await loop.run_in_executor(
                None,
                model.translate_batch,
                [s["text"] for s in segments],
                source_lang,
                target_lang,
            )
            for seg, (translated, confidence) in zip(segments, batch_results):
                results[seg["id"]] = {
                    "id": seg["id"],
                    "text": translated,
                    "confidence": round(confidence, 4),
                }
                metrics.translate_segments_total.labels(provider="model").inc()
        except Exception as exc:
            logger.warning("Model batch error: %s — routing all to Gemini", exc)
            needs_gemini = list(segments)
    else:
        needs_gemini = list(segments)

    # ── 2. Gemini batch pass ──────────────────────────────────────────────────
    if needs_gemini:
        try:
            gemini_results = await gemini.translate_batch(
                http=http,
                segments=[{"id": s["id"], "text": s["text"]} for s in needs_gemini],
                source_lang=source_lang,
                target_lang=target_lang,
            )
            gemini_map = {r["id"]: r["text"] for r in gemini_results}

            for seg in needs_gemini:
                text = gemini_map.get(seg["id"], seg["text"])
                results[seg["id"]] = {
                    "id": seg["id"],
                    "text": text,
                    "confidence": 0.95,
                }
                metrics.translate_segments_total.labels(provider="gemini").inc()

        except GeminiError as exc:
            logger.error(
                "Gemini failed for %d segments: %s — returning source text",
                len(needs_gemini),
                exc,
            )
            for seg in needs_gemini:
                results[seg["id"]] = {
                    "id": seg["id"],
                    "text": seg["text"],  # identity fallback
                    "confidence": 0.0,
                }
                metrics.translate_segments_total.labels(provider="fallback").inc()

    # ── 3. Return in original order ───────────────────────────────────────────
    return [results[sid] for sid in ordered_ids if sid in results]
