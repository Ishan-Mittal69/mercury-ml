"""
Translation orchestration:
  1. Run own model on all segments → collect high-confidence results
  2. Batch the low-confidence / unhandled segments → send to Gemini
  3. Merge and return in original order

When own model weights arrive and `model.is_loaded()` returns True,
Gemini only handles segments below CONFIDENCE_THRESHOLD.
Until then, Gemini handles everything.
"""

import logging
from app import config
from app.model import TranslationModel
from app import gemini

logger = logging.getLogger(__name__)


async def translate(
    segments: list[dict[str, str]],
    source_lang: str,
    target_lang: str,
) -> list[dict[str, str, float]]:
    """
    segments: [{"id": "...", "text": "source"}]
    returns:  [{"id": "...", "text": "translated", "confidence": 0.95}]
    """
    if not segments:
        return []

    model = TranslationModel.get()
    results: dict[str, dict] = {}   # id → {id, text, confidence}
    needs_gemini: list[dict[str, str]] = []

    # ── 1. Own model pass ─────────────────────────────────────────────────────
    if model.is_loaded():
        for seg in segments:
            try:
                translated, confidence = model.translate(
                    seg["text"], source_lang, target_lang
                )
                if confidence >= config.CONFIDENCE_THRESHOLD:
                    results[seg["id"]] = {
                        "id": seg["id"],
                        "text": translated,
                        "confidence": round(confidence, 4),
                    }
                else:
                    needs_gemini.append(seg)
            except Exception as exc:
                logger.warning("Model error for segment %s: %s — falling back to Gemini", seg["id"], exc)
                needs_gemini.append(seg)
    else:
        needs_gemini = segments

    # ── 2. Gemini pass (batch) ────────────────────────────────────────────────
    if needs_gemini:
        try:
            gemini_results = await gemini.translate_batch(
                [{"id": s["id"], "text": s["text"]} for s in needs_gemini],
                source_lang,
                target_lang,
            )
            gemini_map = {r["id"]: r["text"] for r in gemini_results}

            for seg in needs_gemini:
                text = gemini_map.get(seg["id"], seg["text"])  # fallback to source on miss
                results[seg["id"]] = {
                    "id": seg["id"],
                    "text": text,
                    "confidence": 0.95,  # Gemini is high quality; Mercury won't re-fallback
                }
        except gemini.GeminiError as exc:
            logger.error("Gemini failed: %s — returning source text for %d segments", exc, len(needs_gemini))
            for seg in needs_gemini:
                results[seg["id"]] = {
                    "id": seg["id"],
                    "text": seg["text"],  # identity fallback
                    "confidence": 0.0,
                }

    # ── 3. Return in original order ───────────────────────────────────────────
    return [results[seg["id"]] for seg in segments if seg["id"] in results]
