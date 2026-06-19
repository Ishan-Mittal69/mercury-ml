"""
Own NMT model inference.
Loaded in a thread pool at startup so the event loop is never blocked.

To plug in real weights, implement _load() and translate() below.
CTranslate2 example is in the comments.
"""

import asyncio
import logging
import os

from app.config import settings
from app import metrics

logger = logging.getLogger(__name__)


class TranslationModel:
    _instance: "TranslationModel | None" = None

    def __init__(self) -> None:
        self._loaded = False
        # _ct2, _sp set by _load() when weights are available

    def _load(self, path: str) -> None:
        """
        Load a CTranslate2-converted opus-mt model produced by training/finetune_en_fr.py.

        To activate when ct2_model/ is ready:
          1. pip install ctranslate2 sentencepiece  (uncomment in requirements.txt)
          2. Remove the `raise` line at the bottom
          3. Uncomment the 5 lines above it
        """
        # import ctranslate2
        # import sentencepiece
        # self._ct2 = ctranslate2.Translator(path, device="auto", inter_threads=4, intra_threads=4)
        # self._sp_src = sentencepiece.SentencePieceProcessor(os.path.join(path, "source.spm"))
        # self._sp_tgt = sentencepiece.SentencePieceProcessor(os.path.join(path, "target.spm"))
        raise NotImplementedError("Model weights not yet available — uncomment _load() body")

    @classmethod
    def get(cls) -> "TranslationModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def load_async(cls) -> None:
        """Call once at startup — loads model in a thread pool, non-blocking."""
        instance = cls.get()
        if instance._loaded:
            return

        path = settings.MODEL_PATH
        if not path:
            logger.info("MODEL_PATH not set — Gemini handles all translations")
            metrics.model_loaded.set(0)
            return

        if not os.path.isdir(path):
            logger.warning("MODEL_PATH=%s does not exist — Gemini fallback active", path)
            metrics.model_loaded.set(0)
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, instance._load, path)
            instance._loaded = True
            metrics.model_loaded.set(1)
            logger.info("Model loaded from %s", path)
        except NotImplementedError:
            logger.info("Model stub active — implement _load() when weights are ready")
            metrics.model_loaded.set(0)
        except Exception as exc:
            logger.error("Model load failed: %s — Gemini fallback active", exc)
            metrics.model_loaded.set(0)

    def is_loaded(self) -> bool:
        return self._loaded

    def translate(self, text: str, source_lang: str, target_lang: str) -> tuple[str, float]:
        """
        Translate one segment. Returns (translated_text, confidence 0–1).
        Uncomment body when _load() is implemented.
        """
        # tokens = self._sp_src.encode(text, out_type=str)
        # result = self._ct2.translate_batch(
        #     [tokens],
        #     beam_size=4,
        #     max_decoding_length=256,
        # )[0]
        # decoded = self._sp_tgt.decode(result.hypotheses[0])
        # # CTranslate2 scores are log-probs (negative). Normalise to 0-1:
        # # score of -0.1 → ~0.9 confidence; score of -2.0 → ~0.1 confidence
        # import math
        # confidence = math.exp(result.scores[0] / max(len(result.hypotheses[0]), 1))
        # confidence = min(max(confidence, 0.0), 1.0)
        # return decoded, confidence
        raise NotImplementedError("translate() not implemented — uncomment body after _load()")
