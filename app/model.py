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

    # NLLB language codes — set per language pair at translate() time
    NLLB_LANG = {
        "EN": "eng_Latn", "FR": "fra_Latn", "DE": "deu_Latn",
        "IT": "ita_Latn", "ES": "spa_Latn", "NL": "nld_Latn",
        "PT": "por_Latn", "PL": "pol_Latn", "RU": "rus_Cyrl",
        "RO": "ron_Latn", "TR": "tur_Latn", "NO": "nob_Latn",
        "SV": "swe_Latn", "DA": "dan_Latn", "JA": "jpn_Jpan",
        "KO": "kor_Hang",
    }

    def _load(self, path: str) -> None:
        """
        Load NLLB-200-1.3B+QLoRA converted to CTranslate2 INT8.
        Produced by: finetune_nllb_qlora.py → ct2-nllb-converter --quantization int8

        To activate when ct2_model/ is ready:
          1. pip install ctranslate2 sentencepiece  (uncomment in requirements.txt)
          2. Remove the `raise` line at the bottom
          3. Uncomment the 4 lines above it
        """
        # import ctranslate2
        # import sentencepiece
        # self._ct2 = ctranslate2.Translator(path, device="auto", inter_threads=4, intra_threads=4)
        # self._sp  = sentencepiece.SentencePieceProcessor(os.path.join(path, "sentencepiece.bpe.model"))
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
        Translate one segment with NLLB. Returns (translated_text, confidence 0-1).
        NLLB requires the target language token as forced_bos_token_id.
        Uncomment body when _load() is implemented.
        """
        # import math
        # src_code = self.NLLB_LANG.get(source_lang.upper(), "eng_Latn")
        # tgt_code = self.NLLB_LANG.get(target_lang.upper(), "fra_Latn")
        #
        # # Tokenise with source language prefix
        # tokens = self._sp.encode(text, out_type=str)
        # src_tokens = [src_code] + tokens   # NLLB prepends language token
        #
        # # Translate — force target language as first output token
        # tgt_bos_id = self._sp.piece_to_id(tgt_code)
        # result = self._ct2.translate_batch(
        #     [src_tokens],
        #     target_prefix=[[tgt_code]],   # forced BOS = target language token
        #     beam_size=4,
        #     max_decoding_length=256,
        # )[0]
        #
        # # Decode (skip the language token prefix)
        # decoded = self._sp.decode(result.hypotheses[0][1:])  # drop the tgt_code token
        #
        # # Normalise log-prob score to 0–1 confidence
        # confidence = math.exp(result.scores[0] / max(len(result.hypotheses[0]), 1))
        # return decoded, min(max(confidence, 0.0), 1.0)
        raise NotImplementedError("translate() not implemented — uncomment body after _load()")
