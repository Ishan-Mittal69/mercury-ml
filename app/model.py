"""
Own NMT model inference.

Stub now — returns (source_text, 0.0) so every segment falls back to Gemini.
When weights are ready, implement the body of TranslationModel.translate():

  CTranslate2 (fastest CPU/GPU inference for OPUS-MT / NLLB):
    import ctranslate2, sentencepiece
    self._ct2 = ctranslate2.Translator(MODEL_PATH, device="auto", inter_threads=4)
    self._sp  = sentencepiece.SentencePieceProcessor(MODEL_PATH + "/source.spm")

  Triton (GPU cluster):
    import tritonclient.http as triton
    self._client = triton.InferenceServerClient(url=TRITON_URL)
"""

import logging
import os
from app import config

logger = logging.getLogger(__name__)


class TranslationModel:
    _instance: "TranslationModel | None" = None

    def __init__(self) -> None:
        self._loaded = False

        if not config.MODEL_PATH:
            logger.info("MODEL_PATH not set — Gemini handles all translations until model is ready")
            return

        if not os.path.isdir(config.MODEL_PATH):
            logger.warning("MODEL_PATH %s does not exist — falling back to Gemini", config.MODEL_PATH)
            return

        try:
            self._load(config.MODEL_PATH)
            self._loaded = True
            logger.info("Model loaded from %s", config.MODEL_PATH)
        except Exception as exc:
            logger.error("Model load failed: %s — falling back to Gemini", exc)

    def _load(self, path: str) -> None:
        """
        Replace this with real loading when weights arrive.

        Example for CTranslate2 + SentencePiece:
            import ctranslate2, sentencepiece
            self._ct2 = ctranslate2.Translator(path, device="auto", inter_threads=4)
            self._sp  = sentencepiece.SentencePieceProcessor(path + "/source.spm")
        """
        raise NotImplementedError("Model weights not yet available")

    @classmethod
    def get(cls) -> "TranslationModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_loaded(self) -> bool:
        return self._loaded

    def translate(self, text: str, source_lang: str, target_lang: str) -> tuple[str, float]:
        """
        Translate one segment. Returns (translated_text, confidence 0-1).

        Replace with real inference when weights are ready:
            tokens = self._sp.encode(text, out_type=str)
            result = self._ct2.translate_batch(
                [tokens],
                target_prefix=[[f">>>{target_lang}<<<"]],  # NLLB / OPUS format
                max_decoding_length=512,
                beam_size=4,
            )[0]
            decoded = self._sp.decode(result.hypotheses[0])
            confidence = float(result.scores[0])   # log-prob, normalise if needed
            return decoded, min(max(confidence, 0.0), 1.0)
        """
        # Stub: identity translation, zero confidence → Gemini fallback always wins
        return text, 0.0
