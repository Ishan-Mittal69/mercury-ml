"""
M2M100 fine-tuned model inference (4-bit NF4 bitsandbytes, loaded from MODEL_PATH).
Falls back to Gemini when model is not loaded.
"""

import asyncio
import logging
import math
import os
from typing import Optional

from app.config import settings
from app import metrics

logger = logging.getLogger(__name__)

# FLORES-200 language codes used by M2M100 tokenizer
_FLORES: dict[str, str] = {
    "EN": "eng_Latn",
    "FR": "fra_Latn",
    "ES": "spa_Latn",
    "DE": "deu_Latn",
    "IT": "ita_Latn",
    "PT": "por_Latn",
    "NL": "nld_Latn",
    "PL": "pol_Latn",
    "RU": "rus_Cyrl",
    "ZH": "zho_Hans",
    "JA": "jpn_Jpan",
    "KO": "kor_Hang",
    "AR": "arb_Arab",
    "TR": "tur_Latn",
    "TH": "tha_Thai",
    "VI": "vie_Latn",
    "ID": "ind_Latn",
    "HI": "hin_Deva",
}

_NUM_BEAMS = 4
_MAX_NEW_TOKENS = 128


class TranslationModel:
    _instance: "TranslationModel | None" = None

    def __init__(self) -> None:
        self._loaded = False
        self._tok = None
        self._model = None
        self._device = "cpu"

    def _load(self, path: str) -> None:
        import json
        import torch
        from transformers import PreTrainedTokenizerFast, AutoModelForSeq2SeqLM, AutoConfig

        logger.info("Loading tokenizer from %s", path)
        # tokenizer_config.json was saved with transformers 5.0 (extra_special_tokens as list,
        # tokenizer_class "TokenizersBackend") — incompatible with 4.x. Load the fast tokenizer
        # directly from tokenizer.json and reconstruct lang_code_to_id from added tokens.
        self._tok = PreTrainedTokenizerFast(
            tokenizer_file=f"{path}/tokenizer.json",
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="</s>",
            pad_token="<pad>",
            cls_token="<s>",
            mask_token="<mask>",
        )
        # Build lang_code_to_id from added_tokens in tokenizer.json (FLORES-200 codes)
        with open(f"{path}/tokenizer.json") as f:
            tok_data = json.load(f)
        self._tok.lang_code_to_id = {
            t["content"]: t["id"]
            for t in tok_data.get("added_tokens", [])
            if "_" in t["content"] and len(t["content"]) > 4  # language code pattern like eng_Latn
        }
        self._tok.src_lang = "eng_Latn"

        # Weights are float32 (~4.8GB) — quantization_config in config.json is a training
        # artifact. Strip it so bitsandbytes isn't invoked (requires CUDA, not available here).
        config = AutoConfig.from_pretrained(path)
        if hasattr(config, "quantization_config"):
            config.quantization_config = None

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("Loading model weights from %s (bfloat16, device=%s)", path, device)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        self._device = device
        self._model.eval()

    @classmethod
    def get(cls) -> "TranslationModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def load_async(cls) -> None:
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
        except Exception as exc:
            logger.error("Model load failed: %s — Gemini fallback active", exc)
            metrics.model_loaded.set(0)

    def is_loaded(self) -> bool:
        return self._loaded

    def _flores(self, lang: str) -> Optional[str]:
        return _FLORES.get(lang.upper())

    def _tgt_token_id(self, flores_tgt: str) -> int:
        return self._tok.lang_code_to_id[flores_tgt]

    def _encode(self, texts: list[str], flores_src: str) -> dict:
        """Encode texts with the source language token prepended (M2M100/NLLB convention)."""
        import torch
        src_token_id = self._tok.lang_code_to_id[flores_src]
        enc = self._tok(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=_MAX_NEW_TOKENS,
            truncation=True,
        )
        # Prepend the source language token id at position 0
        src_col = torch.full((enc["input_ids"].shape[0], 1), src_token_id, dtype=torch.long)
        enc["input_ids"] = torch.cat([src_col, enc["input_ids"]], dim=1)
        if "attention_mask" in enc:
            ones = torch.ones((enc["attention_mask"].shape[0], 1), dtype=torch.long)
            enc["attention_mask"] = torch.cat([ones, enc["attention_mask"]], dim=1)
        # M2M100 doesn't use token_type_ids — remove it to avoid generate() warnings
        enc.pop("token_type_ids", None)
        return enc

    def _confidence(self, output, seq_idx: int, input_len: int) -> float:
        if not hasattr(output, "sequences_scores") or output.sequences_scores is None:
            return 0.8
        log_prob = output.sequences_scores[seq_idx].item()
        seq_len = max(output.sequences[seq_idx].shape[0] - input_len, 1)
        return min(max(math.exp(log_prob / seq_len), 0.0), 1.0)

    def translate(self, text: str, source_lang: str, target_lang: str) -> tuple[str, float]:
        import torch

        flores_src = self._flores(source_lang)
        flores_tgt = self._flores(target_lang)
        if not flores_src or not flores_tgt:
            raise ValueError(f"Unsupported language pair: {source_lang}→{target_lang}")

        tgt_token_id = self._tgt_token_id(flores_tgt)
        inputs = self._encode([text], flores_src)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                num_beams=_NUM_BEAMS,
                max_new_tokens=_MAX_NEW_TOKENS,
                return_dict_in_generate=True,
                output_scores=True,
            )

        translated = self._tok.decode(output.sequences[0], skip_special_tokens=True)
        confidence = self._confidence(output, 0, input_len)
        return translated, confidence

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[tuple[str, float]]:
        import torch

        flores_src = self._flores(source_lang)
        flores_tgt = self._flores(target_lang)
        if not flores_src or not flores_tgt:
            raise ValueError(f"Unsupported language pair: {source_lang}→{target_lang}")

        tgt_token_id = self._tgt_token_id(flores_tgt)
        inputs = self._encode(texts, flores_src)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                num_beams=_NUM_BEAMS,
                max_new_tokens=_MAX_NEW_TOKENS,
                return_dict_in_generate=True,
                output_scores=True,
            )

        translated_texts = self._tok.batch_decode(output.sequences, skip_special_tokens=True)
        return [
            (text, self._confidence(output, i, input_len))
            for i, text in enumerate(translated_texts)
        ]
