"""
Convert the fine-tuned Headout M2M100 model to CTranslate2 INT8.

Why CTranslate2:
  - 4.4 GB bitsandbytes model → ~1.1 GB CTranslate2 INT8
  - No bitsandbytes / accelerate dependency at inference
  - 3-5× faster CPU inference (GEMM optimised for AVX/NEON)
  - Designed for seq2seq NMT models (M2M100, NLLB, MarianMT)

Usage:
    python convert_to_ct2.py
    # or with explicit paths:
    python convert_to_ct2.py --model ./headout-translator --output ./headout-ct2

After conversion update MODEL_PATH in .env to point at ./headout-ct2
and update headout-translator/translate.py (see bottom of this script).
"""

import argparse, os, shutil, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="./headout-translator",   help="Source model directory")
parser.add_argument("--output", default="./headout-ct2",          help="CTranslate2 output directory")
parser.add_argument("--quant",  default="int8",                   choices=["int8", "int8_float16", "float16", "float32"])
args = parser.parse_args()

MODEL_DIR  = Path(args.model).resolve()
OUTPUT_DIR = Path(args.output).resolve()

print(f"\nConverting {MODEL_DIR.name} → CTranslate2 ({args.quant})")
print(f"Source: {MODEL_DIR}")
print(f"Output: {OUTPUT_DIR}\n")

# ── Step 1: install ctranslate2 if needed ─────────────────────────────────────
try:
    import ctranslate2
    print(f"ctranslate2 {ctranslate2.__version__} already installed")
except ImportError:
    print("Installing ctranslate2...")
    os.system(f"{sys.executable} -m pip install -q ctranslate2")
    import ctranslate2

# ── Step 2: load + dequantize to float32, save clean copy ─────────────────────
FP32_DIR = MODEL_DIR.parent / (MODEL_DIR.name + "-fp32")

if not FP32_DIR.exists():
    print("Loading bitsandbytes model and dequantizing to float32...")
    print("(This requires bitsandbytes — takes ~2 min on CPU)\n")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float32,   # dequantize to float32
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    # Remove the bitsandbytes training config so a clean save works
    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None

    print(f"Saving float32 model to {FP32_DIR} ...")
    model.save_pretrained(str(FP32_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(FP32_DIR))

    # Copy inference_config.json so translate.py can find it
    ic = MODEL_DIR / "inference_config.json"
    if ic.exists():
        shutil.copy(ic, FP32_DIR / "inference_config.json")

    print(f"Float32 model saved: {sum(f.stat().st_size for f in FP32_DIR.rglob('*') if f.is_file())/1e9:.1f} GB\n")
    del model  # free RAM before conversion
else:
    print(f"Float32 model already exists at {FP32_DIR} — skipping dequantization\n")

# ── Step 3: convert to CTranslate2 ────────────────────────────────────────────
if OUTPUT_DIR.exists():
    print(f"Removing existing {OUTPUT_DIR} ...")
    shutil.rmtree(OUTPUT_DIR)

print(f"Converting to CTranslate2 ({args.quant}) ...")
converter = ctranslate2.converters.OpusMTConverter(str(FP32_DIR))
# M2M100 uses the M2M100Converter
try:
    converter = ctranslate2.converters.M2M100Converter(str(FP32_DIR))
    print("  Using M2M100Converter")
except AttributeError:
    # Fall back to generic OPUS-MT converter for older ctranslate2 versions
    print("  Falling back to OpusMTConverter")

converter.convert(str(OUTPUT_DIR), quantization=args.quant, force=True)

# Copy tokenizer files and inference config
for fname in ["sentencepiece.bpe.model", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "special_tokens_map.json", "inference_config.json"]:
    src = FP32_DIR / fname
    if src.exists():
        shutil.copy(src, OUTPUT_DIR / fname)

ct2_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nCTranslate2 model saved: {ct2_size:.2f} GB  ({OUTPUT_DIR})")
print(f"Original size:           {sum(f.stat().st_size for f in MODEL_DIR.rglob('*') if f.is_file())/1e9:.1f} GB")
print(f"Reduction:               {100*(1-ct2_size/4.4):.0f}%")

# ── Step 4: update translate.py for CTranslate2 inference ─────────────────────
CT2_TRANSLATE = OUTPUT_DIR / "translate.py"
CT2_TRANSLATE.write_text('''"""
CTranslate2 inference for the fine-tuned Headout M2M100 model.
Replaces the original bitsandbytes-based translate.py.
~3-5× faster CPU inference, ~1GB RAM instead of 4.4GB.
"""
import json, ctranslate2, sentencepiece
from pathlib import Path

class HeadoutTranslator:
    def __init__(self, model_dir: str, device: str = "cpu"):
        cfg = json.load(open(Path(model_dir) / "inference_config.json"))
        self.device = device
        self.cfg    = cfg

        self.translator = ctranslate2.Translator(
            model_dir,
            device=device,
            inter_threads=4,
            intra_threads=4,
        )
        sp_path = Path(model_dir) / "sentencepiece.bpe.model"
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(str(sp_path))

        self.src_lang = cfg["src_lang"]   # e.g. "eng_Latn"
        self.tgt_token = f"__{cfg['tgt_lang'].split('_')[0]}__"  # e.g. "__fra__"

    def _encode(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    def _decode(self, tokens: list[str]) -> str:
        return self.sp.decode(tokens)

    def translate(self, text: str) -> str:
        tokens = [self.src_lang] + self._encode(text)
        result = self.translator.translate_batch(
            [tokens],
            target_prefix=[[self.tgt_token]],
            beam_size=self.cfg.get("num_beams", 4),
            max_decoding_length=self.cfg.get("max_new_tokens", 64),
        )
        return self._decode(result[0].hypotheses[0][1:])  # skip tgt_token

    def translate_batch(self, texts: list[str]) -> list[str]:
        batch = [[self.src_lang] + self._encode(t) for t in texts]
        results = self.translator.translate_batch(
            batch,
            target_prefix=[[self.tgt_token]] * len(batch),
            beam_size=self.cfg.get("num_beams", 4),
            max_decoding_length=self.cfg.get("max_new_tokens", 64),
        )
        return [self._decode(r.hypotheses[0][1:]) for r in results]

if __name__ == "__main__":
    import sys
    t = HeadoutTranslator(sys.argv[1] if len(sys.argv) > 1 else ".")
    sentences = [
        "Skip the Line: Eiffel Tower Summit Access",
        "Free cancellation available up to 24 hours before the experience.",
        "Children must be accompanied by an adult at all times.",
    ]
    for s in sentences:
        print(f"EN: {s}")
        print(f"FR: {t.translate(s)}\\n")
''')

print(f"\nNew translate.py written to {CT2_TRANSLATE}")
print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print(f"1. Set MODEL_PATH={OUTPUT_DIR} in mercury-ml/.env")
print(f"2. pip install ctranslate2 sentencepiece")
print(f"3. Test: python {CT2_TRANSLATE} {OUTPUT_DIR}")
print(f"4. Restart mercury-ml service")
print("="*60)