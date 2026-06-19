"""
Fine-tune Helsinki-NLP/opus-mt-tc-big-en-fr on Headout TMX data.

Usage (Kaggle / Colab):
    1. Upload your TMX file as a Kaggle dataset or Google Drive file
    2. Set TMX_PATH below
    3. Run all cells / python finetune_en_fr.py

Output:
    ./finetuned-model/    — HuggingFace model (checkpoint)
    ./ct2_model/          — CTranslate2 INT8 model → drop into mercury-ml MODEL_PATH
"""

# ── 0. Install deps (run once) ─────────────────────────────────────────────────
# !pip install -q transformers datasets sacrebleu sentencepiece ctranslate2 accelerate

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from sacrebleu.metrics import BLEU

# ── Config ─────────────────────────────────────────────────────────────────────

TMX_PATH = "/kaggle/input/headout-tmx/en_fr.tmx"   # ← set your TMX path
BASE_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-fr"    # ~300M params, best EN→FR base

OUTPUT_DIR = "./finetuned-model"
CT2_OUTPUT_DIR = "./ct2_model"                       # → this goes into mercury-ml MODEL_PATH

SRC_LANG = "en"
TGT_LANG = "fr"

MAX_SOURCE_LEN = 128   # tokens; tour content rarely exceeds this
MAX_TARGET_LEN = 160   # French is ~20% longer than English
BATCH_SIZE = 32        # T4/P100 16GB handles this comfortably for this model
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
WARMUP_STEPS = 500
EVAL_STEPS = 500       # evaluate every 500 steps
SAVE_STEPS = 500

TRAIN_SPLIT = 0.95     # 95% train, 5% eval
MIN_PAIR_LEN = 3       # words — filter very short pairs
MAX_PAIR_LEN = 200     # words — filter very long / corrupted pairs

# ── 1. Parse TMX ───────────────────────────────────────────────────────────────

def parse_tmx(path: str) -> list[dict]:
    """Extract (en, fr) sentence pairs from a TMX file."""
    print(f"Parsing {path} ...")
    pairs = []
    tree = ET.parse(path)
    root = tree.getroot()

    # TMX namespace varies; handle both namespaced and bare elements
    for tu in root.iter("tu"):
        srcs, tgts = {}, {}
        for tuv in tu.iter("tuv"):
            lang = (
                tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
                or tuv.get("lang")
                or tuv.get("xml:lang")
                or ""
            ).lower().split("-")[0]  # "fr-FR" → "fr"
            seg = tuv.find("seg")
            if seg is not None and seg.text:
                text = seg.text.strip()
                if lang == SRC_LANG:
                    srcs[lang] = text
                elif lang == TGT_LANG:
                    tgts[lang] = text

        if SRC_LANG in srcs and TGT_LANG in tgts:
            pairs.append({"en": srcs[SRC_LANG], "fr": tgts[TGT_LANG]})

    print(f"  Extracted {len(pairs):,} raw pairs")
    return pairs


# ── 2. Filter & clean ──────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def filter_pairs(pairs: list[dict]) -> list[dict]:
    seen = set()
    clean = []
    skipped = 0

    for p in pairs:
        src = normalize(p["en"])
        tgt = normalize(p["fr"])

        # Skip empty
        if not src or not tgt:
            skipped += 1; continue

        # Skip untranslated (source == target after normalization)
        if src.lower() == tgt.lower():
            skipped += 1; continue

        # Skip too short or too long
        sw, tw = word_count(src), word_count(tgt)
        if sw < MIN_PAIR_LEN or tw < MIN_PAIR_LEN:
            skipped += 1; continue
        if sw > MAX_PAIR_LEN or tw > MAX_PAIR_LEN:
            skipped += 1; continue

        # Length ratio sanity check (fr is ~20-40% longer than en)
        ratio = tw / sw
        if ratio < 0.5 or ratio > 3.0:
            skipped += 1; continue

        # Deduplicate on source
        key = src.lower()
        if key in seen:
            skipped += 1; continue
        seen.add(key)

        clean.append({"en": src, "fr": tgt})

    print(f"  After filtering: {len(clean):,} pairs  (skipped {skipped:,})")
    return clean


# ── 3. Tokenise ────────────────────────────────────────────────────────────────

def tokenise(batch, tokenizer: MarianTokenizer):
    model_inputs = tokenizer(
        batch["en"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["fr"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ── 4. BLEU evaluation ────────────────────────────────────────────────────────

def compute_metrics(eval_pred, tokenizer: MarianTokenizer):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 (padding) with pad token id
    labels = [
        [(t if t != -100 else tokenizer.pad_token_id) for t in label]
        for label in labels
    ]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]  # BLEU expects list of references

    bleu = BLEU()
    result = bleu.corpus_score(decoded_preds, decoded_labels)
    return {"bleu": result.score}


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Base model: {BASE_MODEL}")

    # Parse + filter
    raw_pairs = parse_tmx(TMX_PATH)
    pairs = filter_pairs(raw_pairs)

    # Train/eval split
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]
    print(f"Train: {len(train_pairs):,}  Eval: {len(eval_pairs):,}")

    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(eval_pairs)

    # Load tokenizer + model
    print(f"\nLoading {BASE_MODEL} ...")
    tokenizer = MarianTokenizer.from_pretrained(BASE_MODEL)
    model = MarianMTModel.from_pretrained(BASE_MODEL)
    print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")

    # Tokenise
    _tok = lambda batch: tokenise(batch, tokenizer)
    train_ds = train_ds.map(_tok, batched=True, remove_columns=["en", "fr"])
    eval_ds  = eval_ds.map(_tok,  batched=True, remove_columns=["en", "fr"])

    # Training args
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),   # FP16 on GPU, FP32 on CPU
        logging_steps=100,
        report_to="none",                 # disable wandb
        generation_max_length=MAX_TARGET_LEN,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    _metrics = lambda ep: compute_metrics(ep, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("\nStarting fine-tuning ...")
    trainer.train()

    # Save best checkpoint
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}/")

    # Final BLEU on eval set
    results = trainer.evaluate()
    print(f"\nFinal eval BLEU: {results.get('eval_bleu', '?'):.2f}")

    # Quick qualitative check
    print("\n=== Sample translations ===")
    samples = [
        "Skip the line: Eiffel Tower Summit Access with Expert Guide",
        "Explore the heart of Paris with a professional local guide.",
        "Free cancellation available up to 24 hours before the experience.",
    ]
    inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SOURCE_LEN)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()
    translated = model.generate(**inputs)
    for src, tgt in zip(samples, tokenizer.batch_decode(translated, skip_special_tokens=True)):
        print(f"  EN: {src}")
        print(f"  FR: {tgt}\n")

    # Convert to CTranslate2 for mercury-ml deployment
    print("Converting to CTranslate2 (INT8) for mercury-ml ...")
    os.system(
        f"ct2-opus-mt-converter "
        f"--model {OUTPUT_DIR} "
        f"--output_dir {CT2_OUTPUT_DIR} "
        f"--quantization int8 "
        f"--force"
    )
    print(f"\nCTranslate2 model saved to {CT2_OUTPUT_DIR}/")
    print(f"→ Set MODEL_PATH={CT2_OUTPUT_DIR} in mercury-ml .env")
    print("→ Uncomment ctranslate2 + sentencepiece in requirements.txt")
    print("→ Implement model.py _load() and translate() — see comments in that file")


if __name__ == "__main__":
    main()
