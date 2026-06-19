"""
Fine-tune NLLB-200-distilled-1.3B on Headout TMX data using QLoRA.

Why NLLB + QLoRA over opus-mt:
  - NLLB-1.3B has 4× more parameters → better translation quality ceiling
  - QLoRA: train in 4-bit quantization → fits on Kaggle T4 16GB (free!)
  - LoRA adapters: ~100MB saved vs ~5GB full model
  - Multilingual: same model handles all 14 language pairs

GPU memory breakdown on T4 16GB:
  NLLB-1.3B (4-bit weights):  ~1.3 GB
  LoRA adapters (rank=16):    ~50  MB
  Activations + optimizer:    ~6   GB
  Total:                      ~8   GB  (8 GB headroom on T4)

Usage (Kaggle / Colab):
    1. Upload TMX as Kaggle dataset
    2. Set TMX_PATH below
    3. !pip install -q bitsandbytes peft transformers datasets sacrebleu sentencepiece accelerate
    4. Run all cells

Output:
    ./nllb-lora-adapter/     — LoRA adapter weights (~100MB, push to HuggingFace Hub)
    ./nllb-merged/           — Merged model (HuggingFace format)
    ./ct2_model/             — CTranslate2 INT8  →  drop into mercury-ml MODEL_PATH
"""

# ── 0. Install (run once) ──────────────────────────────────────────────────────
# !pip install -q bitsandbytes peft transformers datasets sacrebleu sentencepiece accelerate

import os, re, time, random, argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig, TaskType,
    get_peft_model, prepare_model_for_kbit_training,
)
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from sacrebleu.metrics import BLEU

# ── Config ─────────────────────────────────────────────────────────────────────

TMX_PATH   = "/kaggle/input/headout-tmx/Headout_Human-en_US-fr_FR.tmx"
BASE_MODEL = "facebook/nllb-200-distilled-1.3B"

SRC_LANG   = "eng_Latn"   # NLLB language codes (not ISO 639-1)
TGT_LANG   = "fra_Latn"

OUTPUT_ADAPTER = "./nllb-lora-adapter"
OUTPUT_MERGED  = "./nllb-merged"
CT2_OUTPUT     = "./ct2_model"           # → mercury-ml MODEL_PATH

# QLoRA
LORA_RANK      = 16      # higher = better quality, more params (try 32 for best)
LORA_ALPHA     = 32      # typically 2× rank
LORA_DROPOUT   = 0.05

# Training
BATCH_SIZE     = 8       # T4 16GB handles 8 comfortably with 4-bit
GRAD_ACCUM     = 4       # effective batch = 32
LEARNING_RATE  = 5e-4    # higher LR works well with LoRA
NUM_EPOCHS     = 3       # QLoRA converges faster than full fine-tuning
WARMUP_STEPS   = 200
EVAL_STEPS     = 500
MAX_SRC_LEN    = 128
MAX_TGT_LEN    = 160
TRAIN_SPLIT    = 0.95

# Data
MIN_WORDS      = 3
MAX_WORDS      = 150

# ── 1. Parse TMX ───────────────────────────────────────────────────────────────

def parse_tmx(path: str) -> list[dict]:
    print(f"Parsing {path} ...")
    pairs = []
    tree = ET.parse(path)
    root = tree.getroot()
    for tu in root.iter("tu"):
        srcs, tgts = {}, {}
        for tuv in tu.iter("tuv"):
            lang = (
                tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
                or tuv.get("lang") or ""
            ).lower().split("-")[0]
            seg = tuv.find("seg")
            if seg is not None and seg.text:
                text = seg.text.strip()
                if lang == "en":  srcs["en"] = text
                elif lang == "fr": tgts["fr"] = text
        if "en" in srcs and "fr" in tgts:
            pairs.append({"en": srcs["en"], "fr": tgts["fr"]})
    print(f"  {len(pairs):,} raw pairs")
    return pairs


def filter_pairs(pairs: list[dict]) -> list[dict]:
    seen, clean = set(), []
    for p in pairs:
        en = re.sub(r"\s+", " ", p["en"]).strip()
        fr = re.sub(r"\s+", " ", p["fr"]).strip()
        if not en or not fr: continue
        if en.lower() == fr.lower(): continue
        ew, fw = len(en.split()), len(fr.split())
        if ew < MIN_WORDS or fw < MIN_WORDS: continue
        if ew > MAX_WORDS or fw > MAX_WORDS: continue
        if not (0.4 <= fw/ew <= 3.5): continue
        key = en.lower()
        if key in seen: continue
        seen.add(key)
        clean.append({"en": en, "fr": fr})
    print(f"  {len(clean):,} pairs after filtering")
    return clean


# ── 2. Tokenise for NLLB ──────────────────────────────────────────────────────

def tokenise(batch, tokenizer: NllbTokenizer, tgt_lang: str):
    # Source: set tokenizer to source language
    tokenizer.src_lang = SRC_LANG
    model_inputs = tokenizer(
        batch["en"],
        max_length=MAX_SRC_LEN,
        truncation=True,
        padding=False,
    )
    # Target: encode with target language
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["fr"],
            max_length=MAX_TGT_LEN,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ── 3. BLEU metric ─────────────────────────────────────────────────────────────

def compute_metrics(eval_pred, tokenizer: NllbTokenizer):
    preds, labels = eval_pred
    if isinstance(preds, tuple): preds = preds[0]
    labels = [[(t if t != -100 else tokenizer.pad_token_id) for t in l] for l in labels]
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = BLEU()
    score = bleu.corpus_score(
        [p.strip() for p in decoded_preds],
        [[l.strip()] for l in decoded_labels],
    )
    return {"bleu": score.score}


# ── 4. Main ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  CUDA: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")
    print(f"Base model: {BASE_MODEL}\n")

    # Data
    raw   = parse_tmx(TMX_PATH)
    pairs = filter_pairs(raw)
    random.seed(42); random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_SPLIT)
    train_ds = Dataset.from_list(pairs[:n_train])
    eval_ds  = Dataset.from_list(pairs[n_train:])
    print(f"Train: {len(train_ds):,}  Eval: {len(eval_ds):,}\n")

    # ── QLoRA: load model in 4-bit ─────────────────────────────────────────────
    print("Loading tokenizer + model in 4-bit (QLoRA)...")
    tokenizer = NllbTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 = best quality for fine-tuning
        bnb_4bit_use_double_quant=True,      # saves ~0.4 bits per param
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"  Parameters: {model.num_parameters()/1e6:.0f}M  |  Trainable before LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    # ── LoRA adapters ──────────────────────────────────────────────────────────
    model = prepare_model_for_kbit_training(model)

    # Target the attention projection layers in both encoder and decoder
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "v_proj",          # attention
            "k_proj", "out_proj",        # attention
            "fc1",    "fc2",             # feed-forward
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({trainable/total*100:.1f}%)\n")

    # ── Tokenise ───────────────────────────────────────────────────────────────
    _tok = lambda batch: tokenise(batch, tokenizer, TGT_LANG)
    train_ds = train_ds.map(_tok, batched=True, remove_columns=["en","fr"])
    eval_ds  = eval_ds.map(_tok,  batched=True, remove_columns=["en","fr"])

    # ── Training args ──────────────────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_ADAPTER,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=False,               # 4-bit + bfloat16 compute — don't mix with fp16 trainer
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit", # 8-bit AdamW — saves optimizer memory
        logging_steps=100,
        report_to="none",
        generation_max_length=MAX_TGT_LEN,
        # NLLB needs forced_bos_token_id for target language
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
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

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Starting QLoRA fine-tuning...")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(OUTPUT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_ADAPTER)
    print(f"\nLoRA adapter saved → {OUTPUT_ADAPTER}/  (push to HuggingFace Hub for versioning)")

    # Final BLEU
    results = trainer.evaluate()
    print(f"Final eval BLEU: {results.get('eval_bleu','?'):.2f}")

    # ── Sample translations ────────────────────────────────────────────────────
    print("\n=== Sample translations ===")
    samples = [
        "Skip the line: Eiffel Tower Summit with Expert Guide",
        "Free cancellation available up to 24 hours before the experience.",
        "Children must be accompanied by an adult at all times.",
    ]
    tokenizer.src_lang = SRC_LANG
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)
    inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SRC_LEN)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k,v in inputs.items()}
    translated = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=MAX_TGT_LEN)
    for src, tr in zip(samples, tokenizer.batch_decode(translated, skip_special_tokens=True)):
        print(f"  EN: {src}")
        print(f"  FR: {tr}\n")

    # ── Merge + convert to CTranslate2 ────────────────────────────────────────
    print("Merging LoRA adapters into base model...")
    from peft import PeftModel
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, OUTPUT_ADAPTER)
    merged = merged.merge_and_unload()
    merged.save_pretrained(OUTPUT_MERGED)
    tokenizer.save_pretrained(OUTPUT_MERGED)
    print(f"Merged model saved → {OUTPUT_MERGED}/")

    print(f"\nConverting to CTranslate2 INT8 for mercury-ml...")
    os.system(
        f"ct2-nllb-converter "
        f"--model {OUTPUT_MERGED} "
        f"--output_dir {CT2_OUTPUT} "
        f"--quantization int8 "
        f"--force"
    )
    print(f"CTranslate2 model → {CT2_OUTPUT}/")
    print(f"\n→ Set MODEL_PATH={CT2_OUTPUT} in mercury-ml .env")
    print(f"→ Uncomment ctranslate2 + sentencepiece in requirements.txt")
    print(f"→ Update model.py: set src_lang='eng_Latn', tgt_lang='fra_Latn' in translate()")


if __name__ == "__main__":
    main()
