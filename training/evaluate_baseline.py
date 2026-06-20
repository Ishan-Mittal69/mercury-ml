#!/usr/bin/env python3
"""
Baseline evaluation: EN→FR using opus-mt-en-fr + TM exact match
Reports BLEU, ChrF and ChrF++ for three scenarios:
  1. Model only (no TM)
  2. TM only (exact match, model fills misses)
  3. TM + Model combined

Usage:
    python evaluate_baseline.py ~/Desktop/en_fr.tmx
    python evaluate_baseline.py ~/Desktop/en_fr.tmx --test-size 0.05 --sample 1000
"""

import sys, os, re, time, argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# ── Auto-install deps ─────────────────────────────────────────────────────────
def ensure(pkg: str, import_as: Optional[str] = None) -> None:
    mod = import_as or pkg.split("==")[0].replace("-", "_")
    try:
        __import__(mod)
    except ImportError:
        print(f"  installing {pkg} ...")
        os.system(f"{sys.executable} -m pip install -q {pkg}")

print("Checking dependencies...")
for p in ["transformers", "sacrebleu", "sentencepiece", "torch", "tqdm"]:
    ensure(p)
print("Dependencies OK.\n")

from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from transformers import MarianMTModel, MarianTokenizer

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("tmx", help="Path to EN→FR TMX file")
parser.add_argument("--test-size",  type=float, default=0.1,
                    help="Fraction of pairs to use as test set (default: 0.1)")
parser.add_argument("--sample",     type=int,   default=2000,
                    help="Max test sentences to evaluate (default: 2000, 0=all)")
parser.add_argument("--model",      default="Helsinki-NLP/opus-mt-en-fr",
                    help="HuggingFace model ID (default: opus-mt-en-fr, fast CPU)")
parser.add_argument("--batch-size", type=int,   default=16)
args = parser.parse_args()

TMX_PATH = Path(args.tmx).expanduser()
if not TMX_PATH.exists():
    sys.exit(f"TMX not found: {TMX_PATH}")

# ── 1. Parse TMX ──────────────────────────────────────────────────────────────
print(f"Parsing {TMX_PATH.name} ({TMX_PATH.stat().st_size / 1e6:.1f} MB) ...")
t0 = time.time()

pairs: list[tuple[str, str]] = []
ctx = ET.iterparse(str(TMX_PATH), events=("end",))

src_text: Optional[str] = None
tgt_text: Optional[str] = None
in_src = False
in_tgt = False

for event, elem in ctx:
    tag = elem.tag.split("}")[-1].lower()   # strip namespace
    if tag == "tuv":
        lang = (
            elem.get("{http://www.w3.org/XML/1998/namespace}lang")
            or elem.get("lang")
            or elem.get("xml:lang")
            or ""
        ).lower().split("-")[0]
        seg = elem.find("seg") or elem.find("{*}seg")
        text = (seg.text or "").strip() if seg is not None else ""
        if lang == "en":
            src_text = text
        elif lang == "fr":
            tgt_text = text
    elif tag == "tu":
        if src_text and tgt_text:
            pairs.append((src_text, tgt_text))
        src_text = tgt_text = None
        elem.clear()

print(f"  {len(pairs):,} raw pairs  ({time.time()-t0:.1f}s)")

# ── 2. Filter ─────────────────────────────────────────────────────────────────
def wc(t): return len(t.split())
def norm(t): return re.sub(r"\s+", " ", t).strip()

clean: list[tuple[str, str]] = []
seen: set[str] = set()
for en, fr in pairs:
    en, fr = norm(en), norm(fr)
    if not en or not fr: continue
    if en.lower() == fr.lower(): continue          # untranslated
    if wc(en) < 3 or wc(fr) < 3: continue         # too short
    if wc(en) > 150 or wc(fr) > 150: continue     # too long
    ratio = wc(fr) / wc(en)
    if ratio < 0.4 or ratio > 3.5: continue        # bad ratio
    key = en.lower()
    if key in seen: continue                        # deduplicate
    seen.add(key)
    clean.append((en, fr))

print(f"  {len(clean):,} clean pairs after filtering")

# ── 3. Split ──────────────────────────────────────────────────────────────────
import random
random.seed(42)
random.shuffle(clean)

n_test  = max(100, int(len(clean) * args.test_size))
n_train = len(clean) - n_test

train_pairs = clean[:n_train]
test_pairs  = clean[n_train:]

if args.sample and len(test_pairs) > args.sample:
    test_pairs = random.sample(test_pairs, args.sample)

print(f"\nSplit → TM (train): {n_train:,}  |  test: {len(test_pairs):,}\n")

# ── 4. Build TM (exact match dict) ───────────────────────────────────────────
tm: dict[str, str] = {norm(en).lower(): fr for en, fr in train_pairs}

def tm_lookup(src: str) -> Optional[str]:
    return tm.get(norm(src).lower())

# ── 5. Load model ─────────────────────────────────────────────────────────────
print(f"Loading model: {args.model}")
print("(downloading ~300MB on first run, cached afterwards)\n")
tokenizer = MarianTokenizer.from_pretrained(args.model)
model     = MarianMTModel.from_pretrained(args.model)
model.eval()

def translate_batch(texts: list[str]) -> list[str]:
    if not texts: return []
    toks = tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=128)
    out  = model.generate(**toks, num_beams=4, max_length=160)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

# ── 6. Evaluate all three scenarios ──────────────────────────────────────────
src_refs = [en for en, _ in test_pairs]
references = [fr for _, fr in test_pairs]

# ─ Scenario A: model only ────────────────────────────────────────────────────
print("A) Translating with model only ...")
model_preds: list[str] = []
for i in tqdm(range(0, len(src_refs), args.batch_size), unit="batch"):
    batch = src_refs[i:i+args.batch_size]
    model_preds.extend(translate_batch(batch))

# ─ Scenario B: TM + model (combined) ─────────────────────────────────────────
print("\nB) TM lookup + model for misses ...")
tm_hits    = 0
combined_preds: list[str] = []
miss_indices: list[int]   = []
miss_texts:  list[str]    = []

for idx, src in enumerate(src_refs):
    hit = tm_lookup(src)
    if hit:
        combined_preds.append(hit)
        tm_hits += 1
    else:
        combined_preds.append("")   # placeholder
        miss_indices.append(idx)
        miss_texts.append(src)

# translate misses in batches
for i in tqdm(range(0, len(miss_texts), args.batch_size), unit="batch",
              desc="  model for misses"):
    batch = miss_texts[i:i+args.batch_size]
    batch_preds = translate_batch(batch)
    for j, pred in enumerate(batch_preds):
        combined_preds[miss_indices[i + j]] = pred

tm_hit_rate = tm_hits / len(src_refs) * 100

# ─ Scenario C: TM only (source text for misses — lower bound) ────────────────
tm_only_preds: list[str] = []
for src in src_refs:
    hit = tm_lookup(src)
    tm_only_preds.append(hit if hit else src)   # copy-source for misses

# ── 7. Compute metrics ────────────────────────────────────────────────────────
bleu_metric  = BLEU(effective_order=True)
chrf_metric  = CHRF()
chrfpp_metric = CHRF(word_order=2)   # ChrF++

def score(preds: list[str]) -> dict:
    refs = [references]   # sacrebleu expects list of lists
    return {
        "BLEU":   round(bleu_metric.corpus_score(preds, refs).score, 2),
        "ChrF":   round(chrf_metric.corpus_score(preds, refs).score, 2),
        "ChrF++": round(chrfpp_metric.corpus_score(preds, refs).score, 2),
    }

scores_model    = score(model_preds)
scores_combined = score(combined_preds)
scores_tm_only  = score(tm_only_preds)

# ── 8. Print results ──────────────────────────────────────────────────────────
SEP = "─" * 60
print(f"\n{SEP}")
print("RESULTS — EN→FR baseline evaluation")
print(SEP)
print(f"  Test set:     {len(test_pairs):,} sentences")
print(f"  TM size:      {n_train:,} sentence pairs")
print(f"  TM hit rate:  {tm_hit_rate:.1f}%  ({tm_hits:,} exact matches)")
print(f"  Model:        {args.model}")
print(SEP)
print(f"{'Scenario':<28} {'BLEU':>8} {'ChrF':>8} {'ChrF++':>8}")
print(SEP)
print(f"{'A. Model only':<28} {scores_model['BLEU']:>8} {scores_model['ChrF']:>8} {scores_model['ChrF++']:>8}")
print(f"{'B. TM + Model (combined)':<28} {scores_combined['BLEU']:>8} {scores_combined['ChrF']:>8} {scores_combined['ChrF++']:>8}")
print(f"{'C. TM only (copy-src miss)':<28} {scores_tm_only['BLEU']:>8} {scores_tm_only['ChrF']:>8} {scores_tm_only['ChrF++']:>8}")
print(SEP)

# Score interpretation
print("\nScore guide (ChrF++ is most reliable for FR):")
print("  < 30  BLEU / < 55 ChrF++ = needs improvement")
print("  30-40 BLEU / 55-65 ChrF++ = usable")
print("  40-50 BLEU / 65-75 ChrF++ = good (near human parity for domain)")
print("  > 50  BLEU / > 75 ChrF++ = excellent")

# Sample predictions
print(f"\n{SEP}")
print("Sample predictions (first 5 test sentences):")
print(SEP)
for i in range(min(5, len(src_refs))):
    hit = "✓ TM" if tm_lookup(src_refs[i]) else "  MT"
    print(f"\n[{hit}] EN: {src_refs[i][:90]}")
    print(f"      REF: {references[i][:90]}")
    print(f"      OUT: {combined_preds[i][:90]}")
print(SEP)

print("\nDone. Scores saved above.")
print("Next step: fine-tune opus-mt-en-fr on the train split → compare scores.")
