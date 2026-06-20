"""
Evaluation with Redis TM + opus-mt-en-fr model.
Tests three scenarios on real TMX data with realistic TM hit rate.

Methodology:
  - Build TM from ALL raw pairs (after quality filtering) — simulates production Redis TM
  - Test on a separate held-out set sampled from pairs that have a DUPLICATE source
    (i.e. sentences that appear 2+ times in the TMX — these represent repeated Headout content)
  - For non-duplicate test: sample randomly from cleaned unique pairs
  - Reports metrics for both populations + combined

Usage:
    python evaluate_with_tm.py ~/Desktop/Headout_Human-en_US-fr_FR-2026-06-18.tmx
"""

import sys, os, re, time, argparse, collections, random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

def ensure(pkg):
    try: __import__(pkg.replace("-","_"))
    except ImportError: os.system(f"{sys.executable} -m pip install -q {pkg}")

for p in ["transformers","sacrebleu","sentencepiece","torch","tqdm","redis"]:
    ensure(p)

from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from transformers import MarianMTModel, MarianTokenizer
import redis as redislib

parser = argparse.ArgumentParser()
parser.add_argument("tmx")
parser.add_argument("--model",      default="Helsinki-NLP/opus-mt-en-fr")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--test-size",  type=int, default=1000,
                    help="Test sentences per category (dup + non-dup)")
parser.add_argument("--redis-url",  default="redis://localhost:6379")
args = parser.parse_args()

TMX = Path(args.tmx).expanduser()
SEP = "─" * 65

# ── 1. Parse raw TMX ──────────────────────────────────────────────────────────
INLINE_TAG_RE = re.compile(
    r'<[^>]+?>|<!--[\s\S]*?-->|\{\{[^}]+\}\}|\{[a-zA-Z_][^}]*\}|%[sdife]'
)

def norm(t): return re.sub(r"\s+", " ", t).strip()
def wc(t): return len(t.split())
def to_tagged(text):
    counter = 0
    def rep(m):
        nonlocal counter
        counter += 1
        return f"{{{counter}}}"
    return INLINE_TAG_RE.sub(rep, norm(text))

print(f"Parsing {Path(args.tmx).name} ...")
t0 = time.time()
raw: list[tuple[str,str]] = []
ctx = ET.iterparse(str(TMX), events=("end",))
src = tgt = None
for event, elem in ctx:
    tag = elem.tag.split("}")[-1].lower()
    if tag == "tuv":
        lang = (elem.get("{http://www.w3.org/XML/1998/namespace}lang") or
                elem.get("lang") or "").lower().split("-")[0]
        seg = elem.find("seg") or elem.find("{*}seg")
        text = (seg.text or "").strip() if seg is not None else ""
        if lang == "en": src = norm(text)
        elif lang == "fr": tgt = norm(text)
    elif tag == "tu":
        if src and tgt: raw.append((src, tgt))
        src = tgt = None
        elem.clear()
print(f"  {len(raw):,} raw pairs  ({time.time()-t0:.1f}s)")

# ── 2. Find duplicate sources (the TM's sweet spot) ───────────────────────────
src_counts = collections.Counter(en.lower() for en,_ in raw)

def is_clean(en, fr):
    if not en or not fr: return False
    if en.lower() == fr.lower(): return False
    if en.strip() == "-": return False
    if wc(en) < 2 or wc(fr) < 2: return False
    if wc(en) > 20: return False      # above TM_WORD_LIMIT anyway
    r = wc(fr)/wc(en)
    return 0.4 <= r <= 3.5

# Pairs where the EN source appears 2+ times → duplicates
dup_pairs   = [(en,fr) for en,fr in raw if src_counts[en.lower()] >= 2 and is_clean(en,fr)]
# Pairs where the EN source appears exactly once → unique
uniq_pairs  = [(en,fr) for en,fr in raw if src_counts[en.lower()] == 1 and is_clean(en,fr)]

print(f"  Duplicate-source pairs: {len(dup_pairs):,}  (TM will hit these)")
print(f"  Unique-source pairs:    {len(uniq_pairs):,}  (TM will miss these → model)")

# ── 3. Sample test sets ───────────────────────────────────────────────────────
random.seed(42)
# For dup test: pick one pair per unique source (the second occurrence)
seen_dup = set()
dup_test = []
for en, fr in dup_pairs:
    k = en.lower()
    if k not in seen_dup:
        seen_dup.add(k)
        dup_test.append((en, fr))
random.shuffle(dup_test)
dup_test = dup_test[:args.test_size]

random.shuffle(uniq_pairs)
uniq_test = uniq_pairs[:args.test_size]

# Combined test (realistic mix)
combined_test = random.sample(dup_test + uniq_test, min(args.test_size*2, len(dup_test+uniq_test)))

print(f"\nTest sets → dup: {len(dup_test):,}  unique: {len(uniq_test):,}  combined: {len(combined_test):,}")

# ── 4. Connect Redis TM ───────────────────────────────────────────────────────
import hashlib
def redis_key(source):
    tagged = to_tagged(source)
    h = hashlib.sha256(tagged.lower().encode()).hexdigest()
    return f"tm:en:fr:{h}"

print(f"\nConnecting to Redis: {args.redis_url}")
r = redislib.from_url(args.redis_url, decode_responses=True)
try:
    r.ping()
    print("  Connected ✓")
except Exception as e:
    sys.exit(f"  Redis connection failed: {e}")

def tm_lookup(src: str) -> Optional[str]:
    return r.get(redis_key(src))

# ── 5. Load model ─────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model} ...")
tokenizer = MarianTokenizer.from_pretrained(args.model)
model     = MarianMTModel.from_pretrained(args.model)
model.eval()

def translate_batch(texts):
    if not texts: return []
    toks = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    out  = model.generate(**toks, num_beams=4, max_length=160)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def run_scenario(pairs, label):
    srcs  = [en for en,_ in pairs]
    refs  = [fr for _,fr  in pairs]

    # Model-only predictions
    preds_model = []
    for i in tqdm(range(0, len(srcs), args.batch_size), desc=f"  {label} model", leave=False):
        preds_model.extend(translate_batch(srcs[i:i+args.batch_size]))

    # TM + model predictions
    preds_tm_model = [None] * len(srcs)
    miss_idx, miss_src = [], []
    hits = 0
    for i, src in enumerate(srcs):
        hit = tm_lookup(src)
        if hit:
            preds_tm_model[i] = hit
            hits += 1
        else:
            miss_idx.append(i)
            miss_src.append(src)

    miss_preds = []
    for i in tqdm(range(0, len(miss_src), args.batch_size), desc=f"  {label} misses", leave=False):
        miss_preds.extend(translate_batch(miss_src[i:i+args.batch_size]))
    for i, pred in zip(miss_idx, miss_preds):
        preds_tm_model[i] = pred

    hit_rate = hits / len(srcs) * 100

    bleu  = BLEU(effective_order=True)
    chrf  = CHRF()
    chrfpp = CHRF(word_order=2)

    def sc(preds):
        return {
            "BLEU":   round(bleu.corpus_score(preds, [refs]).score, 2),
            "ChrF":   round(chrf.corpus_score(preds, [refs]).score, 2),
            "ChrF++": round(chrfpp.corpus_score(preds, [refs]).score, 2),
        }

    s_model    = sc(preds_model)
    s_combined = sc(preds_tm_model)

    return hit_rate, s_model, s_combined

# ── 6. Run all three populations ──────────────────────────────────────────────
print(f"\n{SEP}")
print("Running evaluation...")
print(SEP)

hr_dup,  sm_dup,  sc_dup  = run_scenario(dup_test,      "dup    ")
hr_uniq, sm_uniq, sc_uniq = run_scenario(uniq_test,     "unique ")
hr_comb, sm_comb, sc_comb = run_scenario(combined_test, "combined")

# ── 7. Results ────────────────────────────────────────────────────────────────
def row(label, hit_rate, s_model, s_comb):
    delta_bleu  = round(s_comb["BLEU"]   - s_model["BLEU"],  2)
    delta_chrfpp = round(s_comb["ChrF++"] - s_model["ChrF++"], 2)
    return (label, hit_rate,
            s_model["BLEU"], s_model["ChrF"], s_model["ChrF++"],
            s_comb["BLEU"],  s_comb["ChrF"],  s_comb["ChrF++"],
            delta_bleu, delta_chrfpp)

rows = [
    row("Duplicate content (repeats)", hr_dup,  sm_dup,  sc_dup),
    row("Unique content (novel)",      hr_uniq, sm_uniq, sc_uniq),
    row("Combined (realistic mix)",    hr_comb, sm_comb, sc_comb),
]

print(f"\n{SEP}")
print("RESULTS — EN→FR  (opus-mt-en-fr base + Redis TM, 330k pairs)")
print(SEP)
print(f"{'Population':<28} {'TM%':>5}  {'Model only':^20}  {'TM+Model':^20}  {'Δ ChrF++'}")
print(f"{'':28} {'':5}  {'BLEU':>6} {'ChrF':>6} {'ChrF++':>6}  {'BLEU':>6} {'ChrF':>6} {'ChrF++':>6}")
print(SEP)
for r_ in rows:
    label, hr, mb, mc, mcp, cb, cc, ccp, db, dcp = r_
    print(f"{label:<28} {hr:>4.0f}%  {mb:>6} {mc:>6} {mcp:>6}  {cb:>6} {cc:>6} {ccp:>6}  {dcp:>+6.2f}")
print(SEP)

print(f"""
Interpretation:
  Duplicate content ({hr_dup:.0f}% TM hit rate) → TM provides exact human translations
  Unique content    ({hr_uniq:.0f}% TM hit rate) → Model carries the load
  Combined          ({hr_comb:.0f}% TM hit rate) → realistic production mix

Score guide (ChrF++):
  < 55 = needs improvement  |  55-65 = usable  |  65-75 = good  |  > 75 = excellent
""")

# Sample comparisons
print(SEP)
print("Samples — where TM beats model (duplicate content):")
print(SEP)
for en, ref_fr in dup_test[:5]:
    hit = tm_lookup(en)
    print(f"  EN:    {en[:60]}")
    print(f"  REF:   {ref_fr[:60]}")
    print(f"  TM:    {(hit or 'miss')[:60]}")
    print()
