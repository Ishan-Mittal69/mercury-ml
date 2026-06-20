"""
Evaluate Gemini 3 Flash on the same EN→FR test set used for opus-mt.
Reports BLEU, ChrF, ChrF++ so you can compare directly.

Usage:
    python evaluate_gemini.py ~/Desktop/Headout_Human-en_US-fr_FR-2026-06-18.tmx
"""

import sys, os, re, time, json, argparse, random
import xml.etree.ElementTree as ET
from pathlib import Path

def ensure(pkg):
    try: __import__(pkg.replace("-","_"))
    except ImportError: os.system(f"{sys.executable} -m pip install -q {pkg}")

for p in ["sacrebleu", "httpx", "tqdm"]:
    ensure(p)

from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
import httpx

parser = argparse.ArgumentParser()
parser.add_argument("tmx")
parser.add_argument("--test-size",   type=int, default=500,
                    help="Sentences to evaluate (default 500 — costs ~$0.30)")
parser.add_argument("--batch-size",  type=int, default=20,
                    help="Segments per Gemini API call")
parser.add_argument("--model",       default="gemini-2.5-flash")
parser.add_argument("--api-key",     default=os.getenv("GEMINI_API_KEY", ""))
args = parser.parse_args()

if not args.api_key:
    sys.exit("GEMINI_API_KEY not set. Export it or pass --api-key")

SEP = "─" * 60

# ── 1. Parse + filter (same logic as evaluate_baseline.py) ───────────────────
def norm(t): return re.sub(r"\s+", " ", t).strip()
def wc(t):   return len(t.split())

print(f"Parsing {Path(args.tmx).name} ...")
t0 = time.time()
raw = []
ctx = ET.iterparse(str(Path(args.tmx).expanduser()), events=("end",))
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

clean, seen = [], set()
for en, fr in raw:
    if not en or not fr: continue
    if en.lower() == fr.lower(): continue
    if wc(en) < 3 or wc(fr) < 3: continue
    if wc(en) > 150 or wc(fr) > 150: continue
    ratio = wc(fr) / wc(en)
    if ratio < 0.4 or ratio > 3.5: continue
    key = en.lower()
    if key in seen: continue
    seen.add(key)
    clean.append((en, fr))

print(f"  {len(clean):,} clean pairs")

random.seed(42)
random.shuffle(clean)
# Use the same 90/10 split as evaluate_baseline.py
n_train = int(len(clean) * 0.9)
test_pairs = clean[n_train:][:args.test_size]
print(f"  Test set: {len(test_pairs):,} sentences\n")

# ── 2. Gemini translation ─────────────────────────────────────────────────────
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models"
    f"/{args.model}:generateContent?key={args.api_key}"
)

def build_prompt(segments: list[dict]) -> str:
    return f"""Translate the following segments from English to French.

Rules:
- Return a JSON array with exactly the same number of objects: [{{"id": "...", "text": "..."}}]
- Preserve all {{N}} placeholders exactly as-is
- Translate only the human-readable text
- Do not add explanations

Segments:
{json.dumps(segments, ensure_ascii=False)}"""

def gemini_translate(texts: list[str]) -> list[str]:
    segments = [{"id": str(i), "text": t} for i, t in enumerate(texts)]
    payload = {
        "contents": [{"parts": [{"text": build_prompt(segments)}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    resp = httpx.post(GEMINI_URL, json=payload, timeout=60.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:200]}")
    raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    results = json.loads(raw_text)
    by_id = {r["id"]: r["text"] for r in results}
    return [by_id.get(str(i), texts[i]) for i in range(len(texts))]

srcs = [en for en, _ in test_pairs]
refs = [fr for _, fr in test_pairs]

print(f"Translating {len(srcs)} sentences with {args.model} ...")
print(f"(batch size {args.batch_size} → {len(srcs)//args.batch_size + 1} API calls)\n")

preds = []
total_tokens_est = 0
errors = 0

for i in tqdm(range(0, len(srcs), args.batch_size), unit="batch"):
    batch = srcs[i:i + args.batch_size]
    try:
        translated = gemini_translate(batch)
        preds.extend(translated)
        total_tokens_est += sum(len(t.split()) * 1.3 for t in batch) + sum(len(t.split()) * 1.3 for t in translated)
        time.sleep(0.3)  # gentle rate limiting
    except Exception as e:
        print(f"\n  Error on batch {i//args.batch_size}: {e}")
        preds.extend(batch)  # fallback to source
        errors += 1

# ── 3. Score ──────────────────────────────────────────────────────────────────
bleu   = BLEU(effective_order=True)
chrf   = CHRF()
chrfpp = CHRF(word_order=2)

b  = round(bleu.corpus_score(preds, [refs]).score, 2)
c  = round(chrf.corpus_score(preds, [refs]).score, 2)
cp = round(chrfpp.corpus_score(preds, [refs]).score, 2)

# Estimate cost (approx)
input_tokens  = sum(len(s.split()) * 1.3 for s in srcs) + len(srcs) * 50  # prompt overhead
output_tokens = sum(len(p.split()) * 1.3 for p in preds)
cost_usd = (input_tokens / 1e6 * 0.50) + (output_tokens / 1e6 * 3.00)

print(f"\n{SEP}")
print(f"RESULTS — {args.model}  |  EN→FR Headout content")
print(SEP)
print(f"  Test sentences:  {len(test_pairs):,}")
print(f"  API errors:      {errors}")
print(f"  Est. cost:       ${cost_usd:.3f}")
print(SEP)
print(f"  BLEU:    {b}")
print(f"  ChrF:    {c}")
print(f"  ChrF++:  {cp}   ← most reliable for French")
print(SEP)

print(f"\nFor comparison (from evaluate_baseline.py on same test split):")
print(f"  opus-mt-en-fr (no fine-tuning):  BLEU 45.78 | ChrF++ 66.12")
print(f"  {args.model}:  BLEU {b} | ChrF++ {cp}")
delta = round(cp - 66.12, 2)
winner = args.model if cp > 66.12 else "opus-mt-en-fr"
print(f"  Δ ChrF++: {delta:+.2f}  → {winner} wins on Headout content\n")

# Sample outputs
print(SEP)
print("Sample translations (first 5):")
print(SEP)
for i in range(min(5, len(srcs))):
    print(f"  EN:  {srcs[i][:70]}")
    print(f"  REF: {refs[i][:70]}")
    print(f"  OUT: {preds[i][:70]}")
    print()
