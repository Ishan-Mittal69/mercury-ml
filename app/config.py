import os

# ── Model ─────────────────────────────────────────────────────────────────────
# Path to CTranslate2 / OPUS-MT / NLLB model directory.
# Leave empty until weights are ready — Gemini handles everything until then.
MODEL_PATH: str = os.getenv("MODEL_PATH", "")

# Confidence below this → fall back to Gemini for that segment.
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash")

# ── Server ────────────────────────────────────────────────────────────────────
PORT: int = int(os.getenv("PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
