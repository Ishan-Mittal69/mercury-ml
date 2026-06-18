"""
Mercury ML Service
FastAPI inference server for the Mercury translation pipeline.

Endpoints:
  POST /translate   — translate a batch of segments (own model → Gemini fallback)
  GET  /health      — liveness probe
  GET  /ready       — readiness probe (checks Gemini key is configured)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config
from app.model import TranslationModel
from app import translate as translator

logging.basicConfig(
    level=config.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load model on startup (no-op if weights not ready)
    TranslationModel.get()
    model_status = "loaded" if TranslationModel.get().is_loaded() else "not loaded (Gemini active)"
    logger.info("Model status: %s", model_status)
    yield


app = FastAPI(title="Mercury ML Service", version="1.0.0", lifespan=lifespan)


# ── Schema ─────────────────────────────────────────────────────────────────────

class SegmentIn(BaseModel):
    id: str
    text: str


class TranslateRequest(BaseModel):
    segments: list[SegmentIn]
    sourceLang: str
    targetLang: str


class SegmentOut(BaseModel):
    id: str
    text: str
    confidence: float


class TranslateResponse(BaseModel):
    results: list[SegmentOut]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    model = TranslationModel.get()
    return {
        "status": "ok",
        "model": "loaded" if model.is_loaded() else "stub",
        "gemini": "configured" if config.GEMINI_API_KEY else "not configured",
    }


@app.get("/ready")
def ready():
    model = TranslationModel.get()
    if not model.is_loaded() and not config.GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Neither model weights nor GEMINI_API_KEY are configured",
        )
    return {"ready": True}


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest) -> TranslateResponse:
    if not request.segments:
        return TranslateResponse(results=[])

    logger.info(
        "translate %d segments %s→%s",
        len(request.segments),
        request.sourceLang,
        request.targetLang,
    )

    results = await translator.translate(
        segments=[{"id": s.id, "text": s.text} for s in request.segments],
        source_lang=request.sourceLang,
        target_lang=request.targetLang,
    )

    return TranslateResponse(
        results=[SegmentOut(id=r["id"], text=r["text"], confidence=r["confidence"]) for r in results]
    )
