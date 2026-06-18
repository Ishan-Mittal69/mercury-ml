import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.logging_config import setup_logging
from app.model import TranslationModel
from app import translate as translator
from app import metrics

setup_logging()
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Shared HTTP client for Gemini — connection pool lives for app lifetime
    app.state.http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        timeout=settings.GEMINI_TIMEOUT,
    )

    # Load model (non-blocking — runs in thread pool)
    await TranslationModel.load_async()

    logger.info(
        "Mercury ML service ready",
        extra={
            "model": "loaded" if TranslationModel.get().is_loaded() else "stub",
            "gemini": settings.gemini_ready,
        },
    )

    if not TranslationModel.get().is_loaded() and not settings.gemini_ready:
        logger.warning("Neither model nor Gemini is configured — all translations will return source text")

    yield

    await app.state.http.aclose()
    logger.info("HTTP client closed")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Mercury ML Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request ID middleware ──────────────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start = time.perf_counter()

    response: Response = await call_next(request)

    elapsed = time.perf_counter() - request.state.start
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed:.4f}s"
    return response


# ── Schema ─────────────────────────────────────────────────────────────────────

class SegmentIn(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=0, max_length=settings.MAX_TEXT_LENGTH)


class TranslateRequest(BaseModel):
    segments: Annotated[list[SegmentIn], Field(max_length=settings.MAX_SEGMENTS_PER_REQUEST)]
    sourceLang: str = Field(min_length=2, max_length=20)
    targetLang: str = Field(min_length=2, max_length=20)

    @field_validator("segments")
    @classmethod
    def no_duplicate_ids(cls, v: list[SegmentIn]) -> list[SegmentIn]:
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Segment IDs must be unique within a request")
        return v


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
        "gemini": "configured" if settings.gemini_ready else "not configured",
    }


@app.get("/ready")
def ready():
    model = TranslationModel.get()
    if not model.is_loaded() and not settings.gemini_ready:
        raise HTTPException(
            status_code=503,
            detail="Neither model weights nor GEMINI_API_KEY are configured",
        )
    return {"ready": True}


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: Request, body: TranslateRequest) -> TranslateResponse:
    if not body.segments:
        return TranslateResponse(results=[])

    metrics.translate_requests_total.labels(
        source_lang=body.sourceLang, target_lang=body.targetLang
    ).inc()

    with metrics.translate_latency_seconds.labels(
        source_lang=body.sourceLang, target_lang=body.targetLang
    ).time():
        results = await translator.translate(
            http=request.app.state.http,
            segments=[{"id": s.id, "text": s.text} for s in body.segments],
            source_lang=body.sourceLang,
            target_lang=body.targetLang,
        )

    logger.info(
        "translated %d segments %s→%s",
        len(results),
        body.sourceLang,
        body.targetLang,
        extra={"request_id": getattr(request.state, "request_id", "")},
    )

    return TranslateResponse(
        results=[SegmentOut(id=r["id"], text=r["text"], confidence=r["confidence"]) for r in results]
    )


@app.get("/metrics")
def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
