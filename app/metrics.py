from prometheus_client import Counter, Histogram, Gauge

translate_requests_total = Counter(
    "ml_translate_requests_total",
    "Total translation requests",
    ["source_lang", "target_lang"],
)

translate_segments_total = Counter(
    "ml_translate_segments_total",
    "Total segments translated by provider",
    ["provider"],  # "model" | "gemini" | "fallback"
)

translate_latency_seconds = Histogram(
    "ml_translate_latency_seconds",
    "End-to-end translation latency",
    ["source_lang", "target_lang"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

gemini_requests_total = Counter(
    "ml_gemini_requests_total",
    "Gemini API calls by outcome",
    ["outcome"],  # "success" | "error" | "retry"
)

model_loaded = Gauge(
    "ml_model_loaded",
    "1 if own model is loaded, 0 otherwise",
)
