import logging
import sys
from app.config import settings


def setup_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    if settings.LOG_FORMAT == "json":
        try:
            from pythonjsonlogger import jsonlogger

            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                jsonlogger.JsonFormatter(
                    "%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
        except ImportError:
            handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )

    logging.basicConfig(level=level, handlers=[handler], force=True)

    # Quieten noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
