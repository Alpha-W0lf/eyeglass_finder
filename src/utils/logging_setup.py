from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str | Path] = None) -> None:
    """
    Configure loguru with a console sink and an optional file sink.

    Args:
        log_level: Minimum level for logs (e.g., "DEBUG", "INFO").
        log_file: Optional path for a rotating log file.
    """
    logger.remove()

    # Console sink
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level.upper(),
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=log_level.upper(),
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
        )


def configure_worker_logging():
    """
    Configures a basic logger for a worker process.
    This avoids file-based logging and other complexities not needed
    for transient worker processes.
    """
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def get_logger():
    """Return the configured global logger."""
    return logger
