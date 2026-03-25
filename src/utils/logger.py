"""Centralised logging configuration for SentimentScribe."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a logger with a consistent format.

    Parameters
    ----------
    name:
        Logger name (use ``__name__`` in calling modules).
    level:
        Logging level (default INFO).
    log_file:
        Optional path to write logs to a file in addition to stdout.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
