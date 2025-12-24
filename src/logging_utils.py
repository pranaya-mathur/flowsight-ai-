import logging
from logging import Logger
from typing import Optional
from config import settings


def get_logger(name: Optional[str] = None) -> Logger:
    logger = logging.getLogger(name or "flowsight")
    if logger.handlers:
        return logger

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger
