"""Standardized logging module for UTH Computer Vision Coursework."""

import logging
import sys
from pathlib import Path


def get_logger(
    name: str = "uth_cv_coursework",
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Get or configure a standardized logger instance.

    Args:
        name: Name of the logger, typically __name__ or module identifier.
        log_file: Optional path to output log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

    return logger
