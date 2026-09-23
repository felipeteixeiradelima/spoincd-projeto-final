import logging
import os
from pathlib import Path
import sys

import colorlog

_CONSOLE_FORMAT = (
    "%(green)s%(asctime)s%(reset)s | "
    "%(white)s%(levelname)-8s%(reset)s | "
    "%(blue)s%(name)s:%(funcName)s:%(lineno)d%(reset)s - "
    "%(white)s%(message)s%(reset)s"
)

_FILE_FORMAT = "%(asctime)s |  %(levelname)-8s |  %(name)s:%(funcName)s:%(lineno)d - %(message)s"

_LOGGER_SETUP_DONE = False


def setup_logger(
    log_file_path: str | os.PathLike[str] | Path = ".log", encoding: str = "utf-8"
) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    console_formatter = colorlog.ColoredFormatter(_CONSOLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    file_formater = logging.Formatter(_FILE_FORMAT)
    file_handler = logging.FileHandler(log_file_path, encoding=encoding)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formater)
    root_logger.addHandler(file_handler)
    global _LOGGER_SETUP_DONE
    _LOGGER_SETUP_DONE = True


def get_logger(name: str | None) -> logging.Logger:
    if not _LOGGER_SETUP_DONE:
        setup_logger()

    logger = logging.getLogger(name)
    return logger
