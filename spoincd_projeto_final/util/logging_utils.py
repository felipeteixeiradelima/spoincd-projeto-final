import logging


def get_logger(name: str | None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
