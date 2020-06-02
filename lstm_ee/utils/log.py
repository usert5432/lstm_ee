"""
Functions to setup logging.
"""

import logging
import os

def setup_logging(level = logging.DEBUG, log_file = None):
    """Setup logging."""

    logger = logging.getLogger()

    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s]: %(levelname)s %(message)s'
    )

    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if log_file is not None:
        dirname  = os.path.dirname(log_file)
        os.makedirs(dirname, exist_ok = True)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger

