"""
Common setup functions used in autofaiss.
"""
import logging
import multiprocessing
from typing import Optional

import faiss


logger = logging.getLogger(__name__)


def setup_logging(logging_level: int):
    """Setup the logging."""
    logging.config.dictConfig(dict(version=1, disable_existing_loggers=False))
    logging_format = "%(asctime)s [%(levelname)s]: %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)


def setup_faiss_threads(nb_cores: Optional[int]) -> None:
    """Setup faiss threads according to given cores or machine's capacity."""
    if nb_cores is None:
        nb_cores = multiprocessing.cpu_count()
    logger.info(f"Using {nb_cores} omp threads (processes), consider increasing --nb_cores if you have more")
    faiss.omp_set_num_threads(nb_cores)
