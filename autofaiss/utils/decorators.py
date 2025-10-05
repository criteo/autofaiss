""" Useful decorators for fast debuging """

import functools
import time
import logging
from contextlib import ContextDecorator
from datetime import datetime
from typing import Optional

logger = logging.getLogger("autofaiss")


class Timeit(ContextDecorator):
    """Timing class, used as a context manager"""

    def __init__(self, comment: Optional[str] = None, indent: int = 0, verbose: bool = True):
        self.start_time = 0
        self.comment = comment
        self.indent = indent
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            if self.comment is not None:
                space = "\t" * self.indent
                start_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                logger.info(f"{space}{self.comment} {start_date}")
                # flush to make sure we display log in stdout before entering in the wrapped function
                for h in logger.handlers:
                    h.flush()
            self.start_time = time.perf_counter()

    def __exit__(self, *exc):
        if self.verbose:
            end_time = time.perf_counter()
            run_time = end_time - self.start_time
            space = "\t" * self.indent
            logger.info(f'{space}>>> Finished "{self.comment}" in {run_time:.4f} secs')


timeit = Timeit()


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def should_run_once(func):
    """
    Decorator to force a function to run only once.
    The fonction raises a ValueError otherwise.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if wrapper.has_run:
            raise ValueError("Can't call this function twice")
        wrapper.has_run = True
        return func(*args, **kwargs)

    wrapper.has_run = False
    return wrapper
