# pylint: disable=unused-import,missing-docstring

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("autofaiss")

from autofaiss.external.quantize import build_index, score_index, tune_index

from autofaiss.version import __author__, __version__

