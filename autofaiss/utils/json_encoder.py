"""JSON encoder for numpy types"""
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
