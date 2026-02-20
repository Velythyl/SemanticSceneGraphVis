import os
import random
import logging

import numpy as np

log = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("Random seed set as %s", seed)
