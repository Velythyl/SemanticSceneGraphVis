from typing import List, Tuple

import numpy as np
from matplotlib.pyplot import get_cmap


def similarities_to_rgb(
    similarities: np.ndarray,
    cmap_name: str,
) -> List[Tuple[int, int, int]]:
    similarities = np.asarray(similarities, dtype=np.float32)
    if similarities.size == 0:
        return []

    minimum = float(similarities.min())
    maximum = float(similarities.max())
    if maximum == minimum:
        normalized = np.zeros_like(similarities, dtype=np.float32)
    else:
        normalized = (similarities - minimum) / (maximum - minimum)

    cmap = get_cmap(cmap_name)
    colors = [cmap(sim.item())[:3] for sim in normalized]
    return [(int(255 * c[0]), int(255 * c[1]), int(255 * c[2])) for c in colors]
