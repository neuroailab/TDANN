from typing import List
from typing_extensions import Literal

import numpy as np
import scipy.stats as stats

from spacetorch.datasets import floc
from spacetorch.maps.it_map import ITMap
from spacetorch.maps.nsd_floc import Subject
from spacetorch.utils import array_utils

CONTRAST_ORDER = ["Faces", "Bodies", "Characters", "Places", "Objects"]
CONTRASTS = floc.DOMAIN_CONTRASTS
CONTRAST_DICT = {c.name: c for c in CONTRASTS}
ORDERED_CONTRASTS = [CONTRAST_DICT[name] for name in CONTRAST_ORDER]


def get_human_rsm(
    subject: Subject,
    hemi: Literal["left_hemi", "right_hemi"],
    contrasts: List[str] = CONTRAST_ORDER,
) -> np.ndarray:
    stack = []
    for contrast in contrasts:
        data = getattr(subject.data[contrast.lower()], hemi).ravel()
        data = data[~np.isnan(data)]
        stack.append(data)
    rsm = np.corrcoef(np.stack(stack))

    # nan diag
    rsm[np.diag_indices_from(rsm)] = np.nan
    return rsm


def get_model_rsm(
    tissue: ITMap,
    ordered_contrasts: List[floc.Contrast] = ORDERED_CONTRASTS,
    is_itn: bool = False,
) -> np.ndarray:
    stack = []
    for contrast in ordered_contrasts:
        if is_itn:
            stack.append(tissue.maps[contrast.name].ravel())
        else:
            stack.append(
                tissue.responses.selectivity(on_categories=contrast.on_categories)
            )

    stacked = np.stack(stack)
    nan = np.isnan(stacked)
    nan_col = np.any(nan, axis=0)
    safe = ~nan_col
    stacked = stacked[:, safe]

    rsm = np.corrcoef(stacked)
    # nan diag
    rsm[np.diag_indices_from(rsm)] = np.nan
    return rsm


def rsm_similarity(rsm_a: np.ndarray, rsm_b: np.ndarray) -> float:
    lower_tri_a = array_utils.lower_tri(rsm_a)
    lower_tri_b = array_utils.lower_tri(rsm_b)
    return stats.kendalltau(lower_tri_a, lower_tri_b).statistic
