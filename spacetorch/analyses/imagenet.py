import numpy as np
import xarray as xr

from spacetorch.datasets.imagenet import ImageNetResponses
from spacetorch.maps import TissueMap
from spacetorch.utils import array_utils


def create_imnet_tissue(features: np.ndarray, positions: np.ndarray):
    """Creates a simple tissue map from generic features"""
    _data = xr.DataArray(
        data=array_utils.flatten(features),
        dims=["image_idx", "unit_idx"],
    )

    responses = ImageNetResponses(_data=_data)
    return TissueMap(positions, responses)
