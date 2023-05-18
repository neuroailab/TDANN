from dataclasses import dataclass
from typing import Tuple

import numpy as np

from spacetorch.types import Dims


@dataclass
class FlatIndices:
    """FlatIndices stores the x, y, and channel index of units along a flattened array.

    For example, consider a tensor with shape (128, 28, 28): 128 channels, 28 rows, 28
    columns.

    We often flatten these dimensions so that a single axis can be used to index any
    unit, e.g. (128, 28, 28) -> (100352,). Indexing back into the original channel,
    row, and column can be useful
    """

    chan_flat: np.ndarray
    x_flat: np.ndarray
    y_flat: np.ndarray

    def from_dims(cls, dims: Dims) -> "FlatIndices":
        """
        dims should be a CHW tuple
        """
        num_channels, num_x, num_y = dims

        # chan flat goes like [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...]
        chan_flat = np.repeat(np.arange(num_channels), num_x * num_y)

        # x flat goes like [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, ...]
        x_flat = np.repeat(np.tile(np.arange(num_x), num_channels), num_y)

        # y flat goes like [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, ...]
        y_flat = np.tile(np.arange(num_y), num_x * num_channels)

        return cls(chan_flat, x_flat, y_flat)
