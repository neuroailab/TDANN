from dataclasses import dataclass
import math
from typing import List, Optional, Mapping, Callable, Any

import numpy as np
import scipy.stats
from tqdm import tqdm

from spacetorch.maps import TissueMap
from spacetorch.utils.spatial_utils import Window, indices_within_limits
from spacetorch.types import AggMode


@dataclass
class KernelParams:
    width: float
    stride: float


class Smoother:
    """
    Example usage:

    kernel_params = KernelParams(width=1.0, stride=0.5)
    smoother = Smoother(kernel_params)
    extractor: lambda tissue_map: tissue_map.responses.get_preferences("angles")
    smoothed = smoother(tissue_map, extractor, agg)
    """

    def __init__(self, kernel_params: KernelParams, verbose: bool = True):
        self.kernel_params = kernel_params
        self.verbose = verbose

    def _mode(self, arr: np.ndarray, high: Optional[int] = None):
        if len(arr) == 0:
            return np.nan
        counts = np.bincount(arr)
        return np.argmax(counts)

    def _mean(self, arr: np.ndarray, high: Optional[float] = None):
        if len(arr) == 0:
            return np.nan
        return np.mean(arr)

    def _circmean(self, arr: np.ndarray, high: float):
        if len(arr) == 0:
            return np.nan
        return scipy.stats.circmean(arr, high=high - 1)

    def _circstd(self, arr: np.ndarray, high: float):
        if len(arr) == 0:
            return np.nan
        cs = scipy.stats.circstd(arr, high=high - 1)
        return cs / len(arr)

    def agg(self, agg_mode: AggMode, *args, **kwargs) -> float:
        lookup: Mapping[str, Callable[[np.ndarray, Any], float]] = {
            "mode": self._mode,
            "mean": self._mean,
            "circmean": self._circmean,
            "circstd": self._circstd,
        }
        return lookup[agg_mode](*args, **kwargs)  # type: ignore

    def __call__(
        self,
        tissue_map: TissueMap,
        attr_extractor: Callable[[TissueMap], np.ndarray],
        agg_mode: AggMode = "mean",
        high: float = 8,
    ):
        # get values to smooth
        attr = attr_extractor(tissue_map)
        positions = tissue_map.positions

        assert len(attr) == len(
            positions
        ), "the number of positions and points to aggregate do not match"

        # get windows to compute aggregations within
        minx, miny = np.min(positions, axis=0)
        maxx, maxy = np.max(positions, axis=0)

        num_x = int(
            math.floor((maxx - self.kernel_params.width) / self.kernel_params.stride)
        )

        num_y = int(
            math.floor((maxy - self.kernel_params.width) / self.kernel_params.stride)
        )

        x_bin_starts = np.linspace(minx, maxx - self.kernel_params.width, num_x)
        y_bin_starts = np.linspace(miny, maxy - self.kernel_params.width, num_y)

        windows = []
        for x_start in tqdm(x_bin_starts, desc="rows", disable=not self.verbose):
            xlims = [x_start, x_start + self.kernel_params.width]
            for y_start in y_bin_starts:
                ylims = [y_start, y_start + self.kernel_params.width]

                indices = indices_within_limits(
                    positions, [xlims, ylims], unit_limit=None
                )

                windows.append(
                    Window(indices=indices, lims=[xlims, ylims], num_units=len(indices))
                )

        # compute aggregations
        agg_per_window: List[float] = [
            self.agg(agg_mode, attr[window.indices], high) for window in windows
        ]

        return np.fliplr(np.reshape(agg_per_window, (num_x, num_y))).T
