from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from spacetorch.constants import DVA_PER_IMAGE
from spacetorch.datasets import floc, sine_gratings, imagenet
from spacetorch.types import Dims

from spacetorch.utils.array_utils import lower_tri, sem, FlatIndices, get_flat_indices
from spacetorch.utils.spatial_utils import (
    get_adjacent_windows,
    agg_by_distance,
    Window,
    WindowParams,
    indices_within_limits,
)


@dataclass
class Retinotopy:
    eccentricity: np.ndarray
    polar_angle: np.ndarray


class TissueMap:
    def __init__(
        self,
        positions: np.ndarray,
        responses: Union[
            sine_gratings.SineResponses, floc.fLocResponses, imagenet.ImageNetResponses
        ],
        unit_mask: Optional[np.ndarray] = None,
    ):
        self.responses = responses
        self._positions = positions

        if unit_mask is None:
            unit_mask = np.arange(len(positions))
        self.unit_mask = unit_mask

        assert len(self.responses) == len(self._positions)

        # set default window params
        self.default_window_params = WindowParams()

    def reset_unit_mask(self):
        self.unit_mask = np.arange(self._positions.shape[0])

    def set_mask_by_limits(self, limits: List[List[float]]):
        # must reset unit mask for the indices from indices_within_limits to make sense!
        self.reset_unit_mask()
        self.unit_mask = indices_within_limits(self.positions, limits)

    def set_mask_by_pct_limits(self, limits: List[List[float]]):
        self.reset_unit_mask()
        absolute_limits = np.array(limits) / 100 * self.width
        self.set_mask_by_limits(absolute_limits.tolist())

    @property
    def positions(self) -> np.ndarray:
        """
        Returns positions, but after masking with the current unit mask
        """
        return self._positions[self.unit_mask]

    @positions.setter
    def positions(self, new_positions) -> None:
        """
        Returns positions, but after masking with the current unit mask
        """
        self._positions[self.unit_mask] = new_positions

    @property
    def density(self) -> float:
        """Density is the average number of units per mm"""
        windows: List[Window] = self.get_window_indices(
            WindowParams(width=1.0, window_number_limit=None, unit_number_limit=None)
        )

        # because the window is fixed at 1x1mm, we can just return the mean number
        # of units in each window
        return np.mean([window.num_units for window in windows])

    @property
    def width(self) -> float:
        return np.ptp(self.positions[:, 0])

    @property
    def point_size_multiplier(self) -> float:
        """
        Sets a reasonable default for the `s` parameter of pyplot.scatter based on the
        density of units: when the density is higher, dots should be smaller so we can
        see them all. In practice, this needs to be combined with some measure of how
        big the actual axis is.

        The constant 5e3 is found by playing with different options and eyeballing
        the best option.
        """
        return 5e3 / self.density

    @property
    def features(self) -> np.ndarray:
        return self.responses._data.values[:, self.unit_mask]

    @features.setter
    def features(self, new_features: np.ndarray):
        raise NotImplementedError(
            (
                "Sorry, setting responses isn't allowed, since these are generally "
                "DataArray objects that shouldn't have their indexing messed with"
            )
        )

    def get_window_indices(
        self,
        window_params: WindowParams,
        seed: int = None,
        shift: Optional[Tuple[float, float]] = None,
        spacing: float = 1.0,
    ) -> List[Window]:
        """
        Grabs indices corresponding to square windows of a given width
        Inputs
            width: how wide each window should be, in mm
            window_number_limit: if not None, how many windows to keep at most
            unit_number_limit: if not None, how many units to keep per window
            seed: random seed
            shift: a 2-tuple of how much to shift the grid in x and y

        Notes:
            1. Caching was considered and removed, since allowing units to
                move around kind of breaks the concept of a window to begin with
        """
        if seed is not None:
            assert isinstance(seed, int)
            np.random.seed(seed=seed)

        windows: List[Window] = get_adjacent_windows(
            self.positions,
            width=window_params.width,
            shift=shift,
            window_number_limit=window_params.window_number_limit,
            unit_number_limit=window_params.unit_number_limit,
            edge_buffer=window_params.edge_buffer,
            spacing=spacing,
        )
        return windows

    def correlation_over_distance_plot(
        self,
        axis,
        window_params: Optional[WindowParams],
        verbose: bool = True,
        legend_label: Optional[str] = None,
        num_line_bins: int = 15,
        normalize_x_axis: bool = False,
        line_color: Any = None,
    ):
        """
        Creates a plot of correlation vs. distance for pairs of units within
        a randomly selected window of width `window_width`
        """

        # get windows for analysis
        window_params = window_params or self.default_window_params
        windows = self.get_window_indices(window_params)

        profile_per_window = []
        bin_edges = None
        for window in tqdm(windows, desc="windows", disable=not verbose):
            feats = self.features[:, window.indices]
            corr = lower_tri(np.corrcoef(feats.T))
            distances = lower_tri(squareform(pdist(self.positions[window.indices])))

            nan_mask = np.isnan(corr)
            corr = corr[~nan_mask]
            distances = distances[~nan_mask]

            means, spreads, bin_edges = agg_by_distance(
                distances, corr, num_bins=num_line_bins, bin_edges=bin_edges
            )

            profile_per_window.append(means)

        # compute mean and SEM across windows
        stacked_profiles = np.stack(profile_per_window)
        grand_mean = np.mean(stacked_profiles, axis=0)
        se = sem(stacked_profiles, axis=0)

        # set x-axis ticks
        bin_width = np.diff(bin_edges)[0]
        distance_ticks = bin_edges[:-1] + bin_width / 2  # type: ignore

        if normalize_x_axis:
            distance_ticks = distance_ticks / np.max(distance_ticks)

        # plot the data
        line_handle = axis.plot(
            distance_ticks, grand_mean, label=legend_label, color=line_color
        )

        # fill between standard error
        axis.fill_between(
            distance_ticks,
            grand_mean - se,
            grand_mean + se,
            alpha=0.3,
            facecolor=line_handle[0].get_color(),
        )

        # set horizontal line at 0.0 for reference
        axis.axhline(0.0, linestyle="dashed", c="gray")
        axis.set_xlabel(r"$d_{ij}$ (mm)")
        axis.set_ylabel(r"$C_{ij}$")
        axis.set_ylim([-0.2, 0.5])
        return stacked_profiles

    @staticmethod
    def retinotopy(dims: Dims) -> Retinotopy:
        """Polar angle of each unit, relative to the center of the tissue map"""
        flat_indices: FlatIndices = get_flat_indices(dims)
        x_rfs = flat_indices.x_flat
        y_rfs = flat_indices.y_flat

        x_diff = x_rfs - np.mean(x_rfs)
        y_diff = y_rfs - np.mean(y_rfs)

        eccentricity = np.sqrt(x_diff**2 + y_diff**2)
        polar_angle = np.arctan2(y_diff, x_diff)

        # convert eccentricity to dva
        num_taps = dims[-1]
        dva_per_tap = DVA_PER_IMAGE / num_taps
        eccentricity = eccentricity * dva_per_tap

        # convert polar_angle to degrees
        polar_angle = np.degrees(polar_angle)

        return Retinotopy(eccentricity, polar_angle)
