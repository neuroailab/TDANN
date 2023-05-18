from typing import Optional, List
import uuid

import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import pdist, squareform
from scipy import interpolate
from scipy.stats import scoreatpercentile
from tqdm import tqdm

from spacetorch.datasets.sine_gratings import SineResponses, SineGrating2019, Metric
from spacetorch.paths import CACHE_DIR

from spacetorch.utils.array_utils import lower_tri, midpoints_from_bin_edges
from spacetorch.utils.plot_utils import remove_spines
from spacetorch.utils.spatial_utils import agg_by_distance
from spacetorch.utils import generic_utils, optimization_utils

from spacetorch.maps import TissueMap

# constants
metrics: List[Metric] = SineGrating2019.get_metrics()
metric_dict = SineGrating2019.get_metrics(as_dict=True)
angle_metric = metric_dict["angles"]


def interp(xs, ys, new_xs):
    interp_fn = interpolate.interp1d(xs, ys)
    return interp_fn(new_xs)


def tc_fits_from_cache(cache_id: str):
    tc_cache_dir = CACHE_DIR / "tuning_curve_fits" / cache_id
    if tc_cache_dir.exists():
        fits = generic_utils.load_pickle(tc_cache_dir / "fits.pkl")
        responses = generic_utils.load_pickle(tc_cache_dir / "responses.pkl")
        return fits, responses
    return None


def write_tc_fits_to_cache(cache_id: str, fits, responses):
    tc_cache_dir = CACHE_DIR / "tuning_curve_fits" / cache_id
    tc_cache_dir.mkdir(exist_ok=True, parents=True)

    generic_utils.write_pickle(tc_cache_dir / "fits.pkl", fits)
    generic_utils.write_pickle(tc_cache_dir / "responses.pkl", responses)


class V1Map(TissueMap):
    responses: SineResponses
    NUM_INTERPOLATED_ANGLES: int = 50
    angles: np.ndarray = np.linspace(0, 180, 9)[:-1]

    def __init__(
        self,
        positions: np.ndarray,
        sine_responses: SineResponses,
        cache_id: Optional[str] = str(uuid.uuid1()),
        unit_mask: Optional[np.ndarray] = None,
        smooth_orientation_tuning_curves: bool = True,
    ):
        super().__init__(positions, responses=sine_responses, unit_mask=unit_mask)

        self.cache_id = cache_id
        self.rng = default_rng(seed=424)
        if smooth_orientation_tuning_curves:
            self._smooth_orientation_tuning_curves()
        else:
            raw_ori_pref = self.responses.get_preferences(angle_metric.name).data
            self.orientation_preferences = np.array(
                [self.angles[x] for x in raw_ori_pref]
            )

    def _smooth_orientation_tuning_curves(self):
        # attempt to pull from cache
        if self.cache_id is None:
            cached = None
        else:
            cached = tc_fits_from_cache(self.cache_id)
        expanded_angles = np.append(self.angles, 180)
        dense_angles = np.linspace(
            expanded_angles[0], expanded_angles[-1], self.NUM_INTERPOLATED_ANGLES
        )

        if cached is not None:
            fits, responses = cached
            self.orientation_tc_fits = fits
            self.responses = responses
        else:
            print("No cache found, constructing from scratch...")

            original_tc = self.responses.orientation_tuning_curves
            expanded_tc = [np.append(tc, tc[0]) for tc in original_tc]
            interpolated_tc = [
                interp(expanded_angles, tc, dense_angles)
                for tc in tqdm(expanded_tc, desc="Interpolating tuning curves")
            ]

            fits = [
                optimization_utils.fit_vonmises(tc)
                for tc in tqdm(
                    interpolated_tc, desc="Fitting interpolated tuning curves"
                )
            ]

            if self.cache_id is not None:
                write_tc_fits_to_cache(self.cache_id, fits, self.responses)

        self.orientation_tc_fits = fits
        fit_tc = [fit_result.fit for fit_result in self.orientation_tc_fits]
        self.orientation_preferences = np.array(
            [dense_angles[np.argmax(tc)] for tc in fit_tc]
        )

    def set_unit_mask_by_ptp_percentile(self, metric_name: str, pctile: int = 50):
        self.reset_unit_mask()
        tc = np.array(self.responses._data.groupby(metric_name).mean()).T
        ptps = np.ptp(tc, axis=1)
        cutoff = scoreatpercentile(ptps, pctile)
        self.unit_mask = np.nonzero(ptps > cutoff)[0]

    def get_preferences(self, metric: Metric):
        if metric.name == angle_metric.name:
            return self.orientation_preferences[self.unit_mask]

        return self.responses.get_preferences(metric.name).data[self.unit_mask]

    def get_unit_colors(self, metric: Metric = angle_metric) -> np.ndarray:
        """
        return the RGBA colors for each unit in an (N, 4) matrix
        """

        preferences = self.get_preferences(metric)

        if metric == angle_metric:
            return [angle_metric.colormap(pref / 180) for pref in preferences]

        color_options = np.array(
            [
                metric.colormap(x / float(metric.n_unique))
                for x in range(metric.n_unique)
            ]
        )
        return color_options[preferences]

    def metric_difference_over_distance(
        self,
        distance_cutoff: float = 4,
        metric: Metric = angle_metric,
        num_samples: int = 50,
        sample_size: int = 1_000,
        bin_edges: Optional[np.ndarray] = None,
        num_bins: int = 20,
        verbose: bool = True,
        shuffle: bool = False,
    ):
        curves = []
        preferences = self.get_preferences(metric)
        for _ in tqdm(range(num_samples), disable=not verbose):
            indices = self.rng.choice(
                np.arange(len(self.positions)), size=(sample_size,), replace=False
            )
            sample_preferences = preferences[indices, np.newaxis]
            diffs = lower_tri(np.abs(sample_preferences - sample_preferences.T))

            # circular wrapping
            if metric == angle_metric:
                diffs[np.where(diffs >= 90)] = 180 - diffs[np.where(diffs >= 90)]

            # pairwise distances
            dists = lower_tri(squareform(pdist(self.positions[indices])))
            if shuffle:
                self.rng.shuffle(dists)

            dist_mask = np.nonzero(dists <= distance_cutoff)[0]

            # relationships
            means, _, bin_edges = agg_by_distance(
                dists[dist_mask],
                diffs[dist_mask],
                bin_edges=bin_edges,
                num_bins=num_bins,
            )

            curves.append(means)

        midpoints = midpoints_from_bin_edges(bin_edges)
        return midpoints, curves

    def make_parameter_map(
        self,
        axis,
        metric: Metric = angle_metric,
        scale_points=True,
        num_colors=None,
        final_psm=1.0,
        final_s: Optional[float] = 1,
        **kwargs,
    ):
        """
        Plots parameter map for the given metric, e.g., "angles", "sfs", "colors"
        """
        axis_point_scale = axis.bbox.width / 1e3

        assert not isinstance(
            axis, (list, np.ndarray)
        ), "please only provide a single axis for plotting"

        colors = self.get_unit_colors(metric=metric)

        if scale_points:
            if metric.name == "angles":
                selectivity = 1.0 - self.responses.circular_variance[self.unit_mask]
            else:
                selectivity = self.responses.get_peak_heights(metric.name)[
                    self.unit_mask
                ]

            selectivity = np.where(np.isnan(selectivity), 0, selectivity)
            selectivity = (selectivity - np.min(selectivity)) / np.ptp(
                selectivity
            ) + 0.5
        else:
            selectivity = np.ones((len(self.positions),))

        point_sizes = (
            final_psm * self.point_size_multiplier * axis_point_scale * selectivity
        )

        # plot points
        handle = axis.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            s=final_s or point_sizes,
            c=colors,
            cmap=metric.colormap,
            **kwargs,
        )
        remove_spines(axis)
        return handle
