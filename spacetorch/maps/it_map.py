from typing import Optional, List, Callable
from numpy.random import default_rng

from scipy.spatial.distance import pdist, squareform
import skimage.measure
import numpy as np

from spacetorch.datasets.floc import fLocResponses, Contrast, DOMAIN_CONTRASTS
from spacetorch.maps import TissueMap
from spacetorch.maps.patch import labels_to_unit_indices, Patch
from spacetorch.paths import CACHE_DIR
from spacetorch.utils import array_utils, generic_utils, plot_utils, spatial_utils


class ITMap(TissueMap):
    responses: fLocResponses
    cache_id: Optional[str]

    def __init__(
        self,
        positions: np.ndarray,
        floc_responses: fLocResponses,
        unit_mask: Optional[np.ndarray] = None,
        cache_id=None,
    ):
        super().__init__(positions, responses=floc_responses, unit_mask=unit_mask)
        self.patches: List[Patch] = []
        self.cache_id = cache_id

    def find_patches(
        self,
        contrast: Contrast,
        threshold: float = 2,
        minimum_size: float = 100,
        maximum_size: float = 4500,
        min_count: int = 10,
        selectivity_fn: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ] = array_utils.tstat,
        verbose: bool = False,
    ):
        """
        Arguments:
            contrast: a Contrast to find patches for
            threshold: the threshold, in whatever units the data are defined in
            minimum_size: minimum size of the patches kept in square mm
        """
        # determine the cache id
        if self.cache_id is None:
            self.cache_id = "default"

        cache_probe = (
            f"{self.cache_id}"
            f"_{contrast.name}_"
            f"thr{threshold:.2f}_"
            f"min{minimum_size}_"
            f"max{maximum_size}_"
            f"mc{min_count}"
        )
        cache_loc = CACHE_DIR / "patches" / f"{cache_probe}.pkl"
        if cache_loc.exists():
            if verbose:
                print(f"Loading from cache {cache_loc}")
            patches = generic_utils.load_pickle(cache_loc)
        else:
            patches = []
            sel = self.responses.selectivity(
                selectivity_fn=selectivity_fn, on_categories=contrast.on_categories
            )

            # create smoothing anchors
            n_anchors = 100
            anchors = np.linspace(
                np.min(self.positions), np.max(self.positions), n_anchors
            )
            cx, cy = np.meshgrid(anchors, anchors)

            # smooth
            smoothed = np.zeros((n_anchors, n_anchors))

            for row in range(n_anchors):
                for col in range(n_anchors):
                    center = (cx[row, col], cy[row, col])
                    dist_from_center = array_utils.gaussian_2d(
                        self.positions, center, sigma=2.4
                    )
                    weighted = np.average(sel, weights=dist_from_center)
                    smoothed[-row, col] = weighted

            smoothed[smoothed < threshold] = 0
            labels = skimage.measure.label(smoothed > 0)
            clusters: List[np.ndarray] = labels_to_unit_indices(
                labels, self.positions, np.ptp(self.positions, axis=0)
            )

            for cluster in clusters:
                patch = Patch(
                    positions=self.positions[cluster],
                    unit_indices=cluster,
                    selectivities=self.responses.selectivity(
                        on_categories=contrast.on_categories
                    )[cluster],
                    contrast=contrast,
                )
                if (
                    patch.area >= minimum_size
                    and patch.area <= maximum_size
                    and len(patch.unit_indices) >= min_count
                ):
                    patches.append(patch)

            if verbose:
                print(f"Writing to cache {cache_loc}")
            generic_utils.write_pickle(cache_loc, patches)
        self.patches.extend(patches)

    def make_single_contrast_map(
        self,
        axis,
        contrast: Contrast,
        alpha: float = 0.9,
        final_psm=1.0,
        cmap="bwr",
        vmin=-15,
        vmax=15,
        **kwargs,
    ):
        # how small is this axis relative to a full-size axis?
        axis_point_scale = axis.bbox.width / 1e3

        # background: all dots in a light gray
        axis.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            c="gray",
            s=3 * self.point_size_multiplier * axis_point_scale * final_psm,
            alpha=0.1,
            **kwargs,
        )

        sel = self.responses.selectivity(contrast.on_categories)
        sort_ind = np.argsort(np.abs(sel))

        handle = axis.scatter(
            self.positions[sort_ind, 0],
            self.positions[sort_ind, 1],
            c=sel[sort_ind],
            s=np.abs(sel[sort_ind])
            * self.point_size_multiplier
            * axis_point_scale
            * final_psm,
            label=contrast.name,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            **kwargs,
        )
        plot_utils.remove_spines(axis)
        return handle

    def make_selectivity_map(
        self,
        axis,
        selectivity_fn: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ] = array_utils.tstat,
        selectivity_threshold: float = 10.0,
        contrasts: Optional[List[Contrast]] = None,
        grayout=False,
        background_alpha: float = 0.1,
        foreground_alpha: float = 0.9,
        size_mult: float = 0.1,
        scale_points: bool = True,
        marker="s",
        final_s: Optional[float] = None,
        **kwargs,
    ):
        axis_point_scale = axis.bbox.width / 1e3

        if contrasts is None:
            contrasts = DOMAIN_CONTRASTS

        # background: all dots in a light gray
        background_sizes = self.point_size_multiplier
        if scale_points:
            background_sizes *= size_mult * axis_point_scale

        axis.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            c="#eee",
            s=final_s or 50,
            alpha=background_alpha,
            marker=marker,
            **kwargs,
        )

        # foreground: selective units for each contrast
        for contrast in contrasts:
            sel = self.responses.selectivity(
                contrast.on_categories, selectivity_fn=selectivity_fn
            )[self.unit_mask]
            selective_indices = sel > selectivity_threshold

            sizes = self.point_size_multiplier
            if scale_points:
                sizes *= size_mult * axis_point_scale * np.abs(sel[selective_indices])

            axis.scatter(
                self.positions[selective_indices, 0],
                self.positions[selective_indices, 1],
                c="#eee" if grayout else contrast.color,
                s=final_s or sizes,
                label=contrast.name,
                alpha=foreground_alpha,
                marker=marker,
                **kwargs,
            )

    def category_smoothness(
        self,
        contrast: Contrast,
        num_samples: int = 500,
        sample_size: int = 500,
        distance_cutoff=60.0,
        bin_edges=None,
        shuffle: bool = False,
    ):
        """
        Inputs:
            contrast: the fLoc contrast to compute selectivity smoothness for
            num_samples: how many times to sample a group of units from the cortical
                sheet
            distance_cutoff: any pairwise distances beyond this point are dropped from
                the curve
        """
        rng = default_rng(seed=424)
        sel = self.responses.selectivity(contrast.on_categories)
        profiles = []
        mean_responses = self.responses._data.mean("image_idx").values
        for _ in range(num_samples):
            indices = rng.choice(
                np.arange(len(self.positions)), size=(sample_size,), replace=False
            )
            indices = indices[mean_responses[indices] > 0.5]

            subset_sel = sel[indices]
            subset_sel_expanded = subset_sel[:, np.newaxis]
            sel_diff = np.abs(subset_sel_expanded - subset_sel_expanded.T)

            diff = array_utils.lower_tri(sel_diff)
            distances = array_utils.lower_tri(
                squareform(pdist(self.positions[indices]))
            )
            dist_mask = np.nonzero(distances <= distance_cutoff)[0]
            distances = distances[dist_mask]
            if shuffle:
                rng.shuffle(distances)

            means, _spreads, bin_edges = spatial_utils.agg_by_distance(
                distances, diff[dist_mask], num_bins=20, bin_edges=bin_edges
            )
            profiles.append(means)

        stacked_profiles = np.stack(profiles)
        return bin_edges, stacked_profiles
