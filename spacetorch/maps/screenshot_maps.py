"""
Screenshot maps are constructed from portions of JPEGs taken from published results,
for the purposes of reconstructing data from empirical work
"""
from pathlib import Path
from typing import Tuple, Optional, List, Dict

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from numpy.random import default_rng
from PIL import Image
from scipy.spatial.distance import cdist, pdist, squareform
import skimage.measure
from tqdm import tqdm

from spacetorch.datasets.floc import DOMAIN_CONTRASTS, Contrast
from spacetorch.maps.it_map import Patch, labels_to_unit_indices
from spacetorch.paths import git_root
from spacetorch.utils import spatial_utils, array_utils


class SimpleTissue:
    def __init__(self):
        self.data, self.cmap = self._prep_data()
        self.rng = default_rng(seed=424)

        rows, cols = self.data.shape
        row_grid, col_grid = np.meshgrid(range(rows), range(cols))
        ys = row_grid.ravel()
        xs = col_grid.ravel()
        scale_bar_px = (
            self.metadata["scale_bar_box"][2] - self.metadata["scale_bar_box"][0]
        )
        self.px_per_mm = scale_bar_px / self.metadata["scale_bar_len_mm"]
        self.positions = np.stack((xs, ys)).T / self.px_per_mm
        self.preferences = self.data.T.ravel()

    @property
    def metadata(self):
        raise NotImplementedError

    def _prep_data(self):
        raise NotImplementedError

    def _transform_diffs(self, diffs: np.ndarray) -> np.ndarray:
        return diffs

    def difference_over_distance(
        self,
        sample_size: int = 1_000,
        num_bins: int = 20,
        bin_edges: Optional[np.ndarray] = None,
        num_samples: int = 50,
        verbose: bool = True,
        shuffle: bool = False,
    ):
        curves = []
        for _ in tqdm(range(num_samples), disable=not verbose):
            indices = np.random.choice(
                np.arange(len(self.positions)), size=(sample_size,), replace=False
            )
            preferences = self.preferences[indices, np.newaxis]
            diffs = np.abs(preferences - preferences.T)

            # circular wrapping
            diffs = self._transform_diffs(diffs)

            # pairwise distances
            dists = squareform(pdist(self.positions[indices]))
            if shuffle:
                self.rng.shuffle(dists)

            # relationships
            means, _, bin_edges = spatial_utils.agg_by_distance(
                array_utils.lower_tri(dists),
                array_utils.lower_tri(diffs),
                bin_edges=bin_edges,
                num_bins=num_bins,
            )

            curves.append(means)

        midpoints = array_utils.midpoints_from_bin_edges(bin_edges)
        return midpoints, curves


class NauhausOrientationTissue(SimpleTissue):
    def _prep_data(self):
        raw_img = Image.open(self.metadata["path"])
        data = np.array(raw_img.crop(box=self.metadata["map_box"])) / 255.0
        legend_data = np.array(raw_img.crop(box=self.metadata["legend_box"])) / 255.0
        return self._get_orientation_estimated_values(data, legend_data)

    @property
    def metadata(self):
        return {
            "path": git_root
            / "notebooks/assets/F02/41593_2012_Article_BFnn3255_Fig2_HTML.jpg",
            "map_box": [52, 118, 420, 370],
            "scale_bar_box": [563, 950, 810, 1000],
            "scale_bar_len_mm": 0.5,
            "legend_box": [955, 15, 1010, 470],
        }

    def _transform_diffs(self, diffs: np.ndarray) -> np.ndarray:
        diffs[np.where(diffs >= 90)] = 180 - diffs[np.where(diffs >= 90)]
        return diffs

    def _get_orientation_estimated_values(
        self, data: np.ndarray, legend_data: np.ndarray
    ) -> Tuple[np.ndarray, LinearSegmentedColormap]:

        # create a linear colormap from the legend entries
        legend_positions = [
            [25, 25],
            [25, 90],
            [25, 160],
            [25, 225],
            [25, 290],
            [25, 360],
            [25, 420],
        ]
        color_list = [legend_data[pos[1], pos[0]] for pos in legend_positions]
        cmap = LinearSegmentedColormap.from_list("", color_list)

        # define 500 equally spaced and indexed angles, then get the color that maps to
        # each angle
        indexers = np.linspace(0, 1, 500)
        angles = np.linspace(0, 180, 500)
        dense_mapping = np.array([cmap(i) for i in indexers])

        # find the closest angle for each pixel of the raw image
        alpha_dim = np.ones(data.shape[:-1])[..., np.newaxis]
        augmented_data = np.concatenate((data, alpha_dim), axis=-1)
        flat_data = np.reshape(augmented_data, (-1, 4))

        distances = cdist(flat_data, dense_mapping)
        closest = np.argmin(distances, axis=1)
        closest_angle = angles[closest]
        closest_angle_im = np.reshape(closest_angle, data.shape[:-1])
        return closest_angle_im, cmap


class NauhausSFTissue(SimpleTissue):
    def _prep_data(self):
        raw_img = Image.open(self.metadata["path"])
        data = np.array(raw_img.crop(box=self.metadata["map_box"])) / 255.0
        legend_data = np.array(raw_img.crop(box=self.metadata["legend_box"])) / 255.0
        return self._get_sf_estimated_values(data, legend_data)

    @property
    def metadata(self):
        return {
            "path": git_root
            / "notebooks/assets/F02/41593_2012_Article_BFnn3255_Fig2_HTML.jpg",
            "map_box": [52, 613, 420, 865],
            "scale_bar_box": [563, 950, 810, 1000],
            "scale_bar_len_mm": 0.5,
            "legend_box": [955, 510, 1050, 965],
        }

    def _get_sf_estimated_values(
        self, data: np.ndarray, legend_data: np.ndarray
    ) -> Tuple[np.ndarray, LinearSegmentedColormap]:

        # create a linear colormap from the legend entries
        legend_positions = [
            [25, 30],
            [25, 75],
            [25, 125],
            [25, 175],
            [25, 225],
            [25, 275],
            [25, 325],
            [25, 375],
            [25, 420],
        ]
        color_list = [legend_data[pos[1], pos[0]] for pos in legend_positions]
        cmap = LinearSegmentedColormap.from_list("", color_list)

        # define 500 equally spaced and indexed sfs, then get the color that maps onto
        # each sf
        indexers = np.linspace(0, 1, 500)
        sfs = np.logspace(start=0, stop=3, base=2, num=500)
        dense_mapping = np.array([cmap(i) for i in indexers])

        # find the closest angle for each pixel of the raw image
        alpha_dim = np.ones(data.shape[:-1])[..., np.newaxis]
        augmented_data = np.concatenate((data, alpha_dim), axis=-1)
        flat_data = np.reshape(augmented_data, (-1, 4))

        distances = cdist(flat_data, dense_mapping)
        closest = np.argmin(distances, axis=1)
        closest_sf = np.log2(sfs[closest])
        closest_sf_im = np.reshape(closest_sf, data.shape[:-1])
        return closest_sf_im, cmap


class LivingstoneColorTissue(SimpleTissue):
    def _prep_data(self):
        raw_img = Image.open(self.metadata["path"])
        data = np.array(raw_img.crop(box=self.metadata["map_box"])) / 255.0
        color_list = [data[75, 100], data[150, 130]]
        cmap = LinearSegmentedColormap.from_list("", color_list)

        indexers = np.linspace(0, 1, 500)
        colors = np.linspace(0, 1, 500)
        dense_mapping = np.array([cmap(i) for i in indexers])

        data = data[..., :3]
        alpha_dim = np.ones(data.shape[:-1])[..., np.newaxis]
        augmented_data = np.concatenate((data, alpha_dim), axis=-1)
        flat_data = np.reshape(augmented_data, (-1, 4))

        distances = cdist(flat_data, dense_mapping)
        closest = np.argmin(distances, axis=1)
        closest_color = colors[closest]
        data = np.reshape(closest_color, data.shape[:-1])
        return data, cmap

    @property
    def metadata(self):
        return {
            "path": git_root / "notebooks/assets/F02/Livingstone_1984_Figure27B.png",
            "map_box": [500, 700, 700, 900],
            "scale_bar_box": [870, 1200, 928, 1300],
            "scale_bar_len_mm": 1.0,
        }


class ITNScreenshotMaps:
    def __init__(self, maps: Dict[str, np.ndarray], positions: np.ndarray):
        self.maps = maps
        self.positions = positions
        self.patches: List[Patch] = []

    def category_smoothness(
        self,
        contrast: Contrast,
        num_samples: int = 100,
        distance_cutoff=6.0,
        bin_edges=None,
        shuffle: bool = False,
    ):
        sel = self.maps.get(contrast.name)
        if sel is None:
            raise ValueError(f"{contrast.name} is not in the maps dictionary")

        rng = default_rng(seed=424)
        profiles = []
        for _ in range(num_samples):
            indices = rng.choice(
                np.arange(len(self.positions)), size=(1000,), replace=False
            )
            # sel_mask = np.where(np.abs(sel[indices]) > 3)[0]
            # indices = indices[sel_mask]
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

            means, _, bin_edges = spatial_utils.agg_by_distance(
                distances, diff[dist_mask], num_bins=20, bin_edges=bin_edges
            )
            profiles.append(means)

        stacked_profiles = np.stack(profiles)
        return bin_edges, stacked_profiles

    def make_selectivity_map(
        self,
        axis,
        selectivity_threshold: float = 10.0,
        contrasts: Optional[List[Contrast]] = None,
        grayout=False,
        background_alpha: float = 0.1,
        foreground_alpha: float = 0.9,
        final_s=50,
        **kwargs,
    ):

        if contrasts is None:
            contrasts = DOMAIN_CONTRASTS

        axis.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            c="#eee",
            s=final_s,
            marker="s",
            alpha=1.0,
            **kwargs,
        )

        # foreground: selective units for each contrast
        for contrast in contrasts:
            sel = self.maps[contrast.name]
            selective_indices = sel > selectivity_threshold

            axis.scatter(
                self.positions[selective_indices, 0],
                self.positions[selective_indices, 1],
                c="gray" if grayout else contrast.color,
                s=final_s,
                label=contrast.name,
                alpha=foreground_alpha,
                marker="s",
                **kwargs,
            )

    @classmethod
    def from_png(cls, path: Path, tissue_size: float = 70.0):
        # load image
        image = Image.open(path)

        # define ROIs
        _width = 510
        _height = 510
        rois = {
            "Bodies": [17, 59, 17 + _width, 59 + _height],
            "Characters": [710, 59, 710 + _width, 59 + _height],
            "Faces": [1408, 59, 1408 + _width, 59 + _height],
            "Objects": [2102, 59, 2102 + _width, 59 + _height],
            "Places": [2798, 59, 2798 + _width, 59 + _height],
            "cbar": [560, 58, 580, 572],
        }

        # set data sampling points
        anchors = np.linspace(8, 500, 32)
        xx, yy = np.meshgrid(anchors, anchors)
        xx, yy = xx.astype(int), yy.astype(int)

        # sample the colorbar
        colorbar = np.array(image.crop(rois["cbar"])) / 255.0
        sample_points = np.array([[10, x] for x in range(5, colorbar.shape[0] - 5)])
        color_list = [colorbar[pos[1], pos[0]] for pos in sample_points[::-1]]
        cmap = LinearSegmentedColormap.from_list("", color_list)

        indexers = np.linspace(0, 1, 500)
        values = np.linspace(-15, 15, 500)
        dense_mapping = np.array([cmap(i)[:-1] for i in indexers])

        # create maps
        maps = {}

        for roi_name, roi in rois.items():
            if roi_name == "cbar":
                continue
            raw_data = np.array(image.crop(roi))
            resampled = raw_data[yy, xx, :3] / 255.0
            is_black = resampled.prod(axis=2) == 0

            flat_data = resampled.reshape((-1, 3))
            distances = cdist(flat_data, dense_mapping)
            closest = np.argmin(distances, axis=1)
            closest_val = values[closest]
            closest_val_im = np.reshape(closest_val, resampled.shape[:-1])
            closest_val_im[is_black] = np.nan
            maps[roi_name] = closest_val_im

        sels = {k: v.ravel() for k, v in maps.items()}
        posx, posy = np.meshgrid(np.arange(32), np.arange(32))
        positions = np.stack((posx.ravel(), posy.ravel())).T / 32.0 * tissue_size
        return cls(sels, positions)

    def find_patches(
        self,
        contrast: Contrast,
        threshold: float = 8.0,
        minimum_size: float = 100,
        maximum_size: float = 4500,
        min_count: int = 0,
        sigma: float = 0.7,
    ):
        sel = self.maps[contrast.name]
        isnan = np.isnan(sel)
        valid_pos = self.positions[~isnan]
        valid_sel = sel[~isnan]

        # create smoothing anchors
        n_anchors = 100
        anchors = np.linspace(np.min(self.positions), np.max(self.positions), n_anchors)
        cx, cy = np.meshgrid(anchors, anchors)

        # smooth
        smoothed = np.zeros((n_anchors, n_anchors))

        for row in range(n_anchors):
            for col in range(n_anchors):
                center = (cx[row, col], cy[row, col])

                dist_from_center = array_utils.gaussian_2d(
                    valid_pos, center, sigma=sigma
                )
                weighted = np.average(valid_sel, weights=dist_from_center)
                smoothed[-row, col] = weighted

        smoothed[smoothed < threshold] = 0
        labels = skimage.measure.label(smoothed > 0)

        clusters: List[np.ndarray] = labels_to_unit_indices(
            labels, self.positions, np.ptp(self.positions, axis=0)
        )

        for _, cluster in enumerate(clusters):
            try:
                patch = Patch(
                    positions=self.positions[cluster],
                    unit_indices=cluster,
                    selectivities=sel[cluster],
                    contrast=contrast,
                )
            except Exception:
                continue
            if (
                patch.area >= minimum_size
                and patch.area <= maximum_size
                and len(patch.unit_indices) >= min_count
            ):
                self.patches.append(patch)

        return smoothed
