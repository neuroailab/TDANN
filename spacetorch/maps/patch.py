from typing import Tuple

import matplotlib.patches
import numpy as np
from scipy.spatial.distance import cdist
import shapely.geometry

from spacetorch.datasets import floc
from spacetorch.utils.spatial_utils import concave_hull


class Patch:
    """Represents a category-selective patch. Patches are usually contained in a list
    in their corresponding TisseMap

    positions: an N x 2 matrix of the unit positions in the patch
    unit_indices: list of length N of the indices of the positions in the parent
        TissueMap that this patch is composed of
    contrast: the contrast used to define the patch
    center: A 2-d vector of the (x, y) position of the patch center
    area: the area in mm ^ 2 of the patch
    """

    def __init__(
        self,
        positions: np.ndarray,
        unit_indices: np.ndarray,
        selectivities: np.ndarray,
        contrast: floc.Contrast,
        hull_alpha: float = 0.1,
    ):
        self.positions = positions
        self.unit_indices = unit_indices
        self.contrast = contrast
        self.selectivities = selectivities

        # compute the convex hull of the points
        self._points = [shapely.geometry.Point(p) for p in self.positions]
        self.concave_hull = concave_hull(self._points, alpha=hull_alpha)

    def contains(self, x: float, y: float) -> bool:
        return self.concave_hull.contains(shapely.geometry.Point(x, y))

    @property
    def center(self) -> np.ndarray:
        return np.array(self.concave_hull.centroid.coords)[0]

    @property
    def area(self) -> float:
        return self.concave_hull.area

    @property
    def hull_vertices(self) -> np.ndarray:
        """Returns a list of the vertices of the convex hull of this patch. Sort of
        complicated implementation, but we need to access the x, y coordinates
        of the exterior of the convex hull, spread the x and y arrays respectively,
        join them (with zip) into a set of (x, y) tuples, then cast to a list and
        finally a numpy array
        """
        return np.array(list(zip(*self.concave_hull.exterior.coords.xy)))

    def __repr__(self) -> str:
        return (
            f"Patch of {self.contrast.name} with {len(self.unit_indices)} units,"
            f" centered at {self.center}, total area: {self.area:.1f}mm^2"
        )

    def to_mpl_poly(
        self,
        alpha: float = 0.6,
        lw: float = 2,
        hollow: bool = False,
        scaling: float = 1.0,
    ):
        edgecolor = self.contrast.color if hollow else "white"
        fill = False if hollow else True

        return matplotlib.patches.Polygon(
            self.hull_vertices * scaling,
            facecolor=self.contrast.color,
            alpha=alpha,
            edgecolor=edgecolor,
            lw=lw,
            fill=fill,
        )


def labels_to_unit_indices(labels, positions, extent: Tuple[float, float]):
    _BACKGROUND_CLUSTER = 0

    unique_labels = np.unique(labels)

    index_sets = []
    for lab in unique_labels:
        if lab == _BACKGROUND_CLUSTER:
            continue

        # get rows and columns. flipud since rows go up as y goes down
        matching_rows, matching_cols = np.nonzero(lab == np.flipud(labels))
        if len(matching_rows) == 0:
            continue

        # convert each set of coordinates into unit coordinate space
        matching_y = matching_rows / len(labels) * extent[0]
        matching_x = matching_cols / len(labels) * extent[1]

        # get all unit indices that "belong" to this patch by measuring minimum
        # distance to a member of the patch
        matching_pos = np.stack((matching_x, matching_y), axis=1)
        distances = cdist(matching_pos, positions)
        min_distance_for_each_unit = np.min(distances, axis=0)

        # slight dilation to allow units on edge to be included
        min_cutoff = 1
        units_in_cluster = np.nonzero(min_distance_for_each_unit < min_cutoff)[0]
        if len(units_in_cluster) < 3:
            continue

        index_sets.append(units_in_cluster)

    return index_sets
