from typing import List

import numpy as np
from skimage.measure import label

from spacetorch.maps.v1_map import V1Map
from spacetorch.datasets.sine_gratings import SineGrating2019
from spacetorch.analyses.sine_gratings import get_smoothed_map

# constants
_BACKGROUND_CLUSTER = 0

# load angle metric
angle_metric = SineGrating2019.get_metrics(as_dict=True)["angles"]  # type: ignore


def circdiff(x, y) -> float:
    """Circular difference between two angles in the range[0, 180]

    For example:
        circdiff(5, 3) = 2
        circdiff(179, 5) = 6
    """
    raw = x - y
    raw = (raw + 90) % 180 - 90
    return raw


def increments(angles) -> List[float]:
    """
    Given a set of angles, return the list of increments from angle to angle
    """
    return [circdiff(angles[i + 1], angles[i]) for i in range(len(angles) - 1)]


class PinwheelDetector:
    def __init__(self, tissue: V1Map, size_mult: float = 1.5, verbose=False):
        self.tissue = tissue

        # compute the smoothed orientation map
        self.smoothed = get_smoothed_map(
            tissue,
            angle_metric,
            final_width=1 * size_mult,
            final_stride=0.3 * size_mult,
            verbose=verbose,
        )

        # compute the circular standard deviation of the orientation map. High stddev
        # indicates that the units contributing to that value are heterogeneous
        self.var = get_smoothed_map(
            tissue,
            angle_metric,
            final_width=1 * size_mult,
            final_stride=0.3 * size_mult,
            verbose=verbose,
            agg="circstd",
        )

        self._get_winding_numbers()

    def _get_winding_numbers(self):
        rows, cols = self.smoothed.shape
        self.winding_numbers = np.zeros_like(self.smoothed)

        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                values = [
                    self.smoothed[row - 1, col - 1],  # NW
                    self.smoothed[row - 1, col - 0],  # N
                    self.smoothed[row - 1, col + 1],  # NE
                    self.smoothed[row - 0, col + 1],  # E
                    self.smoothed[row + 1, col + 1],  # SE
                    self.smoothed[row + 1, col - 0],  # S
                    self.smoothed[row + 1, col - 1],  # SW
                    self.smoothed[row + 0, col - 1],  # W
                ]
                incs = increments(values)
                rad_incs = np.radians(incs)
                wn = sum(rad_incs) / (2 * np.pi)
                self.winding_numbers[row, col] = wn

    def count_pinwheels(
        self, min_px_count: int = 2, thresh: float = 0.3, var_thresh: float = 0.3
    ):
        pos_passing = (self.var < var_thresh) & (self.winding_numbers > thresh)
        neg_passing = (self.var < var_thresh) & (self.winding_numbers < (-thresh))

        counts = [0, 0]
        self.centers: List[List[float]] = [[], []]
        for idx, mask in enumerate([pos_passing, neg_passing]):
            islands = label(mask)
            unique_labels = np.unique(islands)

            for lab in unique_labels:
                if lab == _BACKGROUND_CLUSTER:
                    continue

                # get rows and columns. flipud since rows go up as y goes down
                matching_px = np.stack(np.flipud(np.nonzero(lab == islands))).T
                if len(matching_px) < min_px_count:
                    continue
                counts[idx] += 1
                ctr = np.mean(matching_px, axis=0)
                self.centers[idx].append(ctr)

        return counts

    def plot(self, ax, s=5):
        mappable = ax.imshow(
            self.smoothed, cmap=angle_metric.colormap, interpolation="nearest"
        )
        if hasattr(self, "centers"):
            pos_centers, neg_centers = self.centers
            for xloc, yloc in pos_centers:
                ax.scatter(xloc, yloc, c="k", s=s, linewidths=0)

            for xloc, yloc in neg_centers:
                ax.scatter(xloc, yloc, c="w", s=s, linewidths=0)

        return mappable
