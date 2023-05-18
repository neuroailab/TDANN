from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
from scipy.stats import scoreatpercentile
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from spacetorch.constants import RNG_SEED
from spacetorch.datasets import DatasetRegistry
from spacetorch.models.positions import LayerPositions
from spacetorch.types import RN18Layer
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.utils import array_utils, spatial_utils


class Shifts(Enum):
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()


def shift(centroids: np.ndarray, direction: Shifts, amount: float) -> np.ndarray:
    centroids = centroids.copy()

    if direction.value == Shifts.TOP.value:
        centroids[:, 1] = centroids[:, 1] + amount
    elif direction.value == Shifts.BOTTOM.value:
        centroids[:, 1] = centroids[:, 1] - amount
    elif direction.value == Shifts.LEFT.value:
        centroids[:, 0] = centroids[:, 0] - amount
    elif direction.value == Shifts.RIGHT.value:
        centroids[:, 0] = centroids[:, 0] + amount

    return centroids


def greedy_optimal_assignment_length(
    source_centroids, target_centroids
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    # build the pairwise distance matrix for all centroids
    distance_matrix = cdist(source_centroids, target_centroids)

    # if there is only one origination and termination, just return their distance
    if len(source_centroids) == 1:
        return distance_matrix[0][0], (0, 0)

    # otherwise, figure out the optimal assignment that minimizes total cost
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # the final cost is the sum of the optimal assignment lengths
    optimal_cost = distance_matrix[row_ind, col_ind].sum()
    return optimal_cost, (row_ind, col_ind)


@dataclass
class WLTissue:
    positions: np.ndarray
    responses: np.ndarray
    active_pctile: float
    kmeans_dist_thresh: float
    num_fibers: Optional[int] = None
    shuffle: bool = False

    def __post_init__(self):
        rng = default_rng(seed=RNG_SEED)

        if self.shuffle:
            rng.shuffle(self.positions)

        # figure out which target units are "on"
        passing_threshold = scoreatpercentile(self.responses, self.active_pctile)
        passing = np.nonzero(self.responses > passing_threshold)
        self.passing_pos = self.positions[passing]
        if self.num_fibers is None:
            self.num_fibers, self.centroids, self.labels = self._choose_num_fibers()
        else:
            kmeans = KMeans(n_init=10, n_clusters=self.num_fibers)
            kmeans.fit(self.passing_pos)

            self.centroids, self.labels = kmeans.cluster_centers_, kmeans.labels_

        self.intra_distance = 0
        for cluster_idx in range(self.num_fibers):
            center = self.centroids[cluster_idx]
            cluster_passing = self.passing_pos[self.labels == cluster_idx]
            distances = cdist(cluster_passing, [center]).squeeze()
            self.intra_distance += np.sum(distances)

    def _choose_num_fibers(self):
        _MAX_K = 100
        max_inertia = len(self.passing_pos) * (self.kmeans_dist_thresh**2)
        for k in range(1, _MAX_K):
            kmeans = KMeans(n_init=10, n_clusters=k)
            kmeans.fit(self.passing_pos)
            assert kmeans.inertia_ is not None
            if kmeans.inertia_ <= max_inertia:
                break

        return k, kmeans.cluster_centers_, kmeans.labels_  # type: ignore

    def plot(self, ax):
        ax.scatter(*self.passing_pos.T, c="k", s=10)
        for c in self.centroids:
            ax.scatter(*c, marker="x", s=500)


class WireLengthExperiment:
    def __init__(
        self,
        model: nn.Module,
        layer_positions: Dict[str, LayerPositions],
        source_layer: RN18Layer,
        target_layer: RN18Layer,
        num_patterns: int,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.layer_positions = layer_positions
        self.source_layer = source_layer
        self.target_layer = target_layer

        self.source_positions = self.layer_positions[self.source_layer].coordinates
        self.target_positions = self.layer_positions[self.target_layer].coordinates

        self.source_extent = np.ptp(self.source_positions[:, 0])

        self._generate_patterns(num_patterns=num_patterns)

    def _generate_patterns(self, num_patterns):
        batch_size = min(64, num_patterns)
        num_batches = math.ceil(num_patterns / batch_size)
        layers = [
            f"base_model.{layer}" for layer in [self.source_layer, self.target_layer]
        ]

        features, inputs, _labels = get_features_from_layer(
            self.model,
            DatasetRegistry.get("ImageNet"),
            layers,
            batch_size=batch_size,
            max_batches=num_batches,
            return_inputs_and_labels=True,
            verbose=False,
        )
        self.features = {
            layer.split("base_model.")[-1]: layer_features[:num_patterns]
            for layer, layer_features in features.items()
        }
        inputs = inputs[:num_patterns]

        # set the last image and label
        self.images = [array_utils.norm(x.transpose(1, 2, 0)) for x in inputs]

    def compute_wl(
        self,
        pattern_idx: int,
        active_pctile: float = 99,
        shuffle: bool = False,
        kmeans_dist_thresh: float = 1,
        lims: Optional[List[List[float]]] = None,
        direction: Shifts = Shifts.TOP,
    ):
        if lims is not None:
            self.source_ind = spatial_utils.indices_within_limits(
                self.source_positions, [lims[0], lims[1]]
            )
            self.target_ind = spatial_utils.indices_within_limits(
                self.target_positions, [lims[0], lims[1]]
            )
        else:
            self.source_ind = np.arange(len(self.source_positions))
            self.target_ind = np.arange(len(self.target_positions))

        # determine the pattern of activity in the input and output layers
        input_activity = self.features[self.source_layer][pattern_idx].ravel()[
            self.source_ind
        ]
        output_activity = self.features[self.target_layer][pattern_idx].ravel()[
            self.target_ind
        ]

        source_pos = self.source_positions[self.source_ind].copy()
        target_pos = shift(
            self.target_positions[self.target_ind].copy(),
            direction=direction,
            amount=self.source_extent,
        )

        # construct the "source tissue", which will include a sweep over "k"
        # to find the number of fibers required to serve each active unit in the source
        # layer
        source_tissue = WLTissue(
            positions=source_pos,
            responses=input_activity,
            active_pctile=active_pctile,
            kmeans_dist_thresh=kmeans_dist_thresh,
            shuffle=shuffle,
        )

        # construct the target layer by fixing the number of fibers to the number that
        # is coming from the source layer
        target_tissue = WLTissue(
            positions=target_pos,
            responses=output_activity,
            active_pctile=active_pctile,
            num_fibers=source_tissue.num_fibers,
            kmeans_dist_thresh=kmeans_dist_thresh,
        )

        self.tissues = {"source": source_tissue, "target": target_tissue}

        # store # of fibers we end up using
        self.num_fibers = source_tissue.num_fibers

        # the intra-layer distance in the target layer is how we previously computed
        # wiring length. It is stored here but doesn't affect the final inter-layer
        # wiring length we compute
        self.intra_distance = self.tissues["target"].intra_distance

        # compute inter-distance
        ##########
        target_centroids = self.tissues["target"].centroids
        source_centroids = self.tissues["source"].centroids

        centroid_dist, assignment = greedy_optimal_assignment_length(
            target_centroids, source_centroids
        )
        self.assignment = assignment

        self.inter_distance = centroid_dist
        return self.inter_distance
