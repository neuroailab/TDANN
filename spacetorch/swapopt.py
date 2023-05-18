"""
Utilities for swapopt
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import numpy as np
from tqdm import tqdm
from vissl.utils.hydra_config import AttrDict

from spacetorch.feature_saver import FeatureSaver
import spacetorch.losses.losses_numpy as losses
from spacetorch.models.positions import LayerPositions
from spacetorch.utils.spatial_utils import (
    precompute_neighborhoods,
    collapse_and_trim_neighborhoods,
)
from spacetorch.paths import POSITION_DIR
from spacetorch.constants import RNG_SEED


def swap(positions, swap_ind):
    """
    Swaps xs and ys at the 2 indices at swap_ind, in place
    """
    positions[[swap_ind[0], swap_ind[1]]] = positions[[swap_ind[1], swap_ind[0]]]


def swap_optimize_positions(
    features,
    positions,
    num_steps=100,
    seed=None,
    loss_params=None,
    disable_progress_bar: bool = True,
    return_metrics: bool = False,
):
    """
    Iteratively swaps the positions of randomly selected pairs of units if
    doing so would reduce a specified loss function

    Inputs:
        features (n_images, n_units): response of each unit to each image
        positions (n_units, 2): positions of each unit
        num_steps (int): how many swaps to attempt
        seed (int): random seed to use (makes unit pair selection deterministic)
        loss params (dict): {
            loss_fn (callable): core loss_fn to compare swap efficacy
            dist_scaling (float): how much to multiply distances by in loss function
        }

    Outputs:
        ret_dict (dict): {
            pos{x,y} (n_units,): optimized positions in {x,y}
            n_swaps (int): number of swaps actually made
        }
    """

    if loss_params is None:
        loss_params = {"loss_fn": losses.pearson_scl, "dist_scaling": 12.0}

    loss_params = copy.deepcopy(loss_params)
    loss_fn = loss_params.pop("loss_fn", losses.standard_scl)

    # set random seed, if provided
    rng = np.random.default_rng(seed=RNG_SEED)

    # compute unit x unit correlations
    correlations = np.corrcoef(features.T)

    # NaN correlations go to 0
    correlations = np.where(
        np.isnan(correlations), np.zeros_like(correlations), correlations
    )

    # start counting how many swaps occurred
    n_swaps = 0
    for i in tqdm(range(num_steps), desc="swapopt", disable=disable_progress_bar):
        # pick a random pair of units to swap
        swap_ind = rng.choice(np.arange(positions.shape[0]), size=(2,), replace=False)

        # compute loss before swapping
        previous_loss = losses.swapopt_spatial_loss_wrapper(
            loss_fn, correlations, positions, swap_ind, **loss_params
        )

        # swap
        swap(positions, swap_ind)

        # compute new loss after swapping
        current_loss = losses.swapopt_spatial_loss_wrapper(
            loss_fn, correlations, positions, swap_ind, **loss_params
        )

        if current_loss >= previous_loss:
            swap(positions, swap_ind)
        else:
            n_swaps += 1

    if return_metrics:
        valid_inds = np.std(features, axis=0) > 0
        nb_loss = losses.spatial_loss_wrapper(
            loss_fn, features[:, valid_inds], positions[valid_inds], **loss_params
        )
        return positions, n_swaps, nb_loss

    return positions


@dataclass
class Metrics:
    num_swaps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)


def swap_optimize_neighborhoods(
    features: np.ndarray,
    positions: np.ndarray,
    num_steps: int = 10_000,
    steps_per_neighborhood: int = 500,
    neighborhood_width: float = 6,  # mm
    max_units_per_neighborhood: int = 500,
    seed: Optional[int] = None,
    disable_progress_bar: bool = False,
    loss_params: Optional[Dict[str, Any]] = None,
    return_metrics: bool = False,
    log_every: int = 1,
):
    """
    Wrapper around swap_optimize_positions that
    1. Selects a "neighborhood" (subset of tissue) at random
    2. Attempts steps_per_neighborhood swaps confined to that neighborhood
    3. Repeats until num_steps total steps have been attempted

    Inputs (see signature for swap_optimize_positions for more):
        features: (n_images, n_units)
        positions: (n_units, 2)
        num_steps: TOTAL number of steps, to be divided among neighborhoods
        steps_per_neighborhood: how many swaps to attempt in each neighborhood
        neighborhood_width: width (in mm) of neighborhoods to be selected
        seed, loss_params: see swap_optimize_positions

    Returns:
        ret_dict (dict): {
            pos{x,y} (n_units,): optimized positions in {x,y}
            n_swaps: number of swaps actually made
            losses: list of computed loss values for each iteration
        }
    """
    # safety checks
    assert neighborhood_width <= np.ptp(
        positions[:, 0]
    ) and neighborhood_width <= np.ptp(
        positions[:, 1]
    ), "neighborhood width cannot be larger than map limits in either dimension"

    assert (
        num_steps >= steps_per_neighborhood
    ), "cannot have fewer total steps than number of steps in a single neighborhood"

    metrics = Metrics()

    # set random seed, if provided
    rng = np.random.default_rng(seed=RNG_SEED)

    num_neighborhoods = int(np.ceil(num_steps / steps_per_neighborhood))

    for neighborhood_idx in tqdm(
        range(num_neighborhoods), desc="swapopt", disable=disable_progress_bar
    ):
        start_x = rng.uniform(
            low=np.min(positions[:, 0]),
            high=np.max(positions[:, 0]) - neighborhood_width,
        )
        start_y = rng.uniform(
            low=np.min(positions[:, 1]),
            high=np.max(positions[:, 1]) - neighborhood_width,
        )

        # find indices of units within the bounding box
        indices = np.where(
            (positions[:, 0] >= start_x)
            & (positions[:, 0] <= start_x + neighborhood_width)
            & (positions[:, 1] >= start_y)
            & (positions[:, 1] <= start_y + neighborhood_width)
        )[0]

        # don't bother if there aren't even 5 units
        if len(indices) < 5:
            continue

        # filter down to unit limit
        if len(indices) >= max_units_per_neighborhood:
            indices = np.rng.choice(
                indices, size=(max_units_per_neighborhood,), replace=False
            )

        # run swapopt without progress bar, restricted to neighborhood units
        log_this_iteration = neighborhood_idx % log_every == 0

        output = swap_optimize_positions(
            features[:, indices],
            np.copy(positions[indices, :]),
            num_steps=steps_per_neighborhood,
            seed=seed,
            loss_params=loss_params,
            disable_progress_bar=True,
            return_metrics=log_this_iteration,
        )

        if log_this_iteration:
            positions[indices], n_swaps, loss = output

            metrics.num_swaps.append(n_swaps)
            metrics.losses.append(loss)
        else:
            positions[indices] = output

    if return_metrics:
        return positions, metrics

    return positions


class Swapper:
    def __init__(
        self, config: AttrDict, feature_path: str, layer: str, dataset_name: str
    ):
        self.config = config
        self.layer = layer
        self.feature_path = feature_path
        self.dataset_name = dataset_name

        # determine initial and new save locations
        self.initial_dir = POSITION_DIR / self.config.initial_position_dir

        # the new save dir _is_ model specific, because the positions now depend on the
        # features. So we need to drill down and make a new folder
        self.new_save_dir = (
            self.initial_dir.parent
            / self.config.name
            / f"{self.initial_dir.name}_swappedon_{self.dataset_name}"
        )

        # check if positions already exist
        self.blocked = False
        if (self.new_save_dir / f"{self.layer}.pkl").exists():
            self.blocked = True

        self._load_positions()
        self._load_features()

    def unblock_overwrite(self):
        self.blocked = False

    def _load_positions(self):
        initial_position_path = self.initial_dir / f"{self.layer}.pkl"
        self.initial_positions: LayerPositions = LayerPositions.load(
            initial_position_path
        )

    def _load_features(self):
        features = FeatureSaver.load_features(self.feature_path, keys=self.layer)[
            self.layer
        ]

        # if not already flattened, flatten now
        self.features = features.reshape((len(features), -1))

    def swap(self):
        if self.blocked:
            print(
                (
                    "Overwriting existing position file is blocked. Call "
                    ".unblock_overwrite() if you're really sure you want to continue"
                )
            )
            return

        self.new_positions, self.metrics = swap_optimize_neighborhoods(
            self.features,
            self.initial_positions.coordinates,
            num_steps=self.config.swapopt.num_steps,
            steps_per_neighborhood=self.config.swapopt.steps_per_neighborhood,
            neighborhood_width=self.initial_positions.neighborhood_width,
            return_metrics=True,
        )

        # need to recompute neighborhoods here!
        neighborhood_list = precompute_neighborhoods(
            self.new_positions,
            radius=self.initial_positions.neighborhood_width / 2,
            n_neighborhoods=20_000,
        )

        self.new_neighborhood_indices = collapse_and_trim_neighborhoods(
            neighborhood_list, keep_fraction=0.95, keep_limit=500, target_shape=None
        )

    def save_positions(self):
        if self.blocked:
            print(
                (
                    "Overwriting existing position file is blocked. Call "
                    ".unblock_overwrite() if you're really sure you want "
                    "to continue"
                )
            )
            return

        assert hasattr(self, "new_positions")

        new_pos = copy.deepcopy(self.initial_positions)
        new_pos.coordinates = self.new_positions
        new_pos.neighborhood_indices = self.new_neighborhood_indices
        new_pos.save(self.new_save_dir)
