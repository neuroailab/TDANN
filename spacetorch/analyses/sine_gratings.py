import functools
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .core import get_features_from_model

from spacetorch.datasets import sine_gratings
from spacetorch.types import AggMode
from spacetorch.utils.figure_utils import add_colorbar
from spacetorch.utils.generic_utils import load_pickle
from spacetorch.maps.smoother import Smoother, KernelParams
from spacetorch.maps.v1_map import V1Map
from spacetorch.paths import CACHE_DIR

METRIC_DICT = sine_gratings.SineGrating2019.get_metrics(as_dict=True)


def get_sine_responses(
    model_name,
    layers,
    step="latest",
    verbose: bool = True,
    exclude_fc: bool = True,
) -> sine_gratings.SineResponses:
    sine_features, _, sine_labels = get_features_from_model(
        model_name,
        "SineGrating2019",
        layers=layers,
        verbose=verbose,
        return_inputs_and_labels=True,
        step=step,
        exclude_fc=exclude_fc,
    )

    return sine_gratings.SineResponses(sine_features, sine_labels)


def get_smoothed_map(
    tissue: V1Map,
    metric: sine_gratings.Metric,
    nb_width: float = 1.5,
    final_width=None,
    final_stride=None,
    agg: Optional[AggMode] = None,
    verbose: bool = False,
) -> np.ndarray:
    def extractor(tissue_map: V1Map, metric: sine_gratings.Metric):
        return tissue_map.get_preferences(metric)

    final_width = final_width or nb_width / 1.5
    final_stride = final_stride or nb_width / 30
    kernel_params = KernelParams(width=final_width, stride=final_stride)
    smoother = Smoother(kernel_params, verbose=verbose)

    if agg is None:
        agg = metric.agg_mode

    smoothed = smoother(
        tissue, functools.partial(extractor, metric=metric), agg, high=metric.high
    )

    return smoothed


def get_sine_tissue(
    model_name,
    positions,
    layer: str = "layer2.0",
    step="latest",
    skip_cache=False,
    exclude_fc: bool = True,
):
    # check for cached responses
    cache_id = f"{model_name}_{step}"
    cache_loc = CACHE_DIR / "tuning_curve_fits" / cache_id / "responses.pkl"
    try:
        if cache_loc.exists() and not skip_cache:
            responses = load_pickle(cache_loc)
            print(f"Loaded {cache_id} from cache")
        else:
            responses = get_sine_responses(
                model_name,
                step=step,
                layers=[layer],
                verbose=True,
                exclude_fc=exclude_fc,
            )

        return V1Map(
            positions.coordinates,
            responses,
            cache_id=f"{model_name}_{step}" if not skip_cache else None,
        )
    except AssertionError as e:
        print(e)
        return None


def add_sine_colorbar(fig, ax, metric: sine_gratings.Metric, **kwargs):
    norm = mpl.colors.Normalize(vmin=0, vmax=metric.high)
    mappable = plt.cm.ScalarMappable(cmap=metric.colormap, norm=norm)
    mappable.set_array([])

    cbar = add_colorbar(fig, ax, mappable, ticks=[0, metric.high], **kwargs)
    cbar.ax.set_yticklabels([metric.xticklabels[0], metric.xticklabels[-1]])
    return cbar


def get_smoothness_curves(
    tissue: V1Map,
    metric: sine_gratings.Metric = METRIC_DICT["angles"],
    shuffle: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Returns:
        (midpoints, curves)
    """
    hcw = 3.5
    analysis_width = hcw * (4 / 3)

    # compute largest possible distance given the window size
    max_dist = np.sqrt(2 * analysis_width**2) / 2
    tissue.set_unit_mask_by_ptp_percentile("angles", 75)
    bin_edges = np.linspace(0, max_dist, 10)
    distances, curves = tissue.metric_difference_over_distance(
        metric=metric,
        distance_cutoff=max_dist,
        bin_edges=bin_edges,
        num_samples=20,
        sample_size=1000,
        shuffle=shuffle,
        verbose=False,
    )

    return distances, curves
