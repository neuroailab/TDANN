import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr

from spacetorch.utils import array_utils, spatial_utils


# Core loss functions
def standard_scl(correlations, distances, dist_scaling=None):
    dist_scaling = dist_scaling or choose_dist_scaling(correlations, distances)
    recip_dist = 1.0 / (dist_scaling * distances + 1)
    return np.mean(np.abs(correlations - recip_dist))


def pearson_scl(correlations, distances, **kwargs):
    """
    Computes (1 - r) / 2, which is in the range[0, 1]
    """
    recip_dist = 1.0 / (distances + 1)
    return (1 - pearsonr(correlations, recip_dist)[0]) / 2


# Wrappers to deal with indexing
def spatial_loss_wrapper(
    loss_fn, features, positions, indices_to_include=None, **loss_fn_kwargs
):
    """
    CPU implementation of SCL.

    Inputs
        features: n_images x n_units
        positions: N x 2  matrix of position values
        indices_to_include: which of the N units to use. If None, uses all units
    """

    # if indices to include not provided, use all of them
    if indices_to_include is None:
        indices_to_include = np.arange(positions.shape[0])

    if len(indices_to_include) < 3:
        return np.nan

    # sub-select units
    features_sub = features[:, indices_to_include]
    positions_sub = positions[indices_to_include]

    correlations = array_utils.lower_tri(np.corrcoef(features_sub.T))
    distances = array_utils.lower_tri(squareform(pdist(positions_sub)))

    return loss_fn(correlations, distances, **loss_fn_kwargs)


def swapopt_spatial_loss_wrapper(
    loss_fn, correlations, positions, swap_indices, **loss_fn_kwargs
):
    """
    Similar to non-swapopt versions, but takes correlations directly
    """
    # compute only the rows of the distance matrix we need
    distances = cdist(positions[swap_indices], positions)
    correlations = correlations[swap_indices, :]

    return loss_fn(correlations.ravel(), distances.ravel(), **loss_fn_kwargs)


def neighborhood_loss(
    loss_fn,
    features,
    positions,
    radius=None,
    agg_func=np.nanmean,
    neighborhoods=None,
    n_neighborhoods=25,
    **loss_fn_kwargs
):
    """
    Computes spatial loss for a number of neighborhoods, either provided or computed
    """

    # make sure we have neighborhoods to work with
    if neighborhoods is None:
        assert (
            radius is not None
        ), "if no neighborhoods are provided, you need to specify the nb radius"
        neighborhoods = spatial_utils.precompute_neighborhoods(
            positions, radius=radius, n_neighborhoods=n_neighborhoods
        )

    # compute loss for each neighborhoods
    neighborhood_losses = [
        spatial_loss_wrapper(
            loss_fn,
            features,
            positions,
            indices_to_include=neighborhood,
            **loss_fn_kwargs
        )
        for neighborhood in neighborhoods
    ]

    # either return all losses, or aggregate
    if agg_func is not None:
        return agg_func(neighborhood_losses)
    return neighborhood_losses


def choose_dist_scaling(correlations, distances):
    """
    Given a distribution of correlations and distances, picks a constant to scale
    distances by so that the median of each distribution is matched
    """
    median_absolute_correlations = np.median(np.abs(correlations).ravel())
    median_distances = np.median(distances.ravel())
    return ((1.0 / median_absolute_correlations) - 1) / median_distances


# Legacy aliases
def spatial_correlation_loss(*args, **kwargs):
    return swapopt_spatial_loss_wrapper(standard_scl, *args, **kwargs)
