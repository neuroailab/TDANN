from dataclasses import dataclass
from typing import Union, Tuple, List

import numpy as np
import scipy.stats as stats


@dataclass
class FlatIndices:
    chan_flat: np.ndarray
    x_flat: np.ndarray
    y_flat: np.ndarray


def norm(x):
    """Normalize values in x to the range [0, 1]"""
    return (x - np.min(x)) / np.ptp(x)


def midpoints_from_bin_edges(be: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Given `be`, a set of histogram bin edges, return the array of midpoints between
    those edges.
    """
    arr = np.array(be)
    width = arr[1] - arr[0]
    return arr[1:] - width / 2


def get_flat_indices(dims):
    """
    dims should be a CHW tuple
    """
    num_channels, num_x, num_y = dims

    # chan flat goes like [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...]
    chan_flat = np.repeat(np.arange(num_channels), num_x * num_y)

    # x flat goes like [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, ...]
    x_flat = np.repeat(np.tile(np.arange(num_x), num_channels), num_y)

    # y flat goes like [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, ...]
    y_flat = np.tile(np.arange(num_y), num_x * num_channels)

    return FlatIndices(chan_flat, x_flat, y_flat)


def lower_tri(x, keep_diagonal: bool = False):
    """return lower triangle of x, excluding diagonal"""
    assert len(x.shape) == 2
    return x[np.tril_indices_from(x, k=-1 if not keep_diagonal else 0)]


def sem(x, axis=None):
    """
    Return standard error of the mean for the given array
    over the specified axis. If axis is None, std is taken over

    """
    num_elements = x.shape[axis] if axis else len(x)
    return np.nanstd(x, axis=axis) / np.sqrt(num_elements)


def dprime(features_on, features_off):
    """
    Compute d-prime for two matrices
    Inputs
        features_on - n_samples x n_features to compute selectivity for
        features_off - n_samples x n_features to compute selectivity against
    """

    m_on = np.nanmean(features_on, axis=0)
    m_off = np.nanmean(features_off, axis=0)
    s_on = np.nanstd(features_on, axis=0)
    s_off = np.nanstd(features_off, axis=0)

    denom = np.sqrt((s_on**2 + s_off**2) / 2)

    # if variance is 0, set d-prime to 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, 0.0, (m_on - m_off) / denom)


def tstat(features_on, features_off):
    """
    Compute the t-statistic for two matrices
    Inputs
        features_on - n_samples x n_features to compute selectivity for
        features_off - n_samples x n_features to compute selectivity against
    """

    m_on = np.nanmean(features_on, axis=0)
    m_off = np.nanmean(features_off, axis=0)
    s_on = np.nanstd(features_on, axis=0)
    s_off = np.nanstd(features_off, axis=0)
    n_on = len(features_on)
    n_off = len(features_off)

    denom = np.sqrt((s_on**2 / n_on) + (s_off**2 / n_off))

    # if variance is 0, set t to 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, 0.0, (m_on - m_off) / denom)


def flatten(x: np.ndarray):
    """
    Flatten matrix along all dims but first
    """
    return x.reshape((len(x), -1))


def gaussian_2d(
    positions: np.ndarray, center: Tuple[float, float], sigma: float
) -> np.ndarray:
    """
    Inputs:
        positions: N x 2
        center: [center_x, center_y]
        sigma: spread of gaussian
    """
    sigma_sq = sigma**2
    return (
        1.0
        / (2.0 * np.pi * sigma_sq)
        * np.exp(
            -(
                (positions[:, 0] - center[0]) ** 2.0 / (2.0 * sigma_sq)
                + (positions[:, 1] - center[1]) ** 2.0 / (2.0 * sigma_sq)
            )
        )
    )


def chisq(x, y):
    """Distance between two sets of proportions: symmetrized chi-squared distance"""
    left = stats.chisquare(x, y).statistic
    right = stats.chisquare(y, x).statistic
    return np.mean([left, right])


def chisq_sim(x, y):
    """Similarity between two sets of proportions: the negative log
    symmetrized chi-squared distance
    """
    return -1 * np.log(chisq(x, y))
