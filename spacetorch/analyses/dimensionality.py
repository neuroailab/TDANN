import numpy as np
from numpy.random import default_rng

from sklearn.decomposition import PCA
from spacetorch.constants import RNG_SEED

rng = default_rng(seed=RNG_SEED)


def effective_dim(eigvals: np.ndarray) -> float:
    squared_sum = np.sum(eigvals) ** 2
    sum_of_squares = np.sum(eigvals**2)
    effective_dim = squared_sum / sum_of_squares
    return effective_dim


def compute_eigvals(features: np.ndarray) -> np.ndarray:
    pca = PCA().fit(features)
    return pca.explained_variance_
