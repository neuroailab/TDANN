from dataclasses import dataclass

import numpy as np
import scipy.special as sc
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    fit: np.ndarray
    mse: float


def fit_vonmises(vec: np.ndarray, p0=None, max_iter: int = 100000) -> FitResult:
    """
    Fits a 1D Von Mises function (circular Gaussian) to vector vec

    Args:
        vec: vector to be fit
        p0: initial parameter vector
        max_iter: maximum number of optimization iterations
    """

    def func(x, a, x0, kappa):
        return a * np.exp(kappa * np.cos(x - x0)) / (2 * np.pi * sc.i0(kappa))

    x = np.linspace(-np.pi, np.pi, vec.shape[0])
    params, cov = curve_fit(func, x, vec, p0=p0, maxfev=int(max_iter))
    fit = func(x, params[0], params[1], params[2])
    mse = np.mean(np.power(fit - vec, 2))
    return FitResult(fit=fit, mse=mse)
