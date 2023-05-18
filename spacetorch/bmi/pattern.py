from dataclasses import dataclass
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


@dataclass
class Pattern:
    pos: np.ndarray
    feats: np.ndarray
    px_resolution: int = 2500

    def __post_init__(self) -> None:
        self.extent_mm = np.ptp(self.pos.ravel())
        self.px_per_mm = self.px_resolution / self.extent_mm
        self.mm_per_px = 1 / self.px_per_mm

    def plot(
        self, ax: plt.Axes, smudged: bool = False, mm_to_show: float = 2, **kwargs
    ) -> AxesImage:
        im = self.smoothed if smudged else self.unsmoothed

        # convert mm to show to px to show
        px_to_show = math.floor(mm_to_show * self.px_per_mm)

        # compute margin on either side
        margin = (im.shape[0] - px_to_show) // 2

        to_plot = im[margin:-margin, margin:-margin]

        # normalize to [0, 1]
        to_plot = (to_plot - np.min(to_plot)) / np.ptp(to_plot)
        return ax.imshow(to_plot, **kwargs)

    def smudge(self, sigma_mm: float) -> None:
        anchors = np.linspace(self.pos.min(), self.pos.max(), self.px_resolution)
        grid_x, grid_y = np.meshgrid(anchors, anchors)
        interp = griddata(
            points=self.pos, values=self.feats, xi=(grid_x, grid_y), method="nearest"
        )

        # the sigma we smooth with is in units of pixels
        sigma_px = sigma_mm * self.px_per_mm

        # if sigma was sent in as 0.0, use original data
        if sigma_px > 0:
            smoothed = gaussian_filter(interp, sigma=sigma_px)
        else:
            smoothed = interp.copy()

        self.smoothed = smoothed.copy()
        self.unsmoothed = interp.copy()
        smoothed = smoothed.T

        # map smoothed activity back to neurons
        norm_pos = self.pos / self.pos.max()

        # rescale up to resolution
        norm_pos = norm_pos * (smoothed.shape[0] - 1)

        # cast to int
        norm_pos = norm_pos.astype(int)

        # pull nearest values
        mapped = []
        for coord in norm_pos:
            val = smoothed[coord[0], coord[1]]
            if np.isnan(val):
                val = 0
            mapped.append(val)

        self.smudged_feats = np.array(mapped)
