from dataclasses import dataclass
from typing import Optional, Type, List, Callable, Union
import glob
import numpy as np
import torch
import torchvision
from skimage import io
from torch.utils.data import Dataset
import xarray as xr

from spacetorch.colormaps import nauhaus_colormaps
from spacetorch.constants import DVA_PER_IMAGE
from spacetorch.datasets.ringach_2002 import load_ringach_data
from spacetorch.types import AggMode
from spacetorch.utils.array_utils import flatten

# types
ComposedTransforms = Type[torchvision.transforms.transforms.Compose]


@dataclass
class Metric:
    name: str
    n_unique: int
    high: float
    xticklabels: Union[List[str], np.ndarray]
    xlabel: str
    agg_mode: AggMode
    colormap: Callable


class SineGrating2019(Dataset):
    """Full-field sine gratings dataset"""

    metrics: List[Metric] = [
        Metric(
            name="angles",
            n_unique=8,
            high=180,
            xticklabels=[f"{x:.0f}" for x in np.linspace(0, 180, 9)],
            xlabel=r"Orientation ($^\circ$)",
            agg_mode="circmean",
            colormap=nauhaus_colormaps["angles"],
        ),
        Metric(
            name="sfs",
            n_unique=8,
            high=110,
            xticklabels=[
                f"{x:.0f}" for x in np.linspace(5.5, 110.0, 9) / DVA_PER_IMAGE
            ],
            xlabel="Spatial Frequency (cpd)",
            agg_mode="mean",
            colormap=nauhaus_colormaps["sfs"],
        ),
        Metric(
            name="colors",
            n_unique=2,
            high=1,
            xticklabels=["B/W", "Color"],
            xlabel="",
            agg_mode="mean",
            colormap=nauhaus_colormaps["colors"],
        ),
    ]

    def __init__(self, sine_dir: str, transforms: Optional[ComposedTransforms] = None):
        self.transforms = transforms
        self.file_list = sorted(glob.glob(f"{sine_dir}/*.jpg"))

        self.labels = np.zeros([len(self.file_list), 4], dtype=float)
        for img_idx, fname in enumerate(self.file_list):
            parts = fname.split("/")[-1].split("_")

            angle = float(parts[1][:-3])
            sf = float(parts[2][:-2])
            phase = float(parts[3][:-5])
            color_string = parts[4].split(".jpg")[0]
            color = 0.0 if color_string == "bw" else 1.0

            self.labels[img_idx, 0] = angle
            self.labels[img_idx, 1] = sf
            self.labels[img_idx, 2] = phase
            self.labels[img_idx, 3] = color

    @classmethod
    def get_metrics(cls, as_dict: bool = False):
        if as_dict:
            return {
                "angles": cls.metrics[0],
                "sfs": cls.metrics[1],
                "colors": cls.metrics[2],
            }
        return cls.metrics

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(self.file_list[idx])
        target = self.labels[idx, :]
        if self.transforms:
            img = self.transforms(img)

        return img, target


class SineResponses:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        normalize_to_ringach_firing_rates: bool = True,
    ):
        self.DVA_PER_IMAGE = DVA_PER_IMAGE

        if normalize_to_ringach_firing_rates:
            # make the medians match, and multiply flat features by that scaling factor
            if np.min(features) < 0:
                features -= np.min(features)
            max_per_unit: np.ndarray = np.max(features, axis=0)
            median_of_peak_neuron_firing_rates = np.median(load_ringach_data("maxdc"))
            ringach_to_features_ratio: float = (
                median_of_peak_neuron_firing_rates / np.median(max_per_unit)
            )

            features *= ringach_to_features_ratio

        self._data = xr.DataArray(
            data=flatten(features),
            coords={
                "angles": ("image_idx", labels[:, 0]),
                "sfs": ("image_idx", labels[:, 1]),
                "phases": ("image_idx", labels[:, 2]),
                "colors": ("image_idx", labels[:, 3]),
            },
            dims=["image_idx", "unit_idx"],
        )

        self._convert_sfs_to_cpd()
        self._compute_circular_variance()

    def __len__(self) -> int:
        return self._data.sizes["unit_idx"]

    @property
    def orientation_tuning_curves(self) -> xr.DataArray:
        mean_response_to_each_orientation = self._data.groupby("angles").mean()
        return mean_response_to_each_orientation.T

    @property
    def circular_variance(self) -> np.ndarray:
        return self._data.circular_variance.values

    def get_preferences(self, metric: str = "angles") -> xr.DataArray:
        mean_response = self._data.groupby(metric).mean()
        return mean_response.argmax(axis=0)

    def get_peak_heights(self, metric: str = "angles"):
        tuning_curve = self._data.groupby(metric).mean()
        return np.ptp(tuning_curve.data, axis=0)

    def _convert_sfs_to_cpd(self):
        cpd: np.ndarray = self._data["sfs"] / self.DVA_PER_IMAGE
        self._data = self._data.assign_coords({"sfs": ("image_idx", cpd.data)})

    def _compute_circular_variance(self):
        n_angles = 8

        # the angles we use evenly span 0 to pi, but do not wrap
        angles = np.linspace(0, np.pi, n_angles + 1)[:-1]

        # compute "R"
        numerator = np.sum(
            self.orientation_tuning_curves * np.exp(angles * 2 * 1j), axis=1
        )
        denominator = np.sum(self.orientation_tuning_curves, axis=1)
        R = numerator / denominator

        # compute circular variance
        CV = 1 - np.abs(R)

        self._data = self._data.assign_coords(
            {"circular_variance": ("unit_idx", CV.data)}
        )
