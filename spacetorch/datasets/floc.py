from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Type

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torchvision
from skimage import io
from torch.utils.data import Dataset
import xarray as xr

from spacetorch.utils.array_utils import flatten, tstat

# types
ComposedTransforms = Type[torchvision.transforms.transforms.Compose]

CATEGORIES: List[str] = [
    "number",
    "word",
    "limb",
    "body",
    "adult",
    "child",
    "car",
    "instrument",
    "house",
    "corridor",
    "scrambled",
]


@dataclass
class Contrast:
    name: str
    color: str
    on_categories: List[str]
    off_categories: Optional[List[str]] = None


DOMAIN_CONTRASTS: List[Contrast] = [
    Contrast(name="Objects", color="#33B0FF", on_categories=["car", "instrument"]),
    Contrast(name="Places", color="#20913E", on_categories=["house", "corridor"]),
    Contrast(name="Characters", color="#4B2021", on_categories=["word", "number"]),
    Contrast(name="Bodies", color="#E8B631", on_categories=["body", "limb"]),
    Contrast(name="Faces", color="#D43328", on_categories=["adult", "child"]),
]


class fLocData(Dataset):
    """VPNL fLoc Dataset"""

    def __init__(self, floc_dir: str, transforms: Optional[ComposedTransforms] = None):
        self.transforms = transforms

        file_dir = Path(floc_dir)
        self.file_list = sorted(file_dir.glob("*.jpg"))
        raw_categories = [f.stem.split("-")[0] for f in self.file_list]
        self.labels = [CATEGORIES.index(category) for category in raw_categories]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(self.file_list[idx])

        # expand third dim
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
        target = self.labels[idx]
        if self.transforms:
            img = self.transforms(img)

        return img, target


class fLocResponses:
    def __init__(self, features: np.ndarray, labels: np.ndarray):

        self.domains = {
            "characters": ["number", "word"],
            "bodies": ["limb", "body"],
            "faces": ["adult", "child"],
            "objects": ["car", "instrument"],
            "places": ["house", "corridor"],
        }

        self.categories = [
            category for domain in self.domains.values() for category in domain
        ]

        self._data = xr.DataArray(
            data=flatten(features),
            coords={
                "categories": ("image_idx", [CATEGORIES[idx] for idx in labels]),
                "category_indices": ("image_idx", labels),
            },
            dims=["image_idx", "unit_idx"],
        )

        self._drop_scrambled()

    def _drop_scrambled(self):
        self._data = self._data.where(self._data.categories != "scrambled", drop=True)

    def selectivity(
        self,
        on_categories: List[str],
        off_categories: Optional[List[str]] = None,
        selectivity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = tstat,
    ) -> np.ndarray:
        """Computes the specified selectivity function for the given category
        Returns:
            Category-selectivity of each unit
        """
        if off_categories is None:
            off_categories = list(set(self.categories) - set(on_categories))

        # get indices of labels that match the "on" and "off" criteria
        on_indices = self._data.categories.isin(on_categories)
        off_indices = self._data.categories.isin(off_categories)

        return selectivity_fn(
            self._data[on_indices].values, self._data[off_indices].values
        )

    def selectivity_category_average(
        self,
        on_categories: List[str],
        off_categories: Optional[List[str]] = None,
        selectivity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = tstat,
    ) -> np.ndarray:
        """Computes the specified selectivity function for the given category, but
        averages within each category first

        Returns:
            Category-selectivity of each unit
        """
        if off_categories is None:
            off_categories = list(set(self.categories) - set(on_categories))

        on_responses = []
        for cat in on_categories:
            responses = self._data[self._data.categories == cat].mean("image_idx")
            on_responses.append(responses.values)

        off_responses = []
        for cat in off_categories:
            responses = self._data[self._data.categories == cat].mean("image_idx")
            off_responses.append(responses.values)

        stacked_on_responses = np.stack(on_responses)
        stacked_off_responses = np.stack(off_responses)

        return selectivity_fn(stacked_on_responses, stacked_off_responses)

    def plot_rsm(
        self,
        ax,
        average_by_category: bool = False,
        cmap: str = "gist_heat",
        vmin: float = -0.25,
        vmax: float = 1.0,
        add_colorbar: bool = True,
        add_ticks: bool = True,
    ):
        if average_by_category:
            rsm = np.corrcoef(self.sorted_by_category.groupby("categories").mean())
        else:
            rsm = np.corrcoef(self.sorted_by_category)
        np.fill_diagonal(rsm, np.nan)
        img = ax.imshow(rsm, cmap=cmap, vmin=vmin, vmax=vmax)

        if add_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.figure.colorbar(img, cax=cax, orientation="vertical")

        if add_ticks:
            category_labels = CATEGORIES[:-1]
            images_per_category = len(rsm) // len(category_labels)

            ticks = np.arange(
                images_per_category // 2,
                stop=len(rsm) + images_per_category // 2,
                step=images_per_category,
            )
            ax.set_xticks(ticks)
            ax.set_xticklabels(category_labels, rotation=30)
            ax.set_yticks(ticks)
            ax.set_yticklabels(category_labels)

    @property
    def sorted_by_category(self) -> xr.DataArray:
        return self._data.sortby("category_indices")

    def __len__(self) -> int:
        return self._data.sizes["unit_idx"]
