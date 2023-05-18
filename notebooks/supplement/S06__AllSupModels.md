---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import pprint
```

```python
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from tqdm import tqdm
```

```python
import spacetorch.analyses.core as core
from spacetorch.utils import (
    figure_utils,
    seed_str,
)
from spacetorch.analyses.sine_gratings import get_sine_tissue
from spacetorch.analyses.floc import get_floc_tissue
from spacetorch.datasets import floc

from spacetorch.maps.v1_map import metric_dict
from spacetorch.maps.pinwheel_detector import PinwheelDetector
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}
pprint.pprint(contrasts)
```

```python
V1_LAYER, VTC_LAYER = "layer2.0", "layer4.1"
positions = core.get_positions("supervised_swap")
random_positions = core.get_positions("retinotopic")
```

```python
seeds = range(5)
tissues = {
    "V1": {
        "SW_0": {
            seed: get_sine_tissue(
                f"nonspatial/supervised_lw0/supervised_spatial_resnet18_swappedon_SineGrating2019_lw0{seed_str(seed)}",
                positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in seeds
        },
        "SW_0pt1": {
            seed: get_sine_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lw01{seed_str(seed)}",
                positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in seeds
        },
        "SW_0pt25": {
            seed: get_sine_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019{seed_str(seed)}",
                positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in seeds
        },
        "SW_0pt5": {
            seed: get_sine_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lwx2{seed_str(seed)}",
                positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in seeds
        },
        "SW_1pt25": {
            seed: get_sine_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lwx5{seed_str(seed)}",
                positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in seeds
        },
        "SW_2pt5": {
            seed: get_sine_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lwx10{seed_str(seed)}",
                positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in seeds
        },
        "SW_25": {
            seed: get_sine_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lwx100{seed_str(seed)}",
                positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in seeds
        },
    },
    "VTC": {
        "SW_0": {
            seed: get_floc_tissue(
                f"nonspatial/supervised_lw0/supervised_spatial_resnet18_swappedon_SineGrating2019_lw0{seed_str(seed)}",
                positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in seeds
        },
        "SW_0pt1": {
            seed: get_floc_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lw01{seed_str(seed)}",
                positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in seeds
        },
        "SW_0pt25": {
            seed: get_floc_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019{seed_str(seed)}",
                positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in seeds
        },
        "SW_0pt5": {
            seed: get_floc_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lwx2{seed_str(seed)}",
                positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in seeds
        },
        "SW_1pt25": {
            seed: get_floc_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lwx5{seed_str(seed)}",
                positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in seeds
        },
        "SW_2pt5": {
            seed: get_floc_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lwx10{seed_str(seed)}",
                positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in seeds
        },
        "SW_25": {
            seed: get_floc_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019_lwx100{seed_str(seed)}",
                positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in seeds
        },
    },
}
```

# All Map Figure

```python
fig = plt.figure(figsize=(3.5, 2.5))
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(5, 7),
    axes_pad=0.01,
)

handle = None
lims = [30, 60]
for col_idx, (name, seeds) in enumerate(tissues["V1"].items()):
    for row_idx, (seed, tissue) in enumerate(seeds.items()):
        ax_idx = np.ravel_multi_index((row_idx, col_idx), (5, 7))
        ax = grid[ax_idx]
        tissue.set_mask_by_pct_limits([lims, lims])
        pindet = PinwheelDetector(tissue)
        pindet.count_pinwheels()

        h = ax.imshow(
            pindet.smoothed, metric_dict["angles"].colormap, interpolation="nearest"
        )
        pos_centers, neg_centers = pindet.centers
        for xloc, yloc in pos_centers:
            ax.scatter(xloc, yloc, c="k", s=0.4)

        for xloc, yloc in neg_centers:
            ax.scatter(xloc, yloc, c="w", s=0.4)

        if name == "SW_0pt25":
            handle = h

        if row_idx == 0:
            if name == "SW_0pt25":
                ax.set_title(r"$\alpha = 0.25$", fontsize=5)
            elif name == "SW_0":
                ax.set_title(r"$\alpha = 0$", fontsize=5)
            else:
                ax.set_title(figure_utils.get_label(name), fontsize=5)

        if col_idx == 0:
            ax.text(-0.25, 0.2, f"Seed {seed}", rotation=90, transform=ax.transAxes)

        ax.axis("off")

figure_utils.save(fig, "S06/orientation_maps")
```
```python
fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(3.5, 2.5))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

for col_idx, (name, seeds) in enumerate(tissues["VTC"].items()):
    for row_idx, (seed, tissue) in enumerate(tqdm(seeds.items())):
        ax = axes[row_idx, col_idx]
        tissue.make_selectivity_map(
            ax,
            selectivity_threshold=12,
            foreground_alpha=0.8,
            size_mult=2e-3,
            marker=".",
            linewidths=0,
            rasterized=True,
        )
        ax.axis("off")

        if row_idx == 0:
            if name == "SW_0pt25":
                ax.set_title(r"$\alpha = 0.25$", fontsize=5)
            elif name == "SW_0":
                ax.set_title(r"$\alpha = 0$", fontsize=5)
            else:
                ax.set_title(figure_utils.get_label(name), fontsize=5)

        if col_idx == 0:
            ax.text(-0.25, 0.2, f"Seed {seed}", rotation=90, transform=ax.transAxes)

figure_utils.save(fig, "S06/sel_maps")
```

```python

```
