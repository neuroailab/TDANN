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
import matplotlib.lines as lines
import numpy as np
from pathlib import Path
import scipy.io
from scipy.spatial.distance import cdist
from tqdm import tqdm
```

```python
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    seed_str,
)
import spacetorch.analyses.core as core
from spacetorch.analyses.sine_gratings import get_sine_tissue
from spacetorch.analyses.floc import get_floc_tissue
from spacetorch.datasets import floc

from spacetorch.maps import nsd_floc
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
def outside_exclusion_radius(existing, row, col, radius=1):
    stacked = np.stack(existing)
    dist = cdist(stacked, [[row, col]])
    return dist.min() <= radius
```

```python
V1_LAYER, VTC_LAYER = "layer2.0", "layer4.1"
positions = core.get_positions("simclr_swap")
random_positions = core.get_positions("retinotopic")
```

```python
model_tissue = get_floc_tissue(
    "relu_rescue/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
    positions[VTC_LAYER]
)
```

```python
data_path = "/oak/stanford/groups/kalanit/biac2/kgs/projects/spacenet/spacetorch/nsd_tvals/figure8_material/model_pos_and_tvals_subj04.mat"
data = scipy.io.loadmat(data_path)
faces = data["faces_tvals"]
places = data["places_tvals"]
model_pos = data["model_pos"]
corrs = data["corr"]
```

```python
midpoint = faces.shape[1] // 2
faces = nsd_floc.trim(faces[:, midpoint:])
places = nsd_floc.trim(places[:, midpoint:])
mapping = nsd_floc.trim(model_pos[:, midpoint:])
corrs = nsd_floc.trim(corrs[:, midpoint:])

flat_corrs = corrs.ravel()
valid = ~np.isnan(flat_corrs)
flat_corrs[~valid] = -2
sort_ind = np.argsort(flat_corrs)
```

```python
offset = 75
human = {"Faces": faces, "Places": places}
contrast_name = "Faces"
indices = [45174, 98311, 116898, 46395]
```

```python
np.ptp(model_tissue.positions)
```

```python
fig, axes = plt.subplots(figsize=(3, 3), nrows=2)
for ax, contrast_name in zip(axes, ["Faces", "Places"]):
    model_tissue.make_single_contrast_map(
        ax,
        contrast_dict[contrast_name],
        final_psm=0.0001,
        vmin=-20,
        vmax=20,
        rasterized=True,
    )

    mappable = ax.imshow(
        human[contrast_name],
        cmap="seismic",
        vmin=-20,
        vmax=20,
        extent=[offset, offset + 70, 0, 70],
        origin="lower",
    )
    ax.set_xlim([0, 150])

    ax.set_xticks([])
    ax.set_yticks([])
    plot_utils.remove_spines(ax, to_remove="all")

    seen = []
    for idx in indices:
        row, col = np.unravel_index(idx, mapping.shape)
        if len(seen) > 0:
            too_close = outside_exclusion_radius(seen, row, col, radius=5)
            if too_close:
                continue
        seen.append([row, col])
        x_factor = 70 / mapping.shape[1]
        y_factor = 70 / mapping.shape[0]
        unit = int(mapping[row, col])
        unit_x, unit_y = model_tissue.positions[unit]
        model_x = col * x_factor + offset
        model_y = row * y_factor

        ax.plot(
            [unit_x, model_x],
            [unit_y, model_y],
            c="k",
            lw=1,
            marker=".",
            markersize=6,
            markeredgecolor="k",
            markerfacecolor="w",
        )

axes[0].text(0.18, 1.1, "TDANN", transform=axes[0].transAxes)
axes[0].text(0.63, 1.1, "Human VTC", transform=axes[0].transAxes)

axes[0].set_ylabel("Faces")
axes[1].set_ylabel("Places")

cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.5])
cb = fig.colorbar(mappable, cax=cbar_ax, ticks=[-20, 0, 20])
cb.set_label(r"$t$-value", labelpad=0)
figure_utils.save(fig, "S18/sel_maps")
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
figure_utils.save(fig, "S08/sel_maps")
```

```python

```
