---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
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
from pathlib import Path
from dataclasses import dataclass
from typing import List
```

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import viridis
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
```

```python
import spacetorch.analyses.core as core
from spacetorch.datasets import floc, DatasetRegistry
from spacetorch.analyses.floc import get_floc_tissue
from spacetorch.analyses.imagenet import create_imnet_tissue
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    array_utils,
)
from spacetorch.analyses.sine_gratings import (
    get_sine_tissue,
    METRIC_DICT,
    get_smoothed_map,
)
from spacetorch.models.positions import get_flat_indices
```

```python
figure_utils.set_text_sizes()
```

# Setup

```python
SEED = 100
rng = default_rng(seed=SEED)
```

# Panel A

```python
model_name = "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
pos_type = "simclr_swap"
position_dict = core.get_positions(pos_type)
```

```python
for k, v in position_dict.items():
    print(k, np.ptp(v.coordinates), v.neighborhood_width)
```

```python
# establish "perfectly retinotopic eccentricity" as the distance from the center of the visual field
xx, yy = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
distance_from_center = np.sqrt(xx**2 + yy**2)
perfect_ret = distance_from_center
```

```python
# load an example image
dataset = DatasetRegistry.get("ImageNet")
sample = dataset[4500][0]
sample = array_utils.norm(sample.detach().cpu().numpy())
sample = np.transpose(sample, (1, 2, 0))
```

```python
cmap = viridis
```

```python
cr_width = cr_height = 80
central_region = [200 - cr_height, 200, 200 - cr_width, 200]
```

```python
fig, ax = plt.subplots(figsize=(2, 2))
red_rect = Rectangle(
    (central_region[0], central_region[2]),
    width=cr_width,
    height=cr_height,
    alpha=0.6,
    facecolor=figure_utils.get_color("TDANN"),
    edgecolor="k",
    lw=1,
)
ax.imshow(sample)
ax.add_patch(red_rect)
ax.axis("off")
figure_utils.save(fig, "F01/xx_bird", ext="png", dpi=600)
```

```python
rf_size_dir = Path("/share/kalanit/users/eshedm/spacetorch/paper_figures/RFs")
rf_layer_names = [
    "base_model_layer1_0",
    "base_model_layer2_0",
    "base_model_layer3_0",
    "base_model_layer4_0",
]
```

```python
@dataclass
class RF:
    # spatial tap x
    x: int

    # spatial tap y
    y: int

    # receptive field as first row, last row, first col, last col
    rf: List[int]

    def compute_overlap(self, region: List[int]):
        """computes overlap in [0, 1] between rf and a specified image region

        Core idea: we just need to find coordinates of the overlapping region
        """

        overlap_r0 = max(region[0], self.rf[0])
        overlap_r1 = min(region[1], self.rf[1])
        overlap_c0 = max(region[2], self.rf[2])
        overlap_c1 = min(region[3], self.rf[3])

        rowdiff = max(0, overlap_r1 - overlap_r0)
        coldiff = max(0, overlap_c1 - overlap_c0)
        overlap = rowdiff * coldiff

        max_possible = (region[1] - region[0]) * (region[3] - region[2])
        return overlap / max_possible
```

```python
def load_rf_data(rf_file):
    with rf_file.open("r") as stream:
        lines = stream.readlines()

    rfs = []
    for line in lines:
        x, y, a, b, c, d = line.strip().split(",")
        rf = RF(x=int(x), y=int(y), rf=[int(a), int(b), int(c), int(d)])
        rfs.append(rf)

    return rfs
```

```python
cmap = LinearSegmentedColormap.from_list(
    "custom", ["#ddd", figure_utils.get_color("TDANN")]
)
```

```python
vmax_lookup = {
    "layer1.0": 0.15,
    "layer2.0": 0.75,
    "layer3.0": 1.1,
    "layer4.0": 1.1,
}
```

```python
for layer, positions in position_dict.items():
    rf_file = rf_size_dir / f"base_model_{layer.replace('.','_')}.csv"
    if not rf_file.is_file():
        continue
    rfd = load_rf_data(rf_file)

    fig, ax = plt.subplots(figsize=(2, 2))

    # to avoid egregious overplotting, choose a random subset of 8000 units to plot
    mask = rng.choice(
        np.arange(len(positions.coordinates)), size=(20_000,), replace=False
    )

    x_flat = get_flat_indices(positions.dims).x_flat[mask]
    y_flat = get_flat_indices(positions.dims).y_flat[mask]

    if layer == "layer2.0" or layer == "layer1.0":
        passing = (
            (x_flat < x_flat.max())
            & (y_flat < y_flat.max())
            & (x_flat > x_flat.min())
            & (y_flat > y_flat.min())
        )
    else:
        passing = np.arange(len(mask))
    mask = mask[passing]

    overlaps = []
    for xf, yf in tqdm(zip(x_flat[passing], y_flat[passing]), total=len(mask)):
        matching = [rf for rf in rfd if (rf.x == xf) and (rf.y == yf)][0]
        overlap = matching.compute_overlap(central_region)
        overlaps.append(overlap)
    overlaps = np.array(overlaps)

    sort_ind = np.arange(len(overlaps))
    # plot units with very faint edges
    ax.scatter(
        *positions.coordinates[mask][sort_ind].T,
        c=overlaps[sort_ind],
        s=4,
        vmin=0,
        vmax=vmax_lookup[layer],
        edgecolors=(0, 0, 0, 0.5),
        linewidths=0.11,
        alpha=0.8,
        cmap=cmap,
        rasterized=True,
    )
    ax.axis("off")

    # add a scale bar a little below where we normally would
    ax.invert_yaxis()

    figure_utils.save(fig, f"F01/positions/purple_{layer}", ext="png", dpi=600)
```

# New Panel C

```python
V1_LAYER, VTC_LAYER = "layer2.0", "layer4.1"
positions = core.get_positions("simclr_swap")[V1_LAYER]
```

```python
v1_tissue = get_sine_tissue(
    "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_4",
    positions,
    layer=V1_LAYER,
)
```

```python
lims = [5, 95]
v1_tissue.set_mask_by_pct_limits([lims, lims])
smoothed = get_smoothed_map(
    v1_tissue, METRIC_DICT["angles"], final_width=1.5, final_stride=0.15, verbose=True
)

zoom_lims = [[30, 50], [30, 50]]
v1_tissue.set_mask_by_pct_limits(zoom_lims)
```

```python
fig, axes = plt.subplots(figsize=(3, 1.5), ncols=2)
cax = fig.add_axes([0.92, 0.16, 0.015, 0.65])
mappable = axes[0].imshow(
    smoothed, cmap=METRIC_DICT["angles"].colormap, interpolation="nearest"
)


rect = Rectangle((55, 105), width=55, height=55, fill=False, edgecolor="k", lw=1)
axes[0].add_patch(rect)


# add raw
v1_tissue.make_parameter_map(
    axes[1], final_psm=0.4, linewidths=0.2, edgecolor=(0, 0, 0, 0.4), rasterized=True
)

for ax in axes:
    ax.axis("off")

total_px = smoothed.shape[0]
total_mm = np.ptp(v1_tissue._positions) * 0.9
px_per_mm = total_px / total_mm
plot_utils.add_scale_bar(axes[0], 10 * px_per_mm, flipud=True)

plot_utils.add_scale_bar(axes[1], 1)
cb = plt.colorbar(mappable=mappable, cax=cax, ticks=[0, 90, 180])

# not sure why setting ticks above doesn't directly work, but this does
cb.set_ticks([0, 90, 180])
figure_utils.save(fig, "F01/bmarks/v1")
```

```python
# VTC stuff
positions = core.get_positions("simclr_swap")[VTC_LAYER]
```

```python
contrasts = floc.DOMAIN_CONTRASTS
contrast_dict = {c.name: c for c in contrasts}
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
ordered_contrasts = [contrast_dict[curr] for curr in contrast_order]
contrast_colors = {c.name: c.color for c in contrasts}
```

```python
vtc_tissue = get_floc_tissue(
    "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
    positions,
    layer=VTC_LAYER,
)
```

```python
vtc_tissue.patches = []
for contrast in contrasts:
    vtc_tissue.find_patches(contrast)
```

```python
rect0, rect1 = 35, 56
w = h = rect1 - rect0

fig, axes = plt.subplots(figsize=(3, 1.5), ncols=2)
for ax in axes:
    ax.axis("off")

axes[0].add_patch(Rectangle((0, 0), width=70, height=70, facecolor="#eee"))
axes[0].add_patch(
    Rectangle(
        (rect0, rect0), width=w, height=h, fill=False, edgecolor="k", lw=1, zorder=9
    ),
)
axes[0].set_xlim([0, 70])
axes[0].set_ylim([0, 70])
vtc_tissue.set_mask_by_limits([[rect0, rect1], [rect0, rect1]])
vtc_tissue.make_selectivity_map(
    axes[1],
    marker=".",
    contrasts=contrasts,
    scale_points=False,
    foreground_alpha=0.8,
    selectivity_threshold=5,
    rasterized=True,
    edgecolor=(0, 0, 0, 0.4),
    final_s=25,
    linewidths=0.2,
)

for patch in vtc_tissue.patches:
    axes[0].add_patch(patch.to_mpl_poly(alpha=0.8, lw=1))

plot_utils.add_scale_bar(axes[0], 10, y_start=-4)
plot_utils.add_scale_bar(axes[1], 10, y_start=rect0 - 2)

figure_utils.save(fig, "F01/bmarks/vtc")
```

## Panel B

```python
imagenet_features = core.get_features_from_model(
    model_name,
    "ImageNet",
    verbose=True,
    max_batches=32,
    batch_size=32,
    layers=["layer4.1"],
)

imnet_tissue = create_imnet_tissue(
    imagenet_features, position_dict["layer4.1"].coordinates
)
```

```python
random_sample = rng.choice(np.arange(len(imnet_tissue.positions)), size=(1000,))
```

```python
sample_feat = imnet_tissue.features[:, random_sample]
sample_pos = imnet_tissue.positions[random_sample, :]
# for schematic, scale down:
sample_pos = sample_pos / 7
print(sample_feat.shape, sample_pos.shape)
```

```python
corr = array_utils.lower_tri(np.corrcoef(sample_feat.T))
dist = array_utils.lower_tri(squareform(pdist(sample_pos)))

mask = np.where(dist <= 4)[0]

corr = array_utils.lower_tri(np.corrcoef(sample_feat.T))[mask]
dist = array_utils.lower_tri(squareform(pdist(sample_pos)))[mask]
```

```python
fig, ax = plt.subplots(figsize=(1.5, 1.5))
ax.scatter(dist, corr, s=0.1, c="k", linewidths=0, alpha=0.5, rasterized=True)
plot_utils.remove_spines(ax)
# ax.set_xticks([0.2, 1])
ax.set_yticks([-0.4, 0.8])
ax.set_ylim([-0.5, 0.9])
ax.set_xticklabels(["", ""])
ax.set_yticklabels(["", ""])
figure_utils.save(fig, "F01/panel_b_corrscat")
```

```python

```
