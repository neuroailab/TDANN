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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
```

```python
import spacetorch.analyses.core as core
from spacetorch.datasets.retinal_waves import (
    DEFAULT_RWAVE_DIRS,
    WavePool,
)
from spacetorch.utils import (
    figure_utils,
    plot_utils,
)
from spacetorch.analyses.sine_gratings import (
    get_sine_tissue,
    add_sine_colorbar,
    METRIC_DICT,
)
from spacetorch.models.positions import NetworkPositions
from spacetorch.maps.pinwheel_detector import PinwheelDetector
from spacetorch.paths import POSITION_DIR
```

```python
figure_utils.set_text_sizes()
```

# 1. Maps grid

```python
V1_LAYER = "layer2.0"
positions = core.get_positions("simclr_swap")[V1_LAYER]

nonspatial_imagenet_swapped = NetworkPositions.load_from_dir(
    (
        f"{POSITION_DIR}"
        "/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0"
        "/resnet18_retinotopic_init_fuzzy_swappedon_ImageNet"
    )
).layer_positions[V1_LAYER]
nonspatial_imagenet_swapped.coordinates = nonspatial_imagenet_swapped.coordinates * 1.5

retwave_positions = NetworkPositions.load_from_dir(
    (
        f"{POSITION_DIR}"
        "/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0"
        "/resnet18_retinotopic_init_fuzzy_swappedon_SquareRetinalWaveData"
    )
).layer_positions[V1_LAYER]
retwave_positions.coordinates = retwave_positions.coordinates * 1.5
```

```python
model = "nonspatial/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0"
tissues = {
    "Retinal Waves": get_sine_tissue(model, retwave_positions),
    "Sine Gratings": get_sine_tissue(model, positions),
    "ImageNet": get_sine_tissue(model, nonspatial_imagenet_swapped),
}
```

```python
angle_metric = METRIC_DICT["angles"]
```

```python
ncols = len(tissues)
fig, ax_rows = plt.subplots(ncols=3, nrows=3, figsize=(5, 5))

# plot models
for axes, (name, tissue) in zip(ax_rows.T, tissues.items()):
    tissue.set_mask_by_pct_limits([[30, 60], [30, 60]])

    for (metric_name, metric), ax in zip(METRIC_DICT.items(), axes):
        scatter_handle = tissue.make_parameter_map(
            ax, metric=metric, scale_points=True, final_psm=0.1
        )
        ax.axis("off")

        # add a colorbar if we're in the last column
        if axes[0] == ax_rows[0, -1]:
            cbar = add_sine_colorbar(fig, ax, metric, label=metric.xlabel)
            cbar.ax.set_yticklabels([metric.xticklabels[0], metric.xticklabels[-1]])

    plot_utils.add_scale_bar(axes[-1], width=2)
    plt.subplots_adjust(hspace=0.05, wspace=0.01)
    axes[0].set_title(name)

for ax in ax_rows[:, 0]:
    ax.axis("off")

# labels
ax_rows[0][0].set_ylabel("Orientation")
ax_rows[1][0].set_ylabel("Spatial Frequency")
ax_rows[2][0].set_ylabel("Chromaticity")
```

```python
ncols = len(tissues)
fig, axes = plt.subplots(ncols=3, figsize=(6, 2))
cbar_ax = fig.add_axes([0.91, 0.18, 0.02, 0.65])

# plot models
for ax, (name, tissue) in zip(axes, tissues.items()):
    tissue.set_mask_by_pct_limits([[40, 80], [40, 80]])
    pindet = PinwheelDetector(tissue)
    mappable = pindet.plot(ax)
    smoothed = pindet.smoothed
    # smoothed = get_smoothed_map(tissue, angle_metric, verbose=True)
    mappable = ax.imshow(smoothed, cmap=angle_metric.colormap, interpolation="nearest")
    ax.set_title(name)
    ax.axis("off")
    total_mm = 0.4 * np.ptp(tissue._positions)
    total_px = smoothed.shape[0]
    px_per_mm = total_px / total_mm
    plot_utils.add_scale_bar(ax, 2 * px_per_mm, flipud=True)

cb = plt.colorbar(
    mappable,
    cax=cbar_ax,
    extend="both",
    extendrect=True,
    label="Preferred Orientation",
    ticks=[0, 90, 180],
)
cb.set_ticks([0, 90, 180])

figure_utils.save(fig, "S12/ori_maps")
```

```python
wave_pool = WavePool(DEFAULT_RWAVE_DIRS, max_waves_per_dir=2)
```

```python
for _ in range(3):
    wave = next(wave_pool)
```

```python
images = np.stack([Image.open(path).convert("RGB") for path in wave])
subset = images[10:-10:15]
```

```python
fig, axes = plt.subplots(nrows=5, figsize=(2 / 5, 2))
plot_utils.remove_spines(axes.ravel(), to_remove=["top", "bottom", "left", "right"])
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

exim = subset[3:]
for idx, (im, ax) in enumerate(zip(exim, axes)):
    ax.imshow(im)

    title = r"$t_{tmp}$".replace("tmp", str(idx))
    ax.text(-0.5, 0.5, title, transform=ax.transAxes)

figure_utils.save(fig, "S12/wave_ex")
```

```python

```
