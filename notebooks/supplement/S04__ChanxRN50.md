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
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
import numpy as np
```

```python
from spacetorch.utils import (
    figure_utils,
    plot_utils,
)
import spacetorch.analyses.core as core
from spacetorch.analyses.sine_gratings import get_sine_tissue
from spacetorch.maps.pinwheel_detector import PinwheelDetector
```

```python
figure_utils.set_text_sizes()
```

# load RN50 and ChanX models

```python
V1_LAYER = "layer2.0"
positions = core.get_positions("simclr_swap")[V1_LAYER]
resnet50_positions = core.get_positions("resnet50", rescale=False)[V1_LAYER]
chanx2_positions = core.get_positions("chanx2", rescale=False)[V1_LAYER]
```

```python
tissues = {
    "ResNet18": get_sine_tissue(
        "nonspatial/simclr_lw0/simclr_spatial_resnet18_swappedon_SineGrating2019_lw0",
        positions,
        layer=V1_LAYER,
    ),
    "ResNet50": get_sine_tissue(
        "nonspatial/supervised_resnet50",
        resnet50_positions,
        layer=V1_LAYER,
        exclude_fc=True,
    ),
    "ResNet18: 2x Channels": get_sine_tissue(
        "nonspatial/supervised_resnet18_chanx2",
        chanx2_positions,
        layer=V1_LAYER,
        exclude_fc=False,
    ),
}
```

```python
sm_lookup = {"ResNet18": 1.5, "ResNet50": 0.54, "ResNet18: 2x Channels": 0.4}

var_lookup = {"ResNet18": 0.3, "ResNet50": 0.3, "ResNet18: 2x Channels": 0.95}

width_lookup = {
    "ResNet18": positions.neighborhood_width,
    "ResNet50": resnet50_positions.neighborhood_width,
    "ResNet18: 2x Channels": chanx2_positions.neighborhood_width,
}
```

```python
def plot_neuron(ax, soma_location, size=1, fill_color="#222", alpha=1):
    x, y = soma_location
    top = [x, y + size]
    right = [x + size, y - size]
    left = [x - size, y - size]

    soma = Polygon(
        [top, right, left],
        fill=fill_color is not None,
        facecolor=fill_color,
        alpha=alpha,
        edgecolor="k",
        transform=ax.transAxes,
        lw=1.5,
    )

    top_line = Line2D([x, x], [y + size, y + 3 * size], color="k", alpha=alpha)
    right_line = Line2D(
        [x + size, x + size * 1.2], [y - size, y - 1.2 * size], color="k", alpha=alpha
    )
    left_line = Line2D(
        [x - size, x - size * 1.2], [y - size, y - 1.2 * size], color="k", alpha=alpha
    )

    ax.add_patch(soma)
    ax.add_artist(top_line)
    ax.add_artist(right_line)
    ax.add_artist(left_line)


def add_lateral_width_diagram(ax, lat_width):
    width = lat_width * 2.6
    height = 0.08
    pad = 0.05

    x = 1 - width - pad
    y = 1 - height - pad

    backdrop_rectangle = Rectangle(
        (x, y),
        width,
        height,
        transform=ax.transAxes,
        fill=True,
        facecolor="white",
        edgecolor="black",
    )

    ax.add_patch(backdrop_rectangle)

    plot_neuron(ax, (x + width / 2, y + height / 2), size=0.025)

    lat_width
    ax.plot(
        [x + width / 2 - lat_width, x + width / 2 + lat_width],
        [y + height / 2, y + height / 2],
        c="k",
        zorder=10,
        transform=ax.transAxes,
    )
```

```python
frac = 0.4
```

```python
fig, axes = plt.subplots(figsize=(6, 2), ncols=3)

for ax, (name, tissue) in zip(axes, tissues.items()):
    tissue.set_mask_by_pct_limits([[30, 70], [30, 70]])

    extent = np.ptp(tissue._positions)

    pindet = PinwheelDetector(tissue, size_mult=sm_lookup[name])
    pos, neg = pindet.count_pinwheels(var_thresh=var_lookup[name])
    total = pos + neg
    pindet.plot(ax)

    extent_mm = extent * frac
    square_area = extent_mm**2
    density = round(total / square_area, 2)

    density_str = f"{density} " r"$\frac{pw}{mm^2}$"
    ax.text(
        0.05,
        0.15,
        density_str,
        transform=ax.transAxes,
        bbox=dict(
            fill=True, facecolor="white", edgecolor="black", boxstyle="square,pad=.1"
        ),
    )
    ax.axis("off")

    total_px = pindet.smoothed.shape[0]
    px_per_mm = total_px / extent_mm
    plot_utils.add_scale_bar(ax, 2 * px_per_mm, flipud=True)
    lat_width_mm = width_lookup[name]
    frac_lat_width = lat_width_mm / extent_mm
    add_lateral_width_diagram(ax, frac_lat_width)
    ax.set_title(name)

figure_utils.save(fig, "S04/chanx")
```

```python

```

```python

```
