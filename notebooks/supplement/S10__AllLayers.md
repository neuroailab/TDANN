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
```

```python
import spacetorch.analyses.core as core
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    generic_utils,
)
from spacetorch.analyses.sine_gratings import (
    METRIC_DICT,
)
from spacetorch.datasets import sine_gratings, floc
from spacetorch.maps.v1_map import V1Map
from spacetorch.maps.it_map import ITMap

from spacetorch.paths import CACHE_DIR
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}
```

```python
positions = core.get_positions("simclr_swap")
random_positions = core.get_positions("retinotopic")
```

```python
layers = [
    "layer1.0",
    "layer1.1",
    "layer2.0",
    "layer2.1",
    "layer3.0",
    "layer3.1",
    "layer4.0",
    "layer4.1",
]
```

```python
def get_tissues(
    model_name, dataset_name, response_constructor, tissue_constructor, positions
):
    feature_dict, _, labels = core.get_features_from_model(
        model_name,
        dataset_name,
        layers=layers,
        verbose=True,
        return_inputs_and_labels=True,
        step="latest",
    )

    response_dict = {
        layer.split("base_model.")[-1]: response_constructor(features, labels)
        for layer, features in feature_dict.items()
    }

    tissues = {
        layer: tissue_constructor(
            positions[layer].coordinates, responses, cache_id=None
        )
        for layer, responses in response_dict.items()
    }

    return tissues
```

```python
cache_loc = CACHE_DIR / "all_layers_tissues_rescale.pkl"
```

```python tags=[]
if cache_loc.is_file():
    tissues = generic_utils.load_pickle(cache_loc)
    if "IT" in tissues.keys():
        tissues["VTC"] = tissues["IT"]
else:
    tissues = {
        "V1": get_tissues(
            "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
            "SineGrating2019",
            sine_gratings.SineResponses,
            V1Map,
            positions,
        ),
        "VTC": get_tissues(
            "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
            "fLoc",
            floc.fLocResponses,
            ITMap,
            positions,
        ),
    }
    generic_utils.write_pickle(cache_loc, tissues)
```

```python
lims = {
    "layer1.0": [[1, 1.5], [1, 1.5]],
    "layer1.1": [[1, 1.5], [1, 1.5]],
    "layer2.0": [[10, 20], [10, 20]],
    "layer2.1": [[10, 20], [10, 20]],
    "layer3.0": [[10, 15], [10, 15]],
    "layer3.1": [[10, 15], [10, 15]],
    "layer4.0": [[14, 56], [14, 56]],
    "layer4.1": [[14, 56], [14, 56]],
}

sizes = {
    "layer1.0": 0.2,
    "layer1.1": 0.2,
    "layer2.0": 0.5,
    "layer2.1": 0.5,
    "layer3.0": 1.3,
    "layer3.1": 1,
    "layer4.0": 0.8,
    "layer4.1": 0.8,
}
```

```python
title_lookup = {
    "layer1.0": "Layer 2",
    "layer1.1": "Layer 3",
    "layer2.0": "Layer 4",
    "layer2.1": "Layer 5",
    "layer3.0": "Layer 6",
    "layer3.1": "Layer 7",
    "layer4.0": "Layer 8",
    "layer4.1": "Layer 9",
}
```

```python
for metric_name, metric in METRIC_DICT.items():
    fig, axes = plt.subplots(ncols=8, figsize=(6, 6 / 9))
    for (layer, tissue), ax in zip(tissues["V1"].items(), axes):
        tissue.reset_unit_mask()
        lim = lims[layer]
        tissue.set_mask_by_limits(lim)
        tissue.make_parameter_map(
            ax,
            scale_points=False,
            metric=metric,
            final_s=sizes[layer],
            rasterized=True,
            linewidths=0,
        )
        ax.axis("off")
        plot_utils.add_scale_bar(ax, 0.5)
        if metric_name == "angles":
            ax.set_title(title_lookup[layer])

    figure_utils.save(fig, f"S10/maps_{metric_name}")
```

## CircVar

```python
# VTC stuff
fig, axes = plt.subplots(ncols=8, figsize=(6, 6 / 9))
for (layer, tissue), ax in zip(tissues["VTC"].items(), axes):
    lim = lims[layer]
    tissue.set_mask_by_limits(lim)
    tissue.make_selectivity_map(
        ax,
        selectivity_threshold=12,
        final_s=sizes[layer] * 10,
        marker=".",
        rasterized=True,
        linewidths=0,
    )
    ax.axis("off")
    plot_utils.add_scale_bar(ax, 0.5)
figure_utils.save(fig, "S10/sel_maps")
```

## Layer3.0 color vs ori

```python
fracs = {"ori": [], "color": [], "both": []}
```

```python
layer = "layer3.0"
fig, ax = plt.subplots(figsize=(2, 2))

tissue = tissues["V1"][layer]
tissue.reset_unit_mask()
cv = tissue.responses.circular_variance
mean_responses = tissue.responses._data.mean("image_idx").values
cv[mean_responses < 1.0] = 1.0

color_responses = tissue.responses._data.groupby("colors").mean().values
bw_resp, color_resp = color_responses
color_diff = color_resp - bw_resp
normalized_diff = np.where(bw_resp > 0, color_diff / bw_resp, 0.0)

passing_ori = np.where(cv <= 0.5)[0]
passing_color = np.where(normalized_diff > 1)[0]
passing_both = np.intersect1d(passing_ori, passing_color)

fracs["ori"].append(len(passing_ori) / len(cv))
fracs["color"].append(len(passing_color) / len(cv))
fracs["both"].append(len(passing_both) / len(cv))

s = 1

# orientation selective
ax.scatter(
    tissue.positions[passing_ori, 0],
    tissue.positions[passing_ori, 1],
    s=s,
    c="#57A750",  # green
    rasterized=True,
)

# color selective
ax.scatter(
    tissue.positions[passing_color, 0],
    tissue.positions[passing_color, 1],
    s=s,
    c="#C73375",  # magenta
    rasterized=True,
)

# color selective
ax.scatter(
    tissue.positions[passing_both, 0],
    tissue.positions[passing_both, 1],
    s=s,
    c="#000",
    rasterized=True,
)
ax.axis("off")
plot_utils.add_scale_bar(ax, 2)
figure_utils.save(fig, "S10/color_v_ori")
```

```python

```
