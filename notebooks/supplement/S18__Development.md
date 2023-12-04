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
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
```

```python
import spacetorch.analyses.core as core
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    spatial_utils,
    array_utils,
)
from spacetorch.analyses.sine_gratings import get_sine_tissue, METRIC_DICT
from spacetorch.analyses.floc import get_floc_tissue
from spacetorch.datasets import floc
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
positions = core.get_positions("simclr_swap")
random_positions = core.get_positions("retinotopic")
```

```python
# load all the SW_0pt25 models
quant_steps = np.arange(start=0, stop=200, step=10)
quant_steps = np.append(quant_steps, 199)
viz_steps = [0, 10, 50, 100, 199]
```

```python
tissues = {
    "V1": {
        step: get_sine_tissue(
            "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
            positions[V1_LAYER],
            layer=V1_LAYER,
            step=step,
        )
        for step in quant_steps
    },
    "VTC": {
        step: get_floc_tissue(
            "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
            positions[VTC_LAYER],
            layer=VTC_LAYER,
            step=step,
        )
        for step in quant_steps
    },
}
```

```python
viz_steps = ["Initialization"] + [0, 10, 50, 100, 199]
print(viz_steps)
```

```python
tissues["V1"]["Initialization"] = get_sine_tissue(
    "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
    positions[V1_LAYER],
    layer=V1_LAYER,
    step="random",
)

tissues["VTC"]["Initialization"] = get_floc_tissue(
    "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
    positions[VTC_LAYER],
    layer=VTC_LAYER,
    step="random",
)
```

```python
fig, axes = plt.subplots(figsize=(4.5, 1.5), ncols=6, nrows=2)
plt.subplots_adjust(hspace=0.01, wspace=0.1)
v1_row = axes[0]
vtc_row = axes[1]

for step, v1_ax, vtc_ax in zip(viz_steps, v1_row, vtc_row):
    v1_tissue = tissues["V1"][step]
    vtc_tissue = tissues["VTC"][step]

    v1_tissue.set_mask_by_pct_limits([[10, 90], [10, 90]])
    pindet = PinwheelDetector(v1_tissue)
    pindet.count_pinwheels()
    pindet.plot(v1_ax, s=1)

    vtc_tissue.make_selectivity_map(
        vtc_ax,
        marker=".",
        size_mult=3e-3,
        selectivity_threshold=12,
        contrasts=[
            contrast_dict["Places"],
            contrast_dict["Objects"],
            contrast_dict["Characters"],
            contrast_dict["Faces"],
            contrast_dict["Bodies"],
        ],
        linewidths=0,
        rasterized=True,
        foreground_alpha=0.8,
    )

    v1_ax.axis("off")
    vtc_ax.axis("off")

    title = step if step == "Initialization" else f"{step + 1} Epochs"
    v1_ax.set_title(title)
figure_utils.save(fig, "S11/maps")
```

```python
metric = METRIC_DICT["angles"]
```

```python
v1_smoothness_results = {"Step": [], "Smoothness": [], "Window": [], "Metric": []}

hcw = 3.5 * (2 / 3)
analysis_width = hcw * 2
max_dist = np.sqrt(2 * analysis_width**2) / 2
bin_edges = np.linspace(0, max_dist, 10)

for metric_name, metric in METRIC_DICT.items():
    for step, tissue in tqdm(tissues["V1"].items()):
        if step == "Initialization":
            step = -10

        tissue.reset_unit_mask()

        tissue.set_unit_mask_by_ptp_percentile(metric_name, 75)
        _, curves = tissue.metric_difference_over_distance(
            distance_cutoff=max_dist,
            bin_edges=bin_edges,
            num_samples=20,
            sample_size=1000,
            shuffle=False,
            verbose=False,
        )
        for window, curve in enumerate(curves):
            smoothness = spatial_utils.smoothness(curve)

            v1_smoothness_results["Metric"].append(figure_utils.get_label(metric_name))
            v1_smoothness_results["Window"].append(window)
            v1_smoothness_results["Smoothness"].append(smoothness)
            v1_smoothness_results["Step"].append(step)
```

```python
N_BINS = 20
distance_cutoff = 60.0  # mm
bin_edges = np.linspace(0, 60, N_BINS)
midpoints = array_utils.midpoints_from_bin_edges(bin_edges)
num_samples = 25
```

```python
vtc_smoothness_results = {"Step": [], "Smoothness": [], "Window": [], "Category": []}

for step, tissue in tqdm(tissues["VTC"].items()):
    if step == "Initialization":
        step = -10

    tissue.reset_unit_mask()
    for contrast in contrasts:
        try:
            _, curves = tissue.category_smoothness(
                contrast=contrast, num_samples=num_samples, bin_edges=bin_edges
            )
        except ValueError:
            continue

        for window, curve in enumerate(curves):
            smoothness = spatial_utils.smoothness(curve)

            vtc_smoothness_results["Category"].append(contrast.name)
            vtc_smoothness_results["Window"].append(window)
            vtc_smoothness_results["Smoothness"].append(smoothness)
            vtc_smoothness_results["Step"].append(step)
```

```python
v1_smoothness_df = pd.DataFrame(v1_smoothness_results)
vtc_smoothness_df = pd.DataFrame(vtc_smoothness_results)
```

```python
fig, axes = plt.subplots(figsize=(0.75, 1.5), nrows=2, gridspec_kw={"hspace": 0.5})
v1_ax, vtc_ax = axes

sns.lineplot(
    data=v1_smoothness_df, x="Step", y="Smoothness", hue="Metric", ax=v1_ax, lw=0.5
)
sns.lineplot(
    data=vtc_smoothness_df,
    x="Step",
    y="Smoothness",
    hue="Category",
    ax=vtc_ax,
    palette=contrast_colors,
    lw=0.5,
)

for ax in axes:
    ax.set_xticks([-10, 50, 100, 199])
    ax.set_xticklabels(["Init", "50", "100", "200"])
    plot_utils.remove_spines(ax)
    figure_utils.move_legend(ax, frameon=False, title="")
    ax.set_xlabel("Epoch")

v1_ax.set_xlabel("")
v1_ax.set_xticks([])
figure_utils.save(fig, "S11/smoothness")
```
