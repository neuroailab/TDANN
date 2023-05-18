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
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import YlGnBu

import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from numpy.random import default_rng
```

```python
import spacetorch.analyses.core as core
from spacetorch.models.trunks.resnet import LAYER_ORDER
from spacetorch.analyses.imagenet import create_imnet_tissue
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    spatial_utils,
    seed_str,
)
from spacetorch.utils.vissl.performance import Performance
from spacetorch.paths import RESULTS_DIR
```

```python
figure_utils.set_text_sizes()
```

```python
spatial_results_path = RESULTS_DIR / "spatial_loss_results.pkl"
task_results_path = RESULTS_DIR / "task_loss_results.pkl"

spatial_loss_df = pd.read_pickle(spatial_results_path)
task_loss_df = pd.read_pickle(task_results_path)
```

```python
spatial_loss_df["Layer"]
```

```python
lookup = {
    "layer1_0": "Layer 2",
    "layer1_1": "Layer 3",
    "layer2_0": "Layer 4",
    "layer2_1": "Layer 5",
    "layer3_0": "Layer 6",
    "layer3_1": "Layer 7",
    "layer4_0": "Layer 8",
    "layer4_1": "Layer 9",
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
spatial_loss_df.Layer.replace(lookup, inplace=True)
palette = {lookup[layer]: figure_utils.get_color(layer) for layer in LAYER_ORDER}
```

```python
fig, axes = plt.subplots(figsize=(2.5, 1), ncols=2, gridspec_kw={"wspace": 1})

common = {
    "x": "Iteration",
    "y": "Loss",
}
sns.lineplot(data=task_loss_df, **common, color="k", ax=axes[0])
sns.lineplot(data=spatial_loss_df, **common, hue="Layer", ax=axes[1], palette=palette)

axes[0].set_ylabel("Task Loss")
axes[1].set_ylabel("Spatial Loss")

for ax in axes:
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim([-0.1, 200])
figure_utils.move_legend(axes[1], frameon=False)

leg = axes[1].get_legend()

for ax in axes:
    plot_utils.remove_spines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylim([0, None])

figure_utils.save(fig, "S01/loss_curves")
```

# other plots

```python
model_name = "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
pos_type = "simclr_swap"
position_dict = core.get_positions(pos_type)
```

```python
# create model tissue for each layer, storing ImageNet responses and positions for each unit
imagenet_features_dict = core.get_features_from_model(
    model_name, "ImageNet", verbose=True, max_batches=32, batch_size=32
)

tissues = {
    layer: create_imnet_tissue(
        imagenet_features_dict[f"base_model.{layer}"], position_dict[layer].coordinates
    )
    for layer in sorted(position_dict.keys())
}
```

```python
# specify the width (in mm) to restrict each plotted region to
widths = {
    "layer1.0": 0.5,
    "layer1.1": 0.5,
    "layer2.0": 5,
    "layer2.1": 5,
    "layer3.0": 10,
    "layer3.1": 10,
    "layer4.0": 20,
    "layer4.1": 20,
}

vmin, vmax = -0.4, 0.4
```

```python
title_lookup = {
    "layer1.0": "Layer 2",
    "layer2.0": "Layer 4",
    "layer3.0": "Layer 6",
    "layer4.0": "Layer 8",
}

fig, axes = plt.subplots(figsize=(2, 0.5), ncols=4, nrows=1)
plt.subplots_adjust(wspace=0.05)
seed = 424
rng = default_rng(seed=seed)

for ax, layer in zip(axes.ravel(), LAYER_ORDER[::2]):
    tissue = tissues[layer]
    tissue.reset_unit_mask()
    mid = max(tissue.positions[:, 0]) / 2
    width = widths[layer]

    lims = 2 * [[mid - width, mid + width]]
    tissue.set_mask_by_limits(lims)

    # further subselect to <= 8000
    num_keep = min(8000, len(tissue.unit_mask))
    tissue.unit_mask = rng.choice(tissue.unit_mask, size=num_keep, replace=False)
    features = tissue.responses._data.values[:, tissue.unit_mask]
    pairwise_corr = np.corrcoef(features.T)
    pairwise_corr[np.diag_indices_from(pairwise_corr)] = np.nan

    ax.set_title(
        title_lookup[layer],
        fontdict={"color": figure_utils.get_color(layer), "size": 6},
        pad=1,
    )
    correlation_vector = pairwise_corr[seed, :]

    scatter_handle = ax.scatter(
        tissue.positions[:, 0],
        tissue.positions[:, 1],
        cmap="bwr",
        c=correlation_vector,
        vmin=vmin,
        vmax=vmax,
        s=0.2,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(*tissue.positions[seed], marker="*", c="k", s=20, linewidths=0)

    plot_utils.remove_spines(ax, to_remove="all")
    ax.set_xticks([])
    ax.set_yticks([])
    plot_utils.add_scale_bar(ax, width / 2)
    print(width / 2)

# custom cbar
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(
    scatter_handle,
    cax=cax,
    orientation="vertical",
    label="Correlation",
    ticks=[-0.3, 0, 0.3],
)
figure_utils.save(fig, "S01/seed_maps")
```

## Curves

```python
def add_custom_cbar(fig):
    ax = fig.add_axes([0.94, 0.12, 0.03, 0.76])

    cmap = YlGnBu
    norm = Normalize(vmin=0, vmax=1)
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical", ticks=[])
    cb.set_label("")
    ax.set_yticks([])
```

```python
fig, ax = plt.subplots(figsize=(2, 1))

widths = {
    "layer1.0": 1,
    "layer1.1": 1,
    "layer2.0": 5,
    "layer2.1": 5,
    "layer3.0": 5,
    "layer3.1": 5,
    "layer4.0": 40,
    "layer4.1": 40,
}

for lidx, layer in enumerate(LAYER_ORDER):
    layer_pos = position_dict[layer]
    nbw = layer_pos.neighborhood_width
    tissue = tissues[layer]
    tissue.reset_unit_mask()
    extent = np.ptp(tissue.positions[:, 0])

    analysis_width = widths[layer]
    window_params = spatial_utils.WindowParams(
        width=widths[layer] / 2,
        window_number_limit=50,
        edge_buffer=0,
        unit_number_limit=1000,
    )

    curves = tissue.correlation_over_distance_plot(
        ax,
        window_params=window_params,
        legend_label=layer,
        line_color=figure_utils.get_color(layer),
        normalize_x_axis=True,
    )

add_custom_cbar(fig)

ax.legend().remove()
ax.set_ylim([-0.1, 0.4])
ax.set_xlabel("Normalized Cortical Distance")
ax.set_ylabel("Response Similarity", labelpad=-1)
plot_utils.remove_spines(ax)
figure_utils.save(fig, "S01/corr_curves")
```

## Performance

```python
from spacetorch.paths import CHECKPOINT_DIR
```

```python
LIN_EVAL_DIR = CHECKPOINT_DIR / "linear_eval"
```

```python
paths = {
    "Functional Only": {
        seed: LIN_EVAL_DIR
        / f"simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0{seed_str(seed)}_linear_eval_checkpoints"
        / "metrics.json"
        for seed in range(5)
    },
    "TDANN": {
        seed: LIN_EVAL_DIR
        / f"simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}_linear_eval_checkpoints"
        / "metrics.json"
        for seed in range(5)
    },
}

performances = []
for name, seeds in paths.items():
    for seed, path in seeds.items():
        if not path.exists():
            continue
        performances.append(
            Performance(model=name, spatial_weight=None, path=path, seed=seed)
        )
```

```python
performance_results = {"Accuracy": [], "Precision": [], "Model": [], "Seed": []}

for performance in performances:
    for precision in ["top_1", "top_5"]:
        performance_results["Accuracy"].append(performance.best(precision))
        performance_results["Precision"].append(figure_utils.get_label(precision))
        performance_results["Model"].append(performance.model)
        performance_results["Seed"].append(performance.seed)

performance_df = pd.DataFrame(performance_results)
top1_df = performance_df[performance_df.Precision == "Top-1"]
```

```python
palette = {
    "Functional Only": figure_utils.get_color("Functional Only"),
    "TDANN": figure_utils.get_color("TDANN"),
}
```

```python
fig, ax = plt.subplots(figsize=(0.4, 1))
sns.barplot(data=top1_df, x="Model", y="Accuracy", ax=ax, palette=palette)

labels = [(0, "Task Only", "white"), (1, "Task + Spatial", "white")]

for idx, txt, col in labels:
    ax.text(
        idx,
        3,
        txt,
        c=col,
        rotation=90,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontdict={"weight": 700, "size": 5},
    )

ax.set_xticks([])
ax.set_xlim([-0.7, 1.7])
ax.set_ylabel("Categorization\nAccuracy (%)")
ax.set_xlabel("")
plot_utils.remove_spines(ax)
figure_utils.save(fig, "S01/cat_acc")
```

```python
display(
    pg.mwu(
        np.array(top1_df.query("Model == 'Functional Only'").Accuracy),
        np.array(top1_df.query("Model == 'TDANN'").Accuracy),
    )
)
```

```python
top1_df.groupby("Model").median()
```

```python

```
