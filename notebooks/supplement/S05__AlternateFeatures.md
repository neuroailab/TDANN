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

# Notebook Prep


## Imports

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
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
from spacetorch.analyses.gfb import get_gfb_tissue
from spacetorch.maps.pinwheel_detector import PinwheelDetector
from spacetorch.models.positions import NetworkPositions
from spacetorch.paths import POSITION_DIR
```

## Basic Setup

```python
figure_utils.set_text_sizes()
marker_fill = "#ccc"
bar_fill = "#ccc"
```

## Load Data

```python
V1_LAYER = "layer2.0"
positions = core.get_positions("simclr_swap")[V1_LAYER]
sup_pos = core.get_positions("supervised_swap")[V1_LAYER]
random_positions = core.get_positions("retinotopic")[V1_LAYER]
```

### Map Quantification

```python
hypercolumn_widths = {
    "Hand-Crafted SOM": 0.75,
    "DNN-SOM": 0.75,
    "SW_0": 3.5,  # this is approximately NN spacing of 90-deg sel units in a topographic model
    "SW_0pt25": 3.5,
    "Unoptimized": 3.5,
    "Macaque V1": 0.75,  # estimated spacing of same-orientation columns
}
```

```python
ylabel_lookup = {
    "angles": (r"$\Delta$ Preferred" "\n" "Orientation"),
    "sfs": (r"$\Delta$ Spatial " "\n" "Frequency"),
    "colors": ("Fraction Preferring " "\n" "Other Color"),
}
```

```python
marker_lookup = {
    "TDANN": None,
    "Hand-Crafted SOM": "s",
    "DNN-SOM": "P",
    "Unoptimized": "X",
    "Functional Only": "D",
    "Macaque V1": None,
    "Post-hoc": "v",
    "Gabor Filterbank": "o",
}

color_lookup = {
    "TDANN": figure_utils.get_color("TDANN"),
    "Hand-Crafted SOM": bar_fill,
    "DNN-SOM": bar_fill,
    "Macaque V1": figure_utils.get_color("Macaque V1"),
    "Functional Only": bar_fill,
    "Post-hoc": bar_fill,
    "Gabor Filterbank": bar_fill,
    "Unoptimized": figure_utils.get_color("Unoptimized"),
}
```

```python
order = ["TDANN", "Hand-Crafted SOM", "DNN-SOM", "Functional Only"]
```

```python
mkw = {"color": marker_fill, "edgecolors": "k", "linewidths": 0.5}
```

## Swap, GFB

```python
imagenet_trained_imagenet_swapped = NetworkPositions.load_from_dir(
    (
        f"{POSITION_DIR}"
        "/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1"
        "/resnet18_retinotopic_init_fuzzy_swappedon_ImageNet"
    )
).layer_positions[V1_LAYER]
imagenet_trained_imagenet_swapped.coordinates = (
    imagenet_trained_imagenet_swapped.coordinates * 1.5
)

imagenet_trained_sine_gratings_swapped = NetworkPositions.load_from_dir(
    (
        f"{POSITION_DIR}"
        "/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1"
        "/resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019"
    )
).layer_positions[V1_LAYER]
imagenet_trained_sine_gratings_swapped.coordinates = (
    imagenet_trained_sine_gratings_swapped.coordinates * 1.5
)

nonspatial_imagenet_swapped = NetworkPositions.load_from_dir(
    (
        f"{POSITION_DIR}"
        "/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0"
        "/resnet18_retinotopic_init_fuzzy_swappedon_ImageNet"
    )
).layer_positions[V1_LAYER]
nonspatial_imagenet_swapped.coordinates = nonspatial_imagenet_swapped.coordinates * 1.5

sup_pos = NetworkPositions.load_from_dir(
    (
        f"{POSITION_DIR}"
        "/supervised_resnet18"
        "/resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019"
    )
).layer_positions[V1_LAYER]
sup_pos.coordinates = sup_pos.coordinates * 1.5
```

```python
all_tissues = {
    "ImageNet": {
        "TDANN": get_sine_tissue(
            "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
            positions,
            layer=V1_LAYER,
        ),
        "Post-hoc": get_sine_tissue(
            "nonspatial/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0",
            nonspatial_imagenet_swapped,
        ),
        "Gabor Filterbank": get_gfb_tissue(position_type="ImageNet"),
    },
    "Sine Gratings": {
        "TDANN": get_sine_tissue(
            "by_training_data/simclr_spatial_resnet18_trained_sine_gratings",
            sup_pos,
            skip_cache=False,
        ),
        "Post-hoc": get_sine_tissue(
            "nonspatial/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0",
            positions,
        ),
        "Gabor Filterbank": get_gfb_tissue(position_type="SineGrating2019"),
    },
}
```

```python
swap_pal = {
    "TDANN": figure_utils.get_color("TDANN"),
    "Post-hoc": figure_utils.get_color("Functional Only"),
    "Gabor Filterbank": "#9EB500",
}
```

```python
fig, ax_rows = plt.subplots(ncols=3, nrows=2, figsize=(3, 2))
plt.subplots_adjust(hspace=0.1, wspace=0.1)

lims = [10, 90]
pw_counts = {}
for axes, (dataset, tissue_dict) in zip(ax_rows, all_tissues.items()):
    N = len(tissue_dict)
    for ax, (name, tissue) in zip(axes, tissue_dict.items()):
        mm_extent = 0.8 * np.ptp(tissue._positions)
        if ax == axes[0]:
            ax.text(
                -0.1,
                0.5,
                dataset,
                rotation=90,
                fontsize=6,
                transform=ax.transAxes,
                verticalalignment="center",
                horizontalalignment="center",
            )

        tissue.set_mask_by_pct_limits([lims, lims])
        pindet = PinwheelDetector(tissue)
        pos, neg = pindet.count_pinwheels()
        pw_counts[f"{dataset}_{name}"] = pos + neg

        density = round((pos + neg) / (mm_extent**2), 2)
        smoothed = pindet.smoothed

        pos_centers, neg_centers = pindet.centers
        for x, y in pos_centers:
            ax.scatter(x, y, c="k", s=1, linewidths=0)

        for x, y in neg_centers:
            ax.scatter(x, y, c="w", s=1, linewidths=0)

        mappable = ax.imshow(
            smoothed, cmap=METRIC_DICT["angles"].colormap, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plot_utils.remove_spines(ax, to_remove=["top", "left", "right", "bottom"])

        px_per_mm = smoothed.shape[0] / mm_extent
        plot_utils.add_scale_bar(ax, 2 * px_per_mm, flipud=True)

        # if dataset == "ImageNet":
        # ax.set_title(name, fontdict={"size": 6, "color": 'k'})


cbar_ax = fig.add_axes([0.92, 0.13, 0.015, 0.75])
cb = fig.colorbar(
    mappable,
    cax=cbar_ax,
    ticks=[0, 180],
    extend="both",
    extendrect=True,
)
cb.set_label("Orientation")

figure_utils.save(fig, "S14/swapopt_maps")
```

```python
swap_smoothness_results = {"Smoothness": [], "Dataset": [], "Model": []}

hcw = hypercolumn_widths["SW_0pt25"] * (2 / 3)
analysis_width = hcw * 2
max_dist = np.sqrt(2 * analysis_width**2) / 2
bin_edges = np.linspace(0, max_dist, 10)
midpoints = array_utils.midpoints_from_bin_edges(bin_edges)

swap_curve_dict = {}
for dataset, tissue_dict in all_tissues.items():
    swap_curve_dict[dataset] = {}

    for name, tissue in tqdm(tissue_dict.items()):
        tissue.reset_unit_mask()
        tissue.set_unit_mask_by_ptp_percentile("angles", 75)
        distances, curves = tissue.metric_difference_over_distance(
            distance_cutoff=max_dist,
            bin_edges=bin_edges,
            num_samples=5,
            sample_size=1000,
            shuffle=False,
            verbose=False,
        )
        _, chance_curves = tissue.metric_difference_over_distance(
            distance_cutoff=max_dist,
            bin_edges=bin_edges,
            num_samples=5,
            sample_size=1000,
            shuffle=True,
            verbose=False,
        )
        chance_mean = np.nanmean(np.concatenate(chance_curves))
        norm_curves = [curve / chance_mean for curve in curves]

        smoos = [spatial_utils.smoothness(curve) for curve in curves]
        for smoo in smoos:
            swap_smoothness_results["Smoothness"].append(smoo)
            swap_smoothness_results["Model"].append(figure_utils.get_label(name))
            swap_smoothness_results["Dataset"].append(dataset)

        swap_curve_dict[dataset][name] = {
            "Distances": distances / hcw * 100,
            "Curves": norm_curves,
        }
```

```python
swap_smoothness_df = pd.DataFrame(swap_smoothness_results)
```

```python
for dataset in swap_smoothness_df.Dataset.unique():
    print(f"=====\n{dataset}\n=====")
    sub_df = swap_smoothness_df.query(f"Dataset == '{dataset}'")
    anova_res = sub_df.anova(dv="Smoothness", between="Model")

    pairwise_res = sub_df.pairwise_tukey(dv="Smoothness", between="Model")

    display(anova_res)
    display(pairwise_res)
```

```python
order = ["TDANN", "Post-hoc", "Gabor Filterbank"]
```

```python
for dataset_name, names in swap_curve_dict.items():
    fig, axes = plt.subplots(
        figsize=(1.75, 0.5),
        ncols=2,
        gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.7},
    )
    for name in ["Gabor Filterbank", "Post-hoc", "TDANN"]:
        res = names[name]
        curves = np.stack(res["Curves"])
        mn_curve = np.mean(curves, axis=0)
        se = np.std(curves, axis=0)
        label = figure_utils.get_label(name)
        line_color = color_lookup[name]
        if line_color == marker_fill:
            line_color = "k"

        line_handle = axes[0].plot(
            res["Distances"],
            mn_curve,
            label=name,
            color=line_color,
            mec="k",
            mfc=marker_fill,
            marker=marker_lookup[label],
            markersize=3,
            mew=0.5,
            lw=1,
        )
        axes[0].fill_between(
            res["Distances"],
            mn_curve - se,
            mn_curve + se,
            alpha=0.3,
            facecolor=line_handle[0].get_color(),
        )

    axes[0].legend().remove()
    axes[0].set_yticks([1])
    axes[0].set_ylim([0, 1.5])
    plot_utils.remove_spines(axes[0])
    if dataset_name == "Sine Gratings":
        axes[0].set_xlabel("Pairwise Distance\n(% Column Spacing)")
    axes[0].set_ylabel(ylabel_lookup["angles"])
    axes[0].set_yticks([0, 1])

    # bars
    sub = swap_smoothness_df[swap_smoothness_df.Dataset == dataset_name]
    sns.barplot(
        data=sub,
        x="Model",
        y="Smoothness",
        palette=color_lookup,
        ax=axes[1],
        order=order,
    )
    for i, name in enumerate(order):
        marker = marker_lookup[name]
        if marker is None:
            continue
        axes[1].scatter(i, 0.1, marker=marker, **mkw, s=10, zorder=5)

    plot_utils.remove_spines(axes[1])
    axes[1].set_xticks([])
    axes[1].set_yticks([0, 1])
    axes[1].set_ylim([0, 1.3])
    axes[1].set_ylabel("Smoothness", labelpad=0)

    figure_utils.save(fig, f"S14/quant_{dataset_name}")
```

```python
# custom legend
```

```python
legend_entries = [
    "TDANN",
    # "Macaque V1",
    "Post-hoc",
    "Gabor Filterbank",
][::-1]
```

```python
fig, ax = plt.subplots(figsize=(0.75, 0.35))
for i, entry in enumerate(legend_entries):
    ax.text(0.2, i / len(legend_entries), entry, horizontalalignment="left", fontsize=5)

    # add icon
    ax.scatter(
        0.1,
        i / len(legend_entries) + 0.11,
        s=10,
        marker=marker_lookup[entry],
        edgecolors="k",
        linewidths=0.5,
        c=color_lookup[entry],
    )

ax.set_xlim([0, 1])
ax.set_ylim([-0.1, 1])
ax.axis("off")
figure_utils.save(fig, "S14/legend")
```

```python

```
