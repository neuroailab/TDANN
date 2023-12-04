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

# Notebook Prep


## Imports

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns
from tqdm import tqdm
```

```python
import spacetorch.analyses.core as core
from spacetorch.analyses.sine_gratings import (
    get_sine_tissue,
    add_sine_colorbar,
    METRIC_DICT,
    get_smoothed_map,
)

from spacetorch.datasets import DatasetRegistry
from spacetorch.datasets.ringach_2002 import load_ringach_data

from spacetorch.utils import (
    figure_utils,
    plot_utils,
    spatial_utils,
    array_utils,
    seed_str,
)

from spacetorch.maps.pinwheel_detector import PinwheelDetector
from spacetorch.maps.screenshot_maps import (
    NauhausOrientationTissue,
    NauhausSFTissue,
    LivingstoneColorTissue,
)
from spacetorch.maps.som import FeaturePoorV1SOM, FeatureRichV1SOM, SOMParams
```

## Basic Setup

```python
figure_utils.set_text_sizes()
marker_fill = "#ccc"
bar_fill = "#ccc"
```

```python
marker_lookup = {
    "TDANN": None,
    "Hand-Crafted SOM": "s",
    "DNN-SOM": "P",
    "Unoptimized": "X",
    "Functional Only": "D",
    "Task Only": "D",
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
    "Task Only": bar_fill,
    "Post-hoc": bar_fill,
    "Gabor Filterbank": bar_fill,
    "Unoptimized": figure_utils.get_color("Unoptimized"),
}
```

## Load Data

```python
V1_LAYER = "layer2.0"
positions = core.get_positions("simclr_swap")[V1_LAYER]
random_positions = core.get_positions("retinotopic")[V1_LAYER]
```

```python
# load neural data
ori_tissue = NauhausOrientationTissue()
sf_tissue = NauhausSFTissue()
color_tissue = LivingstoneColorTissue()

# mask to regions with signal
ori_tissue_mask = sf_tissue_mask = spatial_utils.indices_within_limits(
    ori_tissue.positions, [[0, 0.5], [0, 0.5]]
)
color_tissue_mask = spatial_utils.indices_within_limits(
    color_tissue.positions, [[1, 1.8], [1, 1.8]]
)
```

```python
# load SOMs
soms = {
    "Hand-Crafted SOM": {
        seed: FeaturePoorV1SOM.build(
            params=SOMParams(seed=seed),
            n_training_samples=10_000,
            n_training_iterations=700_000,
        )
        for seed in range(5)
    },
    "DNN-SOM": {
        seed: FeatureRichV1SOM.build(
            params=SOMParams(seed=seed),
            n_training_samples=10_000,
            n_training_iterations=100_000,
        )
        for seed in range(5)
    },
}

for k, seeds in soms.items():
    for seed, v in seeds.items():
        v.make_tissue(
            total_size=4.56, cache_id=f"som_{k}_seed_{seed}"
        )  # total size from swindale bauer scale bars
```

```python
tissues = {
    "SW_0": {
        seed: get_sine_tissue(
            f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0{seed_str(seed)}",
            positions,
            layer=V1_LAYER,
        )
        for seed in range(5)
    },
    "SW_0pt25": {
        seed: get_sine_tissue(
            f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}",
            positions,
            layer=V1_LAYER,
        )
        for seed in range(5)
    },
    "Unoptimized": {
        seed: get_sine_tissue(
            f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}",
            random_positions,
            layer=V1_LAYER,
            step="random",
        )
        for seed in range(5)
    },
    "Hand-Crafted SOM": {
        seed: som.tissue for seed, som in soms["Hand-Crafted SOM"].items()
    },
    "DNN-SOM": {seed: som.tissue for seed, som in soms["DNN-SOM"].items()},
}
```

# Panel A: Stimuli

```python
images, labels = [], []
for item in iter(DatasetRegistry.get("SineGrating2019")):
    images.append(item[0])
    labels.append(item[1])

labels = np.stack(labels)
```

```python
fig, axes = plt.subplots(figsize=(2, 1), ncols=8, nrows=4)
plt.subplots_adjust(hspace=0.1, wspace=0.1)

flag = True
for ori, col in zip(np.unique(labels[:, 0]), axes.T):
    for sf_idx, (sf, ax) in enumerate(zip([18.6, 31.6, 18.6, 31.6], col)):
        flag = not flag
        match = np.where(
            (labels[:, 0] == ori)
            & (labels[:, 1] == sf)
            & (labels[:, 2] == 0)
            & (labels[:, 3] == (sf_idx > 1))
        )[0][0]
        plot_utils.plot_torch_image(ax, images[match][:, 50:100, 50:100])
        ax.axis("off")

figure_utils.save(fig, "F02/a_gratings")
```

# Panel B: Tuning Curves


## Plot TCs

```python
fig, axes = plt.subplots(
    figsize=(2, 1), ncols=4, nrows=2, gridspec_kw={"hspace": 0.8, "wspace": 0.4}
)
tissue = tissues["SW_0pt25"][4]

# choose the units to show
cvs = tissue.responses.circular_variance
sort_ind = np.argsort(cvs)
indices = (100, 10_000, 41_000, 100_000)

sf_tc = tissue.responses._data.groupby("sfs").mean()

for ax_col, idx in zip(axes.T, indices):
    tc_ind = sort_ind[idx]
    tc = tissue.orientation_tc_fits[tc_ind].fit
    plot_utils.plot_tuning_curve(
        ax_col[0], tc, mode="line", plot_params={"lw": 1, "c": "k"}
    )
    mx = np.max(tc)
    ax_col[0].set_ylim([-mx * 0.1, mx * 1.2])
    ax_col[0].set_title(f"CV = {cvs[tc_ind]:.2f}", fontdict={"size": 5})
    ax_col[0].set_yticks([])
    ax_col[0].set_xticks([])

    # spatial frequency tuning curve
    ax_col[1].plot(sf_tc[:, tc_ind], c="k", lw=1)
    plot_utils.remove_spines(ax_col[1])
    ax_col[1].set_yticks([])

axes[0][0].set_xticks([0, 25, 50])
axes[0][0].set_xticklabels([0, 90, 180])
axes[1][0].set_xticks([0, 6])
axes[1][0].set_xticklabels([0, 10])
for ax in axes[1, 1:]:
    ax.set_xticks([])
    ax.set_xticklabels([])
axes[1][0].set_ylabel("")
figure_utils.save(fig, "F02/b_tuning_curves")
```

# Panel C

```python
tissue = tissues["SW_0pt25"][4]
lims = [5, 95]
tissue.set_mask_by_pct_limits([lims, lims])
```

```python
# compute the smoothed orientation preference map (kernel width 1.5 mm, stride 150 microns)
smoothed = get_smoothed_map(
    tissue, METRIC_DICT["angles"], final_width=1.5, final_stride=0.15, verbose=True
)
```

```python
# zoom in for the second panel to a 15% x 15% window
zoom_lims = [[30, 50], [30, 50]]
tissue.set_mask_by_pct_limits(zoom_lims)
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
tissue.make_parameter_map(
    axes[1], final_psm=0.4, linewidths=0.2, edgecolor=(0, 0, 0, 0.4), rasterized=True
)

for ax in axes:
    ax.axis("off")

total_px = smoothed.shape[0]
total_mm = np.ptp(tissue._positions) * 0.9
px_per_mm = total_px / total_mm
plot_utils.add_scale_bar(axes[0], 10 * px_per_mm, flipud=True)

plot_utils.add_scale_bar(axes[1], 1)
cb = plt.colorbar(mappable=mappable, cax=cax, ticks=[0, 90, 180])

# not sure why setting ticks above doesn't directly work, but this does
cb.set_ticks([0, 90, 180])
figure_utils.save(fig, "F02/c_ori_zoom")
```

## Orientation tuning strength analysis

```python

ringach_data = load_ringach_data(fieldname="orivar")


def cv_bin(cv):
    bin_edges = np.linspace(0.0, 1, 40)
    counts, bin_edges = np.histogram(cv, bins=bin_edges)
    counts = counts / np.sum(counts) * 100
    return counts
```

```python
RESP_THRESH = 1.0
CV_THRESH = 0.6

cv_results = {"Name": [], "Seed": [], "% Selective": []}

for name, seeds in tissues.items():
    label = figure_utils.get_label(name)
    for seed, tissue in seeds.items():
        cv = tissue.responses.circular_variance
        mean_responses = tissue.responses._data.mean("image_idx").values
        cv = cv[~np.isnan(cv) & (mean_responses > RESP_THRESH)]

        cv_results["Name"].append(name)
        cv_results["Seed"].append(seed)
        cv_results["% Selective"].append(np.mean(cv < CV_THRESH))

cv_df = pd.DataFrame(cv_results)
```

```python
cv_df.groupby("Name").median()
```

```python
for name in cv_df["Name"].unique():
    matching = cv_df.query(f"Name == '{name}'")["% Selective"]
    print(f"{name}: [{matching.min():.3f}, {matching.max():.3f}]")
```

# Panels (d)-(i)


## Maps

```python
# prepare the macaque data to plot, as (tissue, mask, point_size) tuples
macaque_data = [
    (ori_tissue, ori_tissue_mask, 0.1),
    (sf_tissue, sf_tissue_mask, 0.1),
    (color_tissue, color_tissue_mask, 2.0),
]

# choose TDANN tissue to plot
to_plot = [
    tissues["SW_0pt25"][4],
    tissues["Unoptimized"][4],
]
```

```python
# one column for each model, +1 for neural data
ncols = len(to_plot) + 1
nrows = len(METRIC_DICT)

fig, ax_rows = plt.subplots(
    ncols=ncols, nrows=nrows, figsize=(3, 3.4), gridspec_kw={"hspace": 0.3}
)

for ax in ax_rows.ravel():
    ax.axis("off")

# plot models
for axes, tissue in zip(ax_rows.T[1:], to_plot):

    # restrict to a smaller window (15% of total width on each side)
    tissue.set_mask_by_pct_limits([[30, 45], [30, 45]])

    # make a plot for each "metric": orientations, spatial frequencies, and colors
    for (metric_name, metric), ax in zip(METRIC_DICT.items(), axes):
        scatter_handle = tissue.make_parameter_map(
            ax,
            metric=metric,
            scale_points=True,
            final_psm=0.5,
            rasterized=True,
            linewidths=0.03,
            edgecolor=(0, 0, 0, 0.5),
        )

        # add a colorbar if we're in the last column
        if axes[0] == ax_rows[0, -1]:
            cbar = add_sine_colorbar(fig, ax, metric, label=metric.xlabel)
            cbar.ax.tick_params(labelsize=5)
            cbar.ax.set_yticklabels(
                [metric.xticklabels[0], metric.xticklabels[-1]], rotation=90
            )
            cbar.set_label(label="", fontsize=8)

    plot_utils.add_scale_bar(axes[-1], width=1)
    plt.subplots_adjust(hspace=0.05, wspace=0.01)


# plot macaque data
for ax, (tiss, mask, s) in zip(ax_rows[:, 0], macaque_data):
    ax.scatter(
        *tiss.positions[mask].T,
        c=tiss.data.T.ravel()[mask],
        cmap=tiss.cmap,
        rasterized=True,
        s=s,
        linewidths=0,
    )
    plot_utils.add_scale_bar(ax, 0.2)

figure_utils.save(fig, "F02/dfh_maps")
```

```python
## Revision: Example maps from other models
```

```python
# choose TDANN tissue to plot
bonus_to_plot = {
    "TDANN": tissues["SW_0pt25"][4],
    "Task Only": tissues["SW_0"][0],
    "Unoptimized": tissues["Unoptimized"][4],
    "DNN-SOM": tissues["DNN-SOM"][0],
    "Hand-Crafted SOM": tissues["Hand-Crafted SOM"][0],
}
```

```python
# one column for each model, +1 for neural data
ncols = len(bonus_to_plot) + 1
nrows = len(METRIC_DICT)

fig, ax_rows = plt.subplots(
    ncols=ncols, nrows=nrows, figsize=(ncols * 0.8, nrows), gridspec_kw={"hspace": 0.3}
)

for ax in ax_rows.ravel():
    ax.axis("off")

# plot models
for axes, (name, tissue) in zip(ax_rows.T[1:], bonus_to_plot.items()):
    # restrict to a smaller window (15% of total width on each side)
    tissue.set_mask_by_pct_limits([[30, 45], [30, 45]])
        
    # make a plot for each "metric": orientations, spatial frequencies, and colors
    for (metric_name, metric), ax in zip(METRIC_DICT.items(), axes):
        scatter_handle = tissue.make_parameter_map(
            ax,
            metric=metric,
            scale_points=True,
            final_s=18,
            rasterized=True,
            linewidths=0.03,
            edgecolor=(0, 0, 0, 0.5),
            marker='s' if "SOM" in name else '.'
        )

        # add a colorbar if we're in the last column
        if axes[0] == ax_rows[0, -1]:
            cbar = add_sine_colorbar(fig, ax, metric, label=metric.xlabel)
            cbar.ax.tick_params(labelsize=5)
            cbar.ax.set_yticklabels(
                [metric.xticklabels[0], metric.xticklabels[-1]], rotation=90
            )
            cbar.set_label(label="", fontsize=8)

    plot_utils.add_scale_bar(
        axes[-1],
        width=0.2 if "SOM" in name else 1,
        y_start=1.33 if "SOM" in name else 0
    )
    plt.subplots_adjust(hspace=0.05, wspace=0.01)

# plot macaque data
for ax, (tiss, mask, s) in zip(ax_rows[:, 0], macaque_data):
    ax.scatter(
        *tiss.positions[mask].T,
        c=tiss.data.T.ravel()[mask],
        cmap=tiss.cmap,
        rasterized=True,
        s=s,
        linewidths=0,
    )
    plot_utils.add_scale_bar(ax, 0.2)
figure_utils.save(fig, "F02/dfh_maps_bonus")
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
curve_tissues = {
    "Macaque V1": {0: {"angles": ori_tissue, "sfs": sf_tissue, "colors": color_tissue}},
    "SW_0pt25": tissues["SW_0pt25"],
    "Hand-Crafted SOM": {
        seed: som.tissue for seed, som in soms["Hand-Crafted SOM"].items()
    },
    "DNN-SOM": {seed: som.tissue for seed, som in soms["DNN-SOM"].items()},
    "SW_0": tissues["SW_0"],
    "Unoptimized": tissues["Unoptimized"],
}
```

```python
def get_curves(
    name: str,
    seed: int,
    metric_name: str,
    shuffle: bool = False,
    num_samples: int = 20,
    sample_size: int = 1000,
    verbose: bool = False,
):
    # make the analysis width slightly larger to capture full rise and fall
    hcw = hypercolumn_widths[name]
    analysis_width = hcw * (4 / 3)

    # compute largest possible distance given the window size
    max_dist = np.sqrt(2 * analysis_width**2) / 2

    # create 9 bins, going from 0 (closest) to max_dist
    bin_edges = np.linspace(0, max_dist, 10)
    midpoints = array_utils.midpoints_from_bin_edges(bin_edges)

    # convenience: store arguments shared by both conditional flows into a dict
    common = {
        "num_samples": num_samples,
        "sample_size": sample_size,
        "bin_edges": bin_edges,
        "shuffle": shuffle,
        "verbose": verbose,
    }

    if name == "Macaque V1":
        tissue = curve_tissues["Macaque V1"][0][metric_name]
        _, curves = tissue.difference_over_distance(**common)
    else:
        tissue = curve_tissues[name][seed]
        tissue.reset_unit_mask()
        tissue.set_unit_mask_by_ptp_percentile(metric_name, 75)
        _, curves = tissue.metric_difference_over_distance(
            distance_cutoff=max_dist, **common
        )

    # normalize midpoints to be a fraction of the hypercolumn width
    return midpoints / hcw, curves
```

```python
smoothness_results = {"Smoothness": [], "Model": [], "Seed": [], "Metric": []}

curve_dict = {}
for metric_name in METRIC_DICT.keys():
    curve_dict[metric_name] = {}

    for name, seeds in tqdm(curve_tissues.items()):
        all_curves = []
        for seed, tissue in seeds.items():
            distances, curves = get_curves(name, seed, metric_name)

            # compute smoothness for each curve
            smoos = [spatial_utils.smoothness(curve) for curve in curves]
            mean_smoothness = np.nanmean(smoos)

            smoothness_results["Smoothness"].append(mean_smoothness)
            smoothness_results["Model"].append(figure_utils.get_label(name))
            smoothness_results["Seed"].append(seed)
            smoothness_results["Metric"].append(metric_name)

            # for plotting, figure out what value we'd expect by chance
            _, chance_curves = get_curves(name, seed, metric_name, shuffle=True)
            chance_mean = np.nanmean(np.concatenate(chance_curves))
            norm_curves = [curve / chance_mean for curve in curves]
            all_curves.extend(norm_curves)

        curve_dict[metric_name][name] = {
            "Distances": distances * 100,  # convert to percentages
            "Curves": all_curves,
        }
```

```python
sr = pd.DataFrame(smoothness_results)
model_palette = {name: figure_utils.get_color(name) for name in sr.Model.unique()}
```

```python
# some plotting params
ylabel_lookup = {
    "angles": (r"$\Delta$ Preferred" "\n" "Orientation"),
    "sfs": (r"$\Delta$ Spatial " "\n" "Frequency"),
    "colors": ("Fraction Pref. " "\n" "Other Color"),
}
order = ["TDANN", "Hand-Crafted SOM", "DNN-SOM", "Functional Only", "Unoptimized"]
mkw = {"color": marker_fill, "edgecolors": "k", "linewidths": 0.5}
```

```python
for metric_name, names in curve_dict.items():
    fig, axes = plt.subplots(
        figsize=(2.5, 0.8),
        ncols=2,
        gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.6},
    )

    for name in [
        "Unoptimized",
        "DNN-SOM",
        "Hand-Crafted SOM",
        "Macaque V1",
        "SW_0pt25",
    ]:
        res = names[name]
        curves = np.stack(res["Curves"])
        mn_curve = np.mean(curves, axis=0)
        se = np.std(curves, axis=0)
        label = figure_utils.get_label(name)
        line_color = color_lookup[label]
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
            markevery=2,
            markersize=3,
            mew=0.5,
            lw=1.2,
        )
        axes[0].fill_between(
            res["Distances"],
            mn_curve - se,
            mn_curve + se,
            alpha=0.3,
            facecolor=line_handle[0].get_color(),
        )

    axes[0].legend().remove()
    axes[0].set_yticks([0, 1])
    plot_utils.remove_spines(axes[0])

    # add the x label if we're at the bottom fo the figure: color pref maps
    if metric_name == "colors":
        axes[0].set_xlabel("Pairwise Distance\n(% Column Spacing)")

    # always add the ylabel
    axes[0].set_ylabel(ylabel_lookup[metric_name])

    # bars
    sub = sr[sr.Metric == metric_name]

    # treat macaque differently (as in Fig 5)
    plot_sub = sub[sub.Model != "Macaque V1"]

    sns.barplot(
        data=plot_sub,
        x="Model",
        y="Smoothness",
        palette=color_lookup,
        order=order,
        ax=axes[1],
    )

    macaque_val = np.array(sub[sub.Model == "Macaque V1"].Smoothness)
    unoptimized_val = np.array(sub[sub.Model == "Unoptimized"].Smoothness)

    axes[1].axhline(
        np.mean(macaque_val),
        color=figure_utils.get_color("Macaque V1"),
        linestyle="dashed",
        lw=1,
    )
    for i, name in enumerate(order):
        marker = marker_lookup[name]
        if marker is None:
            continue
        axes[1].scatter(i, 0.1, marker=marker_lookup[name], **mkw, s=5, zorder=5)

    plot_utils.remove_spines(axes[1])
    axes[1].set_xticks([])
    axes[1].set_ylim([0, 1.2])
    axes[1].set_yticks([0, 1])
    axes[1].set_ylabel("Smoothness", labelpad=0)
    axes[1].set_xlabel("")
    figure_utils.save(fig, f"F02/egh_quant_{metric_name}")
```

```python
for metric in sr.Metric.unique():
    print(f"======\n{metric}\n========")
    met_sr = sr.query(f"Metric == '{metric}'")

    for model in met_sr.Model.unique():
        matching = met_sr.query(f"Model== '{model}'")["Smoothness"]
        print(f"{model}: [{min(matching):.3f}, {max(matching):.3f}]")
```

# Panel J

```python
# compute the value for macaque OPM smoothness
ori_only = sr[sr.Metric == "angles"]
macaque_value = ori_only[ori_only.Model == "Macaque V1"]["Smoothness"].item()
```

```python
# compute the relative smoothness (relative to macaque v1) for each model
rel_smoothness = {}
for name in ori_only["Model"].unique():
    if "Macaque" in name:
        continue

    data = ori_only[ori_only.Model == name]["Smoothness"]
    rel = data / macaque_value
    rel_smoothness[name] = rel.tolist()
```

```python
# compute KS distance between CV distributions
cv_sim = {}
for tname, seeds in tissues.items():
    label = figure_utils.get_label(tname)
    cv_sim[label] = []
    for seed, tissue in seeds.items():
        cv = tissue.responses.circular_variance
        mean_responses = tissue.responses._data.mean("image_idx").values
        cv = cv[~np.isnan(cv) & (mean_responses > RESP_THRESH)]
        dist = cv_bin(cv)
        ks, _ = ks_2samp(cv_bin(ringach_data), dist)

        cv_sim[label].append(1 - ks)
```

```python
fig, ax = plt.subplots(figsize=(1.25, 1))
rng = default_rng(seed=424)

ax.axhline(1.0, linestyle="dashed", c=figure_utils.get_color("Macaque V1"), lw=1)
ax.axvline(1.0, linestyle="dashed", c=figure_utils.get_color("Macaque V1"), lw=1)

for name in cv_sim:
    sim_list = np.array(cv_sim[name])
    smoo_list = np.array(rel_smoothness[name])

    ax.scatter(
        sim_list + rng.normal(scale=1e-2, size=sim_list.shape),
        smoo_list + rng.normal(scale=1e-2, size=smoo_list.shape),
        marker=marker_lookup[name],
        s=30,
        edgecolor="k",
        c=color_lookup[name],
        linewidths=0.5,
        alpha=0.7,
    )


ax.set_xlim([0, 1.1])
ax.set_ylim([-0.1, 1.6])
plot_utils.remove_spines(ax)
ax.set_ylabel("Similarity to Macaque\nMap Smoothness")
ax.set_xlabel("Similarity to Macaque CV")
figure_utils.save(fig, "F02/j_scatter")
```

# Panel k

```python
pinwheel_res = {"Model": [], "Seed": [], "Density": []}

# plot models
for name, seeds in tqdm(tissues.items()):
    is_som = "SOM" in name
    edge_size = np.ptp(seeds[0]._positions)
    num_hcol = edge_size / hypercolumn_widths[name]

    for seed, tissue in seeds.items():
        tissue.reset_unit_mask()

        pindet = PinwheelDetector(tissue, size_mult=(4.56 / 24.5) if is_som else 1.5)
        pos, neg = pindet.count_pinwheels(var_thresh=3.5 if is_som else 0.3)
        total = pos + neg

        density = total / (num_hcol**2)

        pinwheel_res["Model"].append(figure_utils.get_label(name))
        pinwheel_res["Seed"].append(seed)
        pinwheel_res["Density"].append(density)

pinwheel_df = pd.DataFrame(pinwheel_res)
```

```python
fig, ax = plt.subplots(figsize=(1.25, 1))

model_order = ["TDANN", "Hand-Crafted SOM", "DNN-SOM", "Functional Only", "Unoptimized"]
sns.barplot(
    data=pinwheel_df,
    x="Model",
    y="Density",
    order=model_order,
    palette=color_lookup,
    ax=ax,
)
ax.axhline(3.14)
```

```python
pinwheel_df.groupby("Model").mean()
```

```python
for model in pinwheel_df.Model.unique():
    matching = pinwheel_df.query(f"Model == '{model}'")["Density"]
    print(f"{model}: [{matching.min():3f}, {matching.max():.3f}]")
```

```python
pinwheel_df.anova(dv="Density", between="Model")
```

```python
pinwheel_df.pairwise_tukey(dv="Density", between="Model")
```

```python
color_lookup
```

```python
fig, ax = plt.subplots(figsize=(1.25, 1))

model_order = ["TDANN", "Hand-Crafted SOM", "DNN-SOM", "Functional Only", "Unoptimized"]
sns.barplot(
    data=pinwheel_df,
    x="Model",
    y="Density",
    order=model_order,
    palette=color_lookup,
    ax=ax,
)

for i, name in enumerate(model_order):
    marker = marker_lookup[name]
    if marker is None:
        continue
    ax.scatter(i, 0.5, marker=marker, **mkw, s=15, zorder=5)

ax.set_xticks([])
ax.set_xticklabels([])
plot_utils.remove_spines(ax)
ax.set_ylabel(("Pinwheels /" "\n" r"Column Spacing$^2$"))
ax.axhline(np.pi, color=figure_utils.get_color("Macaque V1"), linestyle="dashed", lw=1)
ax.set_xlabel("")
figure_utils.save(fig, "F02/k_density")
```

# Legend

```python
legend_entries = [
    "TDANN",
    "Macaque V1",
    "Hand-Crafted SOM",
    "DNN-SOM",
    "Task Only",
    "Unoptimized",
][::-1]
```

```python
fig, ax = plt.subplots(figsize=(1.5, 0.75))
for i, entry in enumerate(legend_entries):
    x_offset = 0 if i > 2 else 0.7
    ax.text(
        0.15 + x_offset,
        (i % 3) / len(legend_entries),
        entry,
        horizontalalignment="left",
    )

    # add icon
    ax.scatter(
        0.1 + x_offset,
        (i % 3) / len(legend_entries) + 0.05,
        s=10,
        marker=marker_lookup[entry],
        edgecolors="k",
        linewidths=0.5,
        c=color_lookup[entry],
    )

ax.set_xlim([0, 1])
ax.set_ylim([-0.1, 1])
ax.axis("off")
figure_utils.save(fig, "F02/xx_legend")
```

```python

```
