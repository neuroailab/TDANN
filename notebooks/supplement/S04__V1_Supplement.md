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
import seaborn as sns
```

```python
import spacetorch.analyses.core as core
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    spatial_utils,
    array_utils,
    seed_str,
)
from spacetorch.analyses.sine_gratings import (
    get_sine_tissue,
    add_sine_colorbar,
    METRIC_DICT,
)
from spacetorch.datasets.ringach_2002 import load_ringach_data
from spacetorch.maps.som import FeaturePoorV1SOM, FeatureRichV1SOM, SOMParams
from spacetorch.maps.screenshot_maps import (
    NauhausOrientationTissue,
    NauhausSFTissue,
    LivingstoneColorTissue,
)
```

```python
figure_utils.set_text_sizes()
```

# Maps: all seeds, SOM, Unoptimized, Task Only

```python
V1_LAYER = "layer2.0"
simclr_positions = core.get_positions("simclr_swap")[V1_LAYER]
sup_pos = core.get_positions("supervised_swap")[V1_LAYER]
random_positions = core.get_positions("retinotopic")[V1_LAYER]
```

```python
model_order = [
    "TDANN",
    "Hand-Crafted SOM",
    "DNN-SOM",
    "Supervised Functional Only",
    "SimCLR Functional Only",
    "Unoptimized",
]
```

```python
def get_tissues(base, positions, layer=V1_LAYER, seeds=None, step="latest"):
    ret = {}
    if seeds is None:
        seeds = range(5)

    for seed in seeds:
        try:
            tissue = get_sine_tissue(
                f"{base}{seed_str(seed)}", positions, step=step, layer=layer
            )
            ret[seed] = tissue
        except Exception:
            continue

    return ret
```

```python
tissues = {
    "Supervised Functional Only": get_tissues(
        "nonspatial/supervised_lw0/supervised_spatial_resnet18_swappedon_SineGrating2019_lw0",
        sup_pos,
    ),
    "SimCLR Functional Only": get_tissues(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0",
        simclr_positions,
    ),
    "TDANN": get_tissues(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
        simclr_positions,
    ),
    "Unoptimized": get_tissues(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
        random_positions,
        step="random",
    ),
}
```

### Load the neural data

```python
# restrict to the regions with reasonable SNR, as per Nauhaus et al 2012
ori_tissue = NauhausOrientationTissue()
sf_tissue = NauhausSFTissue()
color_tissue = LivingstoneColorTissue()

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
        )  # from swindale bauer scale bars
```

```python
# add SOMs to the tissue dictionary
tissues["Hand-Crafted SOM"] = {
    seed: x.tissue for seed, x in soms["Hand-Crafted SOM"].items()
}
tissues["DNN-SOM"] = {seed: x.tissue for seed, x in soms["DNN-SOM"].items()}
```

```python
# decide which models and seeds will get plotted
to_plot = {
    "TDANN": tissues["TDANN"][4],  # for consistency with main figure
    "Hand-Crafted SOM": soms["Hand-Crafted SOM"][0].tissue,
    "DNN-SOM": soms["DNN-SOM"][0].tissue,
    "Task Only": tissues["SimCLR Functional Only"][0],
    "Unoptimized": tissues["Unoptimized"][0],
}
```

```python
ncols = len(to_plot)
fig, ax_rows = plt.subplots(ncols=ncols, nrows=3, figsize=(5, 3))

# plot models
for axes, (full_name, tissue) in zip(ax_rows.T, to_plot.items()):
    tissue.reset_unit_mask()
    name = full_name.split(" - ")[0]
    if "SOM" not in name:
        tissue.set_mask_by_pct_limits([[50, 80], [50, 80]])

    for (metric_name, metric), ax in zip(METRIC_DICT.items(), axes):
        scatter_handle = tissue.make_parameter_map(
            ax,
            metric=metric,
            scale_points=True,
            final_s=3 if name != "SOM" else 1,
            marker="." if name != "SOM" else "s",
            linewidths=0,
            rasterized=True,
        )
        ax.axis("off")

        # add a colorbar if we're in the last column
        if axes[0] == ax_rows[0, -1]:
            cbar = add_sine_colorbar(fig, ax, metric, label=metric.xlabel)
            cbar.ax.tick_params(labelsize=5)
            cbar.ax.set_yticklabels(
                [metric.xticklabels[0], metric.xticklabels[-1]], rotation=90
            )
            cbar.set_label(label="", fontsize=8)

    plot_utils.add_scale_bar(axes[-1], width=2)
    plt.subplots_adjust(hspace=0.05, wspace=0.01)
    axes[0].set_title(full_name)
figure_utils.save(fig, "S03/maps")
```

<!-- #region tags=[] -->
# Circular Variance
<!-- #endregion -->

```python
def cv_bin(cv, bin_edges):
    """Bin circular variance into a histogram with specified bin edges"""
    counts, bin_edges = np.histogram(cv, bins=bin_edges)
    counts = counts / np.sum(counts) * 100
    return counts
```

```python
RESP_THRESH = 1
SEL_THRESH = 0.6
bin_edges = np.linspace(0.2, 1, 10)
midpoints = array_utils.midpoints_from_bin_edges(bin_edges)

cv_results = {"Circular Variance": [], "% Units": [], "Model": [], "Seed": []}
orisel_results = {"Model": [], "Seed": [], "% Selective": []}

for name, seeds in tissues.items():
    for seed, tissue in seeds.items():
        cv = tissue.responses.circular_variance
        mean_responses = tissue.responses._data.mean("image_idx").values
        cv = cv[~np.isnan(cv) & (mean_responses > RESP_THRESH)]

        counts = cv_bin(cv, bin_edges)
        for x, y in zip(midpoints, counts):
            cv_results["Circular Variance"].append(x)
            cv_results["% Units"].append(y)
            cv_results["Model"].append(figure_utils.get_label(name))
            cv_results["Seed"].append(seed)

        orisel_results["Model"].append(figure_utils.get_label(name))
        orisel_results["Seed"].append(seed)
        orisel_results["% Selective"].append(np.mean(cv < SEL_THRESH) * 100)
```

```python
# real data
ringach_data = load_ringach_data(fieldname="orivar")
ringach_counts = cv_bin(ringach_data, bin_edges)
for x, y in zip(midpoints, ringach_counts):
    cv_results["Circular Variance"].append(x)
    cv_results["% Units"].append(y)
    cv_results["Model"].append("Macaque V1")
    cv_results["Seed"].append(0)

macaque_orisel = np.mean(ringach_data < SEL_THRESH)
```

```python
# Set up figure to plot results
```

```python
mosaic = """AB
            CD
            """
fig = plt.figure(figsize=(3, 2))
axd = fig.subplot_mosaic(mosaic, gridspec_kw={"wspace": 0.4, "hspace": 0.7})
fontsize = 5
```

```python
model_palette = {name: figure_utils.get_color(name) for name in model_order}
model_palette["Macaque V1"] = figure_utils.get_color("Macaque V1")
model_palette["Hand-Crafted SOM"] = figure_utils.get_color("Hand-Crafted SOM")
model_palette["DNN-SOM"] = figure_utils.get_color("DNN-SOM")
```

```python
ax = axd["D"]
sns.barplot(
    data=orisel_results,
    x="Model",
    y="% Selective",
    order=model_order,
    palette=model_palette,
    ax=ax,
)
ax.axhline(
    macaque_orisel, color=figure_utils.get_color("Macaque V1"), linestyle="dashed", lw=1
)
ax.set_xticks([])
ax.set_xticklabels([])

# add text labels
labels = [
    (0, 0.02, "TDANN", "white"),
    (1, 0.02, "H.C. SOM", "black"),
    (2, 0.02, "DNN-SOM", "black"),
    (3, 0.02, "Self-Sup", "white"),
    (4, 0.02, "Categ.", "white"),
    (5, 0.02, "Unoptimized", "black"),
]

for idx, y, text, color in labels:
    ax.text(
        idx,
        y,
        text,
        rotation="vertical",
        color=color,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontdict={"weight": 700, "size": fontsize},
    )

ax.set_ylabel("% Selective")
plot_utils.remove_spines(ax)
```

```python
ax = axd["C"]
sns.lineplot(
    data=cv_results,
    x="Circular Variance",
    y="% Units",
    hue="Model",
    palette=model_palette,
    legend=False,
    ax=ax,
)
ax.axvline(SEL_THRESH, c="#DDD")
plot_utils.remove_spines(ax)
```

# Preferred Orientations

```python
# these values are ripped from Figure 5 of De Valois, Yund, Hepler '82 then passed through web plot digitizer
foveal_dvd = np.array(
    [
        [-0.004290201486248257, 66.09515054010572],
        [1.0036007048188158, 49.24691641768176],
        [1.996537194514671, 76.96284379069945],
        [2.9885543553206158, 54.085037922316715],
        [3.9957097985137513, 66.09515054010573],
    ]
)[:, 1]
normed_dvd = foveal_dvd / foveal_dvd.sum() * 100
```

```python
def card_ind(x):
    """define cardinality index as fraction of preferred orientations on the cardinals"""
    cardinality_index = (x[0] + x[2] + x[4]) / (x.sum())
    return cardinality_index
```

```python
## in this figure, 0 is horizontal
xs = [0, 45, 90, 135, 180]
pref_ori_results = {"Orientation": [], "% Units": [], "Model": [], "Seed": []}
cardinality_index_results = {"Model": [], "Seed": [], "Cardinality Index": []}

for name, seeds in tissues.items():
    for seed, tissue in seeds.items():
        tissue.reset_unit_mask()
        da = tissue.responses._data
        vals = da.groupby("angles").mean().argmax(axis=0).values
        counts = np.array(
            [
                np.sum(vals == 4),  # horizontal
                np.sum(vals == 2),  # +45 deg
                np.sum(vals == 0),  # vertical
                np.sum(vals == 6),  # -45 deg
                np.sum(vals == 4),  # horizontal
            ]
        )
        normed_counts = counts / counts.sum() * 100

        cardinality_index_results["Model"].append(figure_utils.get_label(name))
        cardinality_index_results["Seed"].append(seed)
        cardinality_index_results["Cardinality Index"].append(card_ind(counts))

        for x, y in zip(xs, normed_counts):
            pref_ori_results["Orientation"].append(x)
            pref_ori_results["% Units"].append(y)
            pref_ori_results["Model"].append(figure_utils.get_label(name))
            pref_ori_results["Seed"].append(seed)


# add macaque data
for x, y in zip(xs, normed_dvd):
    pref_ori_results["Orientation"].append(x)
    pref_ori_results["% Units"].append(y)
    pref_ori_results["Model"].append("Macaque V1")
    pref_ori_results["Seed"].append(0)

macaque_card_ind = card_ind(foveal_dvd)
```

```python
ax = axd["B"]
sns.barplot(
    data=cardinality_index_results,
    x="Model",
    y="Cardinality Index",
    ax=ax,
    order=model_order,
    palette=model_palette,
)
ax.set_xticks([])
ax.set_xticklabels([])
ax.axhline(
    macaque_card_ind,
    color=figure_utils.get_color("Macaque V1"),
    lw=1,
    linestyle="dashed",
)
plot_utils.remove_spines(ax)

# add labels
labels = [
    (0, 0.05, "TDANN", "white"),
    (1, 0.05, "H.C. SOM", "black"),
    (2, 0.05, "DNN-SOM", "black"),
    (3, 0.05, "Self-Sup", "white"),
    (4, 0.05, "Categ.", "white"),
    (5, 0.05, "Unoptimized", "black"),
]

for idx, y, text, color in labels:
    ax.text(
        idx,
        y,
        text,
        rotation="vertical",
        color=color,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontdict={"weight": 700, "size": fontsize},
    )

ax.set_ylabel("Cardinality Index")
ax.set_xlabel("")
```

```python
# establish new plotting order
order = model_order[::-1] + ["Macaque V1"]
```

```python
ax = axd["A"]
sns.lineplot(
    data=pref_ori_results,
    x="Orientation",
    y="% Units",
    hue="Model",
    palette=model_palette,
    hue_order=order,
    ax=ax,
)
ax.set_xlabel("Preferred Orientation")
ax.set_ylabel("% Units")
ax.legend().remove()
plot_utils.remove_spines(ax)
```

```python
figure_utils.save(fig, "S03/quant")
fig
```

```python

```
