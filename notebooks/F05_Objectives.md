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
import pprint
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pingouin as pg
import seaborn as sns
from tqdm import tqdm
```

```python
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    spatial_utils,
    array_utils,
    seed_str,
)
import spacetorch.analyses.core as core
from spacetorch.analyses.sine_gratings import (
    get_sine_tissue,
    METRIC_DICT,
)
from spacetorch.analyses.floc import get_floc_tissue
from spacetorch.datasets import floc, DatasetRegistry
from spacetorch.maps.pinwheel_detector import PinwheelDetector
from spacetorch.maps.screenshot_maps import NauhausOrientationTissue
from spacetorch.maps import nsd_floc

from spacetorch.paths import PROJ_DIR, _base_fs
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}
MARKERSIZE = 6
pprint.pprint(contrasts)
```

```python
V1_LAYER, VTC_LAYER = "layer2.0", "layer4.1"
simclr_positions = core.get_positions("simclr_swap")
supervised_positions = core.get_positions("supervised_swap")
random_positions = core.get_positions("retinotopic")
```

```python
tissues = {
    "V1": {
        "TDANN": {
            seed: get_sine_tissue(
                f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}",
                simclr_positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in range(5)
        },
        "Categorization": {
            seed: get_sine_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019{seed_str(seed)}",
                supervised_positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in range(5)
        },
        "Absolute SL": {
            seed: get_sine_tissue(
                f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl{seed_str(seed)}",
                simclr_positions[V1_LAYER],
                layer=V1_LAYER,
            )
            for seed in range(5)
        },
        "Unoptimized": {
            seed: get_sine_tissue(
                f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}",
                random_positions[V1_LAYER],
                layer=V1_LAYER,
                step="random",
            )
            for seed in range(5)
        },
    },
    "VTC": {
        "TDANN": {
            seed: get_floc_tissue(
                f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}",
                simclr_positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in range(5)
        },
        "Categorization": {
            seed: get_floc_tissue(
                f"supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019{seed_str(seed)}",
                supervised_positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in range(5)
        },
        "Absolute SL": {
            seed: get_floc_tissue(
                f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl{seed_str(seed)}",
                simclr_positions[VTC_LAYER],
                layer=VTC_LAYER,
            )
            for seed in range(5)
        },
        "Unoptimized": {
            seed: get_floc_tissue(
                f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}",
                random_positions[VTC_LAYER],
                layer=VTC_LAYER,
                step="random",
            )
            for seed in range(5)
        },
    },
}
```

```python
mat_dir = PROJ_DIR / "nsd_tvals"
subjects = nsd_floc.load_data(mat_dir, domains=contrasts, find_patches=True)
```

```python
model_order = ["TDANN", "Categorization", "Absolute SL"]
```

```python
# choose the order in which dots will be added to the scatter plot
plot_con_order = ["Objects", "Characters", "Places", "Faces", "Bodies"]
contrasts = [contrast_dict[n] for n in plot_con_order]
```

```python
fig, axes = plt.subplots(figsize=(3 * 1.25, 1.25), ncols=3)
plt.subplots_adjust(wspace=0.01)
for ax, name in zip(axes, model_order):
    seeds = tissues["VTC"][name]
    ax.axis("off")

    tissue = seeds[1]
    tissue.make_selectivity_map(
        ax,
        marker=".",
        contrasts=contrasts,
        size_mult=3e-3,
        foreground_alpha=0.8,
        selectivity_threshold=12,
        rasterized=True,
        linewidths=0,
    )
    plot_utils.add_scale_bar(ax, 10)
figure_utils.save(fig, "F05/vtc_maps")
```

```python
vtc_smoothness_curves = {
    "Distance": [],
    "Sel. Diff.": [],
    "Model": [],
    "Seed": [],
    "Iteration": [],
}
vtc_smoothness = {"Model": [], "Smoothness": [], "Seed": []}
```

```python
contrast = contrast_dict["Faces"]
```

```python
def trim(curve):
    """Helper function to remove the last few entries of the curve if they're NaN"""
    nans = np.where(np.isnan(curve))[0]
    if len(nans) > 0:
        curve = curve[: nans[0]]

    return curve


num_samples = 25

# the distance curve will have N_BINS - 1 entries
N_BINS = 10

# for the human data, this is the farthest distance we'll consider
distance_cutoff = 60.0  # mm

bin_edges = np.linspace(0, distance_cutoff, N_BINS)
midpoints = array_utils.midpoints_from_bin_edges(bin_edges)
```

```python
# model smoothness

for name, seeds in tissues["VTC"].items():
    for seed, tissue in seeds.items():
        try:
            distances, curves = tissue.category_smoothness(
                contrast=contrast,
                num_samples=25,
                bin_edges=bin_edges,
            )
            _, chance_curves = tissue.category_smoothness(
                contrast=contrast, num_samples=25, bin_edges=bin_edges, shuffle=True
            )
            chance_mean = np.nanmean(chance_curves)
        except ValueError:
            continue

        smoos = [spatial_utils.smoothness(curve) for curve in curves]
        mn_smoo = np.nanmean(smoos)

        vtc_smoothness["Model"].append(name)
        vtc_smoothness["Smoothness"].append(mn_smoo)
        vtc_smoothness["Seed"].append(seed)

        for curve_idx, curve in enumerate(curves):
            curve = curve / chance_mean
            for x, y in zip(distances, curve):
                vtc_smoothness_curves["Distance"].append(x)
                vtc_smoothness_curves["Sel. Diff."].append(y)
                vtc_smoothness_curves["Model"].append(figure_utils.get_label(name))
                vtc_smoothness_curves["Seed"].append(seed)
                vtc_smoothness_curves["Iteration"].append(curve_idx)
```

### Add human data

```python
for subj in subjects:
    for hemi in ["left_hemi", "right_hemi"]:
        distances, curves = subj.smoothness(
            contrast,
            hemi,
            bin_edges,
            distance_cutoff=distance_cutoff,
            num_samples=num_samples,
        )
        _, chance_curves = subj.smoothness(
            contrast,
            hemi,
            bin_edges,
            distance_cutoff=distance_cutoff,
            num_samples=num_samples,
            shuffle=True,
        )

        # trim curves to remove nans at the end
        curves = [trim(curve) for curve in curves]

        # compute the value expected by chance
        chance_mean = np.nanmean(np.concatenate(chance_curves))

        # compute smoothness from raw curves
        smoo = [spatial_utils.smoothness(curve) for curve in curves]
        vtc_smoothness["Model"].append("Human")
        vtc_smoothness["Smoothness"].append(np.mean(smoo))
        vtc_smoothness["Seed"].append(f"{subj.name}_{hemi}")

        for curve_idx, curve in enumerate(curves):
            curve = curve / chance_mean
            for x, y in zip(distances, curve):
                vtc_smoothness_curves["Distance"].append(x)
                vtc_smoothness_curves["Sel. Diff."].append(y)
                vtc_smoothness_curves["Model"].append("Human")
                vtc_smoothness_curves["Seed"].append(f"{subj.name}_{hemi}")
                vtc_smoothness_curves["Iteration"].append(curve_idx)
```

```python
vtc_smoothness_curves_df = pd.DataFrame(vtc_smoothness_curves)
```

```python
model_palette = {model: figure_utils.get_color(model) for model in tissues["VTC"]}
model_palette["Human"] = figure_utils.get_color("Macaque V1")
model_palette["Macaque V1"] = figure_utils.get_color("Macaque V1")
print(model_palette)
```

# Number of Patches...

```python
for name, seeds in tissues["VTC"].items():
    print(name)
    for seed, tissue in tqdm(seeds.items()):
        tissue.patches = []
        for contrast in contrasts:
            tissue.find_patches(contrast, verbose=False)
```

```python
patch_count_results = {
    "Species": [],
    "Contrast": [],
    "Hemisphere": [],
    "Count": [],
    "Seed": [],
}

for subject in subjects:
    hemis = {"Left Hemi": subject.lh_patches, "Right Hemi": subject.rh_patches}
    for hemisphere_name, hemi_patches in hemis.items():
        for contrast in contrasts:
            matching_patches = list(
                filter(lambda p: p.contrast == contrast, hemi_patches)
            )
            patch_count_results["Species"].append("Human")
            patch_count_results["Contrast"].append(contrast.name)
            patch_count_results["Hemisphere"].append(hemisphere_name)
            patch_count_results["Count"].append(len(matching_patches))
            patch_count_results["Seed"].append(0)

# add the model data
for name, seeds in tissues["VTC"].items():
    for seed, tissue in seeds.items():
        for contrast in contrasts:
            matching_patches = list(
                filter(lambda p: p.contrast == contrast, tissue.patches)
            )
            patch_count_results["Species"].append(figure_utils.get_label(name))
            patch_count_results["Contrast"].append(contrast.name)
            patch_count_results["Hemisphere"].append("cyclops")
            patch_count_results["Count"].append(len(matching_patches))
            patch_count_results["Seed"].append(seed)

patch_count_df = pd.DataFrame(patch_count_results)
```

```python
fig, ax = plt.subplots(figsize=(1, 1))

# plot all the non-human data
data_to_plot = patch_count_df[
    (patch_count_df.Species != "Human") & (patch_count_df.Species != "Unoptimized")
]

sns.barplot(
    data=data_to_plot,
    x="Species",
    y="Count",
    order=model_order,
    palette=model_palette,
    ax=ax,
)

# add individual dots
indiv = data_to_plot.groupby(["Species", "Seed"]).mean(numeric_only=True).reset_index()
sns.stripplot(
    data=indiv,
    x="Species",
    order=model_order,
    y="Count",
    color='#ccc',
    edgecolor='k',
    linewidth=.5,
    jitter=.1,
    size=2,
    ax=ax,
)


# plot the human data
ax.axhline(
    patch_count_df.query("Species == 'Human'").Count.mean(),
    linestyle="dashed",
    c=figure_utils.get_color("Human"),
    lw=1,
)

plot_utils.remove_spines(ax)
ax.set_xlabel("")
ax.set_ylabel("Patch Count")
ax.set_xticks([])
ax.legend(title="", frameon=False)

ax.set_ylabel("# Patches")
ax.set_xlabel("")
figure_utils.save(fig, "F05/patch_counts")
```

```python
patch_count_df.groupby("Species").mean(numeric_only=True)
```

```python
display(
    pg.mwu(
        patch_count_df[patch_count_df.Species == "TDANN"]
        .groupby("Seed")
        .mean()["Count"],
        patch_count_df[patch_count_df.Species == "Categorization"]
        .groupby("Seed")
        .mean()["Count"],
    )
)
display(
    pg.mwu(
        patch_count_df[patch_count_df.Species == "TDANN"]
        .groupby("Seed")
        .mean()["Count"],
        patch_count_df[patch_count_df.Species == "Absolute SL"]
        .groupby("Seed")
        .mean()["Count"],
    )
)
```

## V1 Maps and Smoothness

```python
pindets = []
for name in model_order:
    tissue = tissues["V1"][name][3]
    tissue.set_mask_by_pct_limits([[30, 80], [30, 80]])
    pindet = PinwheelDetector(tissue)
    pindets.append(pindet)
```

```python
fig, axes = plt.subplots(figsize=(3.3, 1), ncols=3, gridspec_kw={"wspace": 0.5})
metric = METRIC_DICT["angles"]

handles = []
for ax, name, pindet in zip(axes, model_order, pindets):
    pos, neg = pindet.count_pinwheels()
    smoothed = pindet.smoothed

    pos_centers, neg_centers = pindet.centers
    for x, y in pos_centers:
        ax.scatter(x, y, c="k", s=1)

    for x, y in neg_centers:
        ax.scatter(x, y, c="w", s=1)

    image_handle = ax.imshow(smoothed, cmap=metric.colormap, interpolation="nearest")
    handles.append(image_handle)
    ax.axis("off")

    # scale bar gymnastics
    mm_expanse = np.ptp(tissue.positions[:, 0])
    px_expanse = smoothed.shape[0]
    px_per_mm = px_expanse / mm_expanse
    plot_utils.add_scale_bar(ax, width=2 * px_per_mm, flipud=True, height=1)


cax = fig.add_axes([0.33, 0.2, 0.02, 0.6])
cbar = plt.colorbar(
    handles[0], cax=cax, extend="both", extendrect=True, ticks=[0, 90, 180]
)
cbar.ax.tick_params(rotation=90)
cbar.set_ticks([0, 90, 180])
figure_utils.save(fig, "F05/v1_maps")
```

```python
hypercolumn_widths = {
    "TDANN": 3.5,
    "Categorization": 3.5,
    "Absolute SL": 3.5,
    "Macaque V1": 0.75,
}
```

```python

ori_tissue = NauhausOrientationTissue()
```

```python
curve_tissues = {
    "Macaque V1": {0: ori_tissue},
    "TDANN": tissues["V1"]["TDANN"],
    "Absolute SL": tissues["V1"]["Absolute SL"],
    "Categorization": tissues["V1"]["Categorization"],
}
```

```python
def get_curves(
    name: str,
    seed: int,
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
        tissue = curve_tissues["Macaque V1"][0]
        _, curves = tissue.difference_over_distance(**common)
    else:
        tissue = curve_tissues[name][seed]
        tissue.reset_unit_mask()
        tissue.set_unit_mask_by_ptp_percentile("angles", 75)
        _, curves = tissue.metric_difference_over_distance(
            distance_cutoff=max_dist, **common
        )

    # normalize midpoints to be a fraction of the hypercolumn width
    return midpoints / hcw, curves
```

```python
v1_smoothness = {"Smoothness": [], "Model": [], "Seed": []}

for name, seeds in tqdm(curve_tissues.items()):
    for seed, tissue in seeds.items():
        distances, curves = get_curves(name, seed)

        # compute smoothness for each curve
        smoos = [spatial_utils.smoothness(curve) for curve in curves]
        mean_smoothness = np.nanmean(smoos)

        v1_smoothness["Smoothness"].append(mean_smoothness)
        v1_smoothness["Model"].append(figure_utils.get_label(name))
        v1_smoothness["Seed"].append(seed)
```

```python
common = {
    "hue": "Model",
    "palette": model_palette,
}
```

```python
# plot smoothness
v1_sm_df = pd.DataFrame(v1_smoothness)
vtc_sm_df = pd.DataFrame(vtc_smoothness)
```

```python
common = {
    "hue": "Model",
    "palette": model_palette,
    "x": "Model",
    "y": "Smoothness",
    "order": ["TDANN", "Categorization", "Absolute SL"],
    "dodge": False,
}
```

```python
fig, v1_ax = plt.subplots(figsize=(1, 1))

# split human into horizontal bar
data_to_plot =v1_sm_df[
    (v1_sm_df["Model"] != "Unoptimized") & (v1_sm_df["Model"] != "Macaque V1")
]

sns.barplot(
    data=data_to_plot,
    ax=v1_ax,
    **common,
)

# add individual dots
sns.stripplot(
    data=data_to_plot,
    x="Model",
    order=model_order,
    y="Smoothness",
    color='#ccc',
    edgecolor='k',
    linewidth=.5,
    jitter=.1,
    size=2,
    ax=v1_ax,
)
v1_ax.axhline(
    v1_sm_df[v1_sm_df["Model"] == "Macaque V1"].Smoothness.mean(),
    ls="dashed",
    lw=1,
    color=figure_utils.get_color("Macaque V1"),
)


v1_ax.legend().remove()
v1_ax.set_xticks([])
v1_ax.set_ylabel("OPM Smoothness")
v1_ax.set_xlabel("")
plot_utils.remove_spines(v1_ax)
figure_utils.save(fig, "F05/v1_smoothness_bars")
```

```python
fig, vtc_ax = plt.subplots(figsize=(1, 1))

data_to_plot = vtc_sm_df[
    (vtc_sm_df["Model"] != "Unoptimized") & (vtc_sm_df["Model"] != "Human")
]

sns.barplot(
    data=data_to_plot,
    ax=vtc_ax,
    **common,
)
sns.stripplot(
    data=data_to_plot,
    x="Model",
    order=model_order,
    y="Smoothness",
    color='#ccc',
    edgecolor='k',
    linewidth=.5,
    jitter=.1,
    size=2,
    ax=vtc_ax,
)
vtc_ax.axhline(
    vtc_sm_df[vtc_sm_df["Model"] == "Human"].Smoothness.mean(),
    ls="dashed",
    lw=1,
    color=figure_utils.get_color("Human"),
)

vtc_ax.legend().remove()
vtc_ax.set_xticks([])
vtc_ax.set_ylabel("Face Selectivity\n Map Smoothness")
vtc_ax.set_xlabel("")
plot_utils.remove_spines(vtc_ax)
figure_utils.save(fig, "F05/face_selectivity_smoothness_bars")
```

```python
v1_sm_df.groupby("Model").mean()
```

```python
display(
    pg.mwu(
        v1_sm_df.query("Model == 'TDANN'")["Smoothness"],
        v1_sm_df.query("Model == 'Categorization'")["Smoothness"],
    )
)
display(
    pg.mwu(
        v1_sm_df.query("Model == 'TDANN'")["Smoothness"],
        v1_sm_df.query("Model == 'Absolute SL'")["Smoothness"],
    )
)
```

```python
vtc_sm_df.groupby("Model").mean()
```

```python
display(
    pg.mwu(
        vtc_sm_df.query("Model == 'TDANN'")["Smoothness"],
        vtc_sm_df.query("Model == 'Categorization'")["Smoothness"],
    )
)

display(
    pg.mwu(
        vtc_sm_df.query("Model == 'TDANN'")["Smoothness"],
        vtc_sm_df.query("Model == 'Absolute SL'")["Smoothness"],
    )
)
```

# Pinwheel density

```python
hypercol_width = 3.5
```

```python
pinwheel_results = {"Density": [], "Model": [], "Seed": []}

for name, seeds in tissues["V1"].items():
    for seed, tissue in tqdm(seeds.items(), desc=name):
        edge_size = np.ptp(tissue._positions)
        num_hcol = edge_size / hypercol_width
        tissue.reset_unit_mask()
        pindet = PinwheelDetector(tissue)
        pos, neg = pindet.count_pinwheels()
        total = pos + neg

        density = total / (num_hcol**2)

        pinwheel_results["Model"].append(name)
        pinwheel_results["Seed"].append(seed)
        pinwheel_results["Density"].append(density)

pinwheel_df = pd.DataFrame(pinwheel_results)
```

```python
pinwheel_df.groupby("Model").mean()
```

```python
display(
    pg.mwu(
        pinwheel_df.query("Model == 'TDANN'")["Density"],
        pinwheel_df.query("Model == 'Categorization'")["Density"],
    )
)
display(
    pg.mwu(
        pinwheel_df.query("Model == 'TDANN'")["Density"],
        pinwheel_df.query("Model == 'Absolute SL'")["Density"],
    )
)
```

```python
fig, ax = plt.subplots(figsize=(1, 1))
data_to_plot = pinwheel_df[pinwheel_df.Model != "Unoptimized"]
sns.barplot(
    data=data_to_plot,
    x="Model",
    y="Density",
    palette=model_palette,
    ax=ax,
)
sns.stripplot(
    data=data_to_plot,
    x="Model",
    y="Density",
    color='#ccc',
    edgecolor='k',
    linewidth=.5,
    jitter=.1,
    size=2,
    ax=ax,
)
plot_utils.remove_spines(ax)
ax.set_xlabel("")
ax.set_xticks([])
ax.legend(title="", frameon=False)
ax.set_ylabel(("Pinwheels /" "\n" r"Column Spacing$^2$"))
ax.axhline(np.pi, color=figure_utils.get_color("Macaque V1"), linestyle="dashed", lw=1)
figure_utils.save(fig, "F05/pinwheel_counts")
```

# Schematic 

```python
# get example ImageNet images
dataset = DatasetRegistry.get("ImageNet")
```

```python
image_indices = [4500, 10000]
for idx in image_indices:
    sample = dataset[idx][0]
    sample = array_utils.norm(sample.detach().cpu().numpy())
    sample = np.transpose(sample, (1, 2, 0))

    fig, ax = plt.subplots(figsize=(0.25, 0.25))
    ax.imshow(sample)
    ax.axis("off")
    figure_utils.save(fig, f"F05/example_{idx}")
```

# One-to-one corrs

```python
# from Dawn Finzi
rh_subj_vals = [
    0.43202289,
    0.42191846,
    0.40613521,
    0.44393879,
    0.4246062,
    0.38117872,
    0.41842483,
    0.35186099,
]

lh_subj_vals = [
    0.35160881,
    0.39486351,
    0.35669901,
    0.38471974,
    0.41227402,
    0.30587512,
    0.36741204,
    0.29522951,
]

subj_vals = np.stack([lh_subj_vals, rh_subj_vals]).mean(axis=0)
```

```python
dfs = []
for hemi in ["lh", "rh"]:
    for seed in range(5):
        pkl_path = Path(
            f"{_base_fs}/projects/Dawn/NSD/"
            f"results/spacetorch/for_Eshed_ventral_only_{hemi}_corrected_means_seed{seed}.pkl"
        )

        with pkl_path.open("rb") as stream:
            data = pickle.load(stream)

        data["seed"] = [seed] * len(data)
        data["hemi"] = [hemi] * len(data)
        dfs.append(data)

data = pd.concat(dfs)
```

```python
data["version"].replace("self-supervised", "TDANN", inplace=True)
data["version"].replace("supervised", "Categorization", inplace=True)
data["version"].replace("old_scl", "Absolute SL", inplace=True)
```

```python
selfsup_lw0 = data[(data.spatial_weight == 0) & (data.version == "TDANN")]
```

```python
# remove incorrect sw0 results
data = data.drop(
    data[(data.spatial_weight == 0) & (data.version == "Absolute SL")].index
)

abs_lw0 = selfsup_lw0.replace("TDANN", "Absolute SL")
```

```python
full_df = pd.concat([data, abs_lw0], ignore_index=True)
```

```python
palette = {
    "TDANN": figure_utils.get_color("TDANN"),
    "Categorization": figure_utils.get_color("Categorization"),
    "Absolute SL": figure_utils.get_color("Absolute SL"),
}
```

```python
fig, ax = plt.subplots(figsize=(1, 1))
sns.lineplot(
    data=full_df,
    x="spatial_weight",
    y="correlation",
    hue="version",
    marker=".",
    markersize=MARKERSIZE,
    palette=palette,
    lw=1,
    ax=ax,
)

ax.axhline(np.mean(subj_vals), color=figure_utils.get_color("Human"), ls="dashed", lw=1)
ax.axvline(0.25, c="k", alpha=0.1)

ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 50])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("1-to1 Unit-to-Voxel\nCorrelation")
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_ylim([0, 0.5])
ax.legend().remove()

plot_utils.remove_spines(ax)
figure_utils.save(fig, "F05/sub2sub")
```

```python
# stats
```

```python
data.groupby(["spatial_weight", "version"]).mean()
```

```python
data.anova(dv="correlation", between=["version", "spatial_weight"])
```

```python
data.pairwise_tukey(dv="correlation", between="spatial_weight")
```
