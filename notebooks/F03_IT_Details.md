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

# Notebook prep


## imports

```python
import copy
import random
```

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from numpy.random import default_rng
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm
```

```python
import spacetorch.analyses.core as core
import spacetorch.analyses.rsa as rsa
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    spatial_utils,
    array_utils,
    seed_str,
)
from spacetorch.analyses.floc import get_floc_tissue
from spacetorch.datasets import floc
from spacetorch.maps import nsd_floc
from spacetorch.maps.screenshot_maps import ITNScreenshotMaps
from spacetorch.maps.it_map import ITMap
from spacetorch.paths import PROJ_DIR, RESULTS_DIR
from spacetorch.constants import RNG_SEED

from spacetorch.utils.generic_utils import load_pickle
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_dict = {c.name: c for c in contrasts}
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
ordered_contrasts = [contrast_dict[curr] for curr in contrast_order]
contrast_colors = {c.name: c.color for c in contrasts}

marker_fill = "#ccc"
bar_fill = "#ccc"
rng = default_rng(seed=RNG_SEED)
```

```python
marker_lookup = {
    "TDANN": None,
    "DNN-SOM": "P",
    "Unoptimized": "X",
    "Functional Only": "D",
    "Task Only": "D",
    "Human": None,
    "ITN": "o",
}

color_lookup = {
    "TDANN": figure_utils.get_color("TDANN"),
    "DNN-SOM": bar_fill,
    "Human": figure_utils.get_color("Human"),
    "Functional Only": bar_fill,
    "Task Only": bar_fill,
    "ITN": bar_fill,
    "Unoptimized": figure_utils.get_color("Unoptimized"),
}

mkw = {"color": marker_fill, "edgecolors": "k", "linewidths": 0.5}
```

## Load human data

```python
mat_dir = PROJ_DIR / "nsd_tvals"
subjects = nsd_floc.load_data(mat_dir, domains=contrasts, find_patches=True)
```

## Load other models

```python
itn = ITNScreenshotMaps.from_png("assets/pnas.2112566119.sapp.p20.C.png")

# find patches
itn.patches = []
for contrast in contrasts:
    itn.find_patches(contrast=contrast)
```

```python
# SOM
load_dir = RESULTS_DIR / "SOM"
load_paths = [f for f in load_dir.glob("*.pkl") if "alexnet" in f.name]
soms = {}
for lp in load_paths:
    seed = int(lp.stem.split("seed_")[-1])
    som = load_pickle(lp)

    positions = (
        np.stack(som.get_euclidean_coordinates(), axis=-1).reshape((-1, 2))
        / 200.0
        * 70.0
    )
    tissue = ITMap(positions, som.floc_responses)
    tissue.cache_id = f"som_seed_{seed}"
    soms[seed] = tissue


for seed, tissue in tqdm(soms.items()):
    tissue.patches = []
    for contrast in contrasts:
        tissue.find_patches(contrast, maximum_size=4500, threshold=10)
```

```python
VTC_LAYER = "layer4.1"
positions = core.get_positions("simclr_swap")[VTC_LAYER]
random_positions = core.get_positions("retinotopic")[VTC_LAYER]
```

```python
tissues = {
    "SW_0": {
        seed: get_floc_tissue(
            f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0{seed_str(seed)}",
            positions,
            layer=VTC_LAYER,
        )
        for seed in range(5)
    },
    "SW_0pt25": {
        seed: get_floc_tissue(
            f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}",
            positions,
            layer=VTC_LAYER,
        )
        for seed in range(5)
    },
    "Unoptimized": {
        seed: get_floc_tissue(
            f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{seed_str(seed)}",
            random_positions,
            layer=VTC_LAYER,
            step="random",
        )
        for seed in range(5)
    },
    "DNN-SOM": soms,
    "ITN": {0: itn},
}
```

```python
# define the colors we're going to be using
palette = {name: figure_utils.get_color(name) for name in tissues}
palette["Human"] = figure_utils.get_color("Human")
palette["TDANN"] = figure_utils.get_color("SW_0pt25")
palette["Functional Only"] = figure_utils.get_color("SW_0")
```

```python
# find patches in all models
for name, seeds in tissues.items():
    if name in ["ITN", "DNN-SOM"]:  # we already found these
        continue
    for seed, tissue in tqdm(seeds.items()):
        tissue.patches = []
        for contrast in contrasts:
            tissue.find_patches(contrast)
```

# Result 1: RSA

```python
human_rsms = []

for subject in subjects:
    for hemi in ["left_hemi", "right_hemi"]:
        human_rsms.append(rsa.get_human_rsm(subject, hemi))
```

```python
# model RSMs
model_rsms = {}

for name, seeds in tqdm(tissues.items()):
    model_rsms[name] = []
    for seed, tissue in seeds.items():
        rsm = rsa.get_model_rsm(tissue, is_itn=(name == "ITN"))
        model_rsms[name].append(rsm)
```

```python
rsa_results = {"Name": [], "Sim": [], "Seed": []}

for name, rsms in model_rsms.items():
    for seed, rsm_a in enumerate(rsms):
        sims = []
        for rsm_b in human_rsms:
            s = rsa.rsm_similarity(rsm_a, rsm_b)

            rsa_results["Name"].append(figure_utils.get_label(name))
            rsa_results["Sim"].append(s)
            rsa_results["Seed"].append(seed)
```

```python
human2human = []
for i, rsm_a in enumerate(human_rsms):
    for j, rsm_b in enumerate(human_rsms):
        if i == j:
            continue
        human2human.append(rsa.rsm_similarity(rsm_a, rsm_b))
lower, upper = pg.compute_bootci(human2human, func="mean")
```

```python
print(lower, upper)
```

```python
rsa_df = pd.DataFrame(rsa_results)
```

```python
for name in rsa_df.Name.unique():
    sims = rsa_df[rsa_df.Name == name].Sim
    lower, upper = pg.compute_bootci(sims, func="mean", decimals=3)
    print(f"{name}: [{lower}, {upper}]")
```

```python
rsa_df.anova(dv="Sim", between="Name")
```

```python
pg.pairwise_tukey(data=rsa_df, dv="Sim", between="Name")
```

```python
fig, ax = plt.subplots(figsize=(1.25, 1))
sns.barplot(
    data=rsa_results,
    x="Name",
    y="Sim",
    palette=color_lookup,
    order=["TDANN", "DNN-SOM", "ITN", "Functional Only", "Unoptimized"],
)
ax.axhline(
    np.mean(human2human),
    ls="dashed",
    lw=1,
    color=figure_utils.get_color("Human"),
)

ax.set_xticks([])
ax.set_xlabel("")
ax.set_ylabel(("Similarity to Human" "\n" r"VTC [Kendall's $\tau$]"))
plot_utils.remove_spines(ax)
figure_utils.save(fig, "F03/rsm_bar")
```

```python
groups = copy.deepcopy(model_rsms)
groups["Human"] = human_rsms
```

```python
fig, axes = plt.subplots(figsize=(2.3, 1), ncols=2, gridspec_kw={"wspace": 0.5})
ax_dict = {"TDANN": axes[0], "Human": axes[1]}

for group_name, group_rsms in groups.items():
    label = figure_utils.get_label(group_name)
    ax = ax_dict.get(label)
    if ax is None:
        continue

    mappable = ax.imshow(
        np.mean(np.stack(group_rsms), axis=0), cmap="seismic", vmin=-1, vmax=1
    )

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels([""] * 5)
    ax.set_yticklabels([""] * 5)
    ax.set_title(figure_utils.get_label(label))

    if group_name == "Human":
        cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cb = fig.colorbar(mappable, cax=cax, label="Correlation", ticks=[-1, 1])
        cb.set_label("Correlation", labelpad=-5)

figure_utils.save(fig, "F03/rsm")
```

# Panels a-b: responses and maps

```python
# this is the one we'll use in the figure
vtc_tissue = tissues["SW_0pt25"][0]
```

```python
fig, unit_row = plt.subplots(figsize=(4, 0.2), ncols=5, gridspec_kw={"wspace": 0.5})

most_sel_indices = {}
for ax, contrast_name in zip(unit_row, contrast_order):
    contrast = contrast_dict[contrast_name]
    sel = vtc_tissue.responses.selectivity(
        on_categories=contrast.on_categories, off_categories=contrast.off_categories
    )
    unit = np.argsort(sel)[-5]
    unit_resp = vtc_tissue.responses._data[:, unit]

    resp_df = {"Category": [], "Response": []}
    for con in contrasts:
        on_indices = vtc_tissue.responses._data.categories.isin(con.on_categories)
        responses = vtc_tissue.responses._data[on_indices, unit]
        responses = rng.choice(responses, size=(50,), replace=False)

        for val in responses:
            resp_df["Category"].append(con.name)
            resp_df["Response"].append(float(val))

    sns.barplot(
        data=resp_df,
        x="Category",
        y="Response",
        palette=contrast_colors,
        ax=ax,
        errorbar=None,
        order=contrast_order,
        alpha=0.8,
    )
    sns.stripplot(
        data=resp_df,
        x="Category",
        y="Response",
        hue="Category",
        palette=contrast_colors,
        ax=ax,
        order=contrast_order,
        size=0.5,
        alpha=0.8,
        jitter=0.15,
        rasterized=True,
    )
    mx = max(resp_df["Response"])

    if contrast_name == contrast_order[0]:
        ax.set_ylabel("Activation\n(a.u.)")
    else:
        ax.set_ylabel("")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["F", "B", "C", "P", "O"])
    ax.set_xlabel("")
    ax.legend().remove()
    plot_utils.remove_spines(ax)
    most_sel_indices[contrast.name] = unit
figure_utils.save(fig, "F03/c_activations")
```

```python
panel_a_fig, ic_row = plt.subplots(figsize=(4, 0.7), ncols=5)
plot_utils.remove_spines([ax for ax in ic_row], to_remove="all")

for contrast_name, ax in zip(contrast_order, ic_row):
    contrast = contrast_dict[contrast_name]
    handle = vtc_tissue.make_single_contrast_map(
        ax, contrast, final_psm=1e-4, rasterized=True, vmin=-20, vmax=20, linewidths=0
    )
    most_sel = most_sel_indices[contrast_name]
    ax.scatter(
        vtc_tissue.positions[most_sel, 0],
        vtc_tissue.positions[most_sel, 1],
        s=30,
        marker="*",
        linewidths=0,
        c="black",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylim([0, 70])
    ax.set_xlim([0, 70])
    plot_utils.add_scale_bar(ax, 10)

# add a colorbar to the last axis
cax = panel_a_fig.add_axes([0.92, 0.35, 0.01, 0.35])
cb = plt.colorbar(handle, cax=cax, ticks=[-20, 0, 20])
cax.set_xlabel(r"$t$-value", x=2)
figure_utils.save(panel_a_fig, "F03/ab_resp_single_cond_maps")
```

# Panel e: patches

```python
to_plot = {
    "SW_0pt25": tissues["SW_0pt25"][0],
    "SW_0": tissues["SW_0"][1],
    "DNN-SOM": soms[0],
    "ITN": itn,
}
```

```python
%%capture
mosaic = """AAABBB
            AAABBB
            AAABBB
            CCDDEE
            CCDDEE
            """
fig = plt.figure(figsize=(2, 1.8))
axd = fig.subplot_mosaic(mosaic, gridspec_kw={"wspace": 0.4, "hspace": 1.1})
grid_mapping = {"SW_0pt25": "B", "SW_0": "C", "DNN-SOM": "D", "ITN": "E"}
```

```python
# plot the human data
human_ax = axd["A"]

# the fourth subject is a good example
subject = subjects[3]

# show the underlying cortical sheet as a faint gray
mask = ~np.isnan(subject.data["faces"].right_hemi)
mm_per_px = subject.xform_info.right_hemi.mm_per_px
lims = mm_per_px * np.array(np.shape(mask))
extent = [0, lims[0], lims[1], 0]
human_ax.imshow(mask.T, alpha=0.2, cmap="gray_r", extent=extent)

# add any patches
for contrast in contrasts:
    matchy_patchy = [
        patch for patch in subject.rh_patches if patch.contrast.name == contrast.name
    ]
    for patch in matchy_patchy:
        human_ax.add_patch(patch.to_mpl_poly(alpha=0.8, lw=0.5))

human_ax_lim = human_ax.get_ylim()
human_ax.set_ylim([human_ax_lim[0], human_ax_lim[0] / 2 + human_ax_lim[1]])
human_ax.axis("off")

# add a scale bar
scale_ratio = subject.data["faces"].right_hemi.shape[0] / human_ax.get_xlim()[-1]
plot_utils.add_scale_bar(
    human_ax, 10 * subject.xform_info.right_hemi.px_per_mm / scale_ratio
)
```

```python
# add patches for each model

for name, model in to_plot.items():
    ax = axd[grid_mapping[name]]

    # add a gray rectangle as a backgrop
    ax.add_patch(Rectangle((0, 0), height=70, width=70, facecolor="#ddd"))

    # add all patches
    for patch in model.patches:
        ax.add_patch(patch.to_mpl_poly(alpha=0.8, lw=0.5))

    # show the full 10 x 10 mm
    ax.set_xlim([0, 70])
    ax.set_ylim([0, 70])
    ax.axis("off")

    # flip ITN patches so they look like the ones in the paper
    if name == "ITN":
        ax.set_ylim([70, 0])
        ax.set_xlim([0, 70])

    # add a scale bar
    plot_utils.add_scale_bar(ax, 10, y_start=0)
```

```python
figure_utils.save(fig, "F03/e_patches")
fig
```

# Panels c-d: smoothness

```python
# randomly sample sets of units 25 times
num_samples = 25

# the distance curve will have N_BINS - 1 entries
N_BINS = 10

# for the human data, this is the farthest distance we'll consider
distance_cutoff = 60.0  # mm

human_bin_edges = np.linspace(0, distance_cutoff, N_BINS)
human_midpoints = array_utils.midpoints_from_bin_edges(human_bin_edges)
```

```python
def trim(curve):
    """Helper function to remove the last few entries of the curve if they're NaN"""
    nans = np.where(np.isnan(curve))[0]
    if len(nans) > 0:
        curve = curve[: nans[0]]

    return curve
```

```python
smoothness_results = {"Model": [], "Seed": [], "Smoothness": [], "Contrast": []}

curve_dict = {}
for contrast in tqdm(contrasts):
    curve_dict[contrast.name] = {}
    all_curves = []
    for subj in subjects:
        for hemi in ["left_hemi", "right_hemi"]:
            distances, curves = subj.smoothness(
                contrast,
                hemi,
                human_bin_edges,
                distance_cutoff=distance_cutoff,
                num_samples=num_samples,
            )
            _, chance_curves = subj.smoothness(
                contrast,
                hemi,
                human_bin_edges,
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
            smoothness_results["Model"].append("Human")
            smoothness_results["Seed"].append(f"{subj.name}_{hemi}")
            smoothness_results["Smoothness"].append(np.mean(smoo))
            smoothness_results["Contrast"].append(contrast.name)

            # normalize curves based on chance
            normalized_curves = [curve / chance_mean for curve in curves]
            all_curves.extend(normalized_curves)

    curve_dict[contrast.name]["Human"] = {
        "Distances": distances,
        "Curves": all_curves,
    }
```

```python
bin_edges = human_bin_edges

for contrast in tqdm(contrasts):
    common = {
        "num_samples": num_samples,
        "bin_edges": bin_edges,
        "contrast": contrast,
        "distance_cutoff": distance_cutoff,
    }
    for name, seeds in tissues.items():
        all_curves = []
        for seed, tissue in seeds.items():
            try:
                distances, curves = tissue.category_smoothness(**common)
                _, chance_curves = tissue.category_smoothness(**common, shuffle=True)
            except ValueError:
                continue

            # compute the value expected by chance
            chance_mean = np.nanmean(np.concatenate(chance_curves))

            smoo = [spatial_utils.smoothness(curve) for curve in curves]
            smoothness_results["Model"].append(figure_utils.get_label(name))
            smoothness_results["Seed"].append(seed)
            smoothness_results["Smoothness"].append(np.nanmean(smoo))
            smoothness_results["Contrast"].append(contrast.name)

            normalized_curves = [curve / chance_mean for curve in curves]
            all_curves.extend(normalized_curves)

        curve_dict[contrast.name][name] = {
            "Distances": distances,
            "Curves": all_curves,
        }
```

```python
sr = pd.DataFrame(smoothness_results)
```

## Permutation testing

```python
def smooth_stat(groupA, groupB):
    """Return the median of all pairwise distances"""
    vals = []
    for a in groupA:
        for b in groupB:
            vals.append(np.mean(a - b))

    return np.mean(vals)


def permutation_test(groupA, groupB, statistic=smooth_stat, n_perm=25_000):
    # compute the true value
    true = smooth_stat(groupA, groupB)

    # compute the permuted value
    combined = groupA + groupB
    nA = len(groupA)

    null_distribution = []
    for _ in tqdm(range(n_perm)):
        random.shuffle(combined)
        stat = statistic(combined[:nA], combined[nA:])
        null_distribution.append(stat)

    null_distribution = np.array(null_distribution)

    # greater
    greater = null_distribution >= true
    greater_p = (np.sum(greater) + 1) / (n_perm + 1)

    lesser = null_distribution <= true
    lesser_p = (np.sum(lesser) + 1) / (n_perm + 1)

    two_sided_p = 2 * min(greater_p, lesser_p)

    return null_distribution, true, two_sided_p
```

```python
def get_profiles(df, model):
    """get_profiles is a helper function to get the smoothness of each category
    for each instance of a model
    """
    profs = []

    categories = sorted(df.Contrast.unique())
    matching = df[df.Model == model]
    for seed in matching.Seed.unique():
        prof = []
        for cat in categories:
            prof.append(
                float(
                    matching[(matching.Seed == seed) & (matching.Contrast == cat)][
                        "Smoothness"
                    ].item()
                )
            )

        profs.append(np.array(prof))

    return profs
```

```python
sr.groupby("Model").mean()["Smoothness"]
```

```python
# run the actual permutation tests (including human-to-human)
human_profiles = get_profiles(sr, "Human")

# in these plots (not published), the red line is the true difference between humans and each group, and the blue distribution is the null distribution
for model in sr.Model.unique():
    model_profiles = get_profiles(sr, model)
    rand, true, p = permutation_test(human_profiles, model_profiles)

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.hist(rand, bins=150)
    ax.axvline(true, c="red")
    ax.set_title(f"{model}: p = {p:.4f}")
```

```python
panel_b_fig, (curves_row, smoothness_row) = plt.subplots(
    figsize=(4, 1.7),
    ncols=5,
    nrows=2,
    gridspec_kw={"hspace": 1, "height_ratios": [1, 1]},
)
```

```python
# plot curves
for ax, contrast_name in zip(curves_row, contrast_order):
    names = curve_dict[contrast_name]
    for name in ["Unoptimized", "DNN-SOM", "ITN", "Human", "SW_0pt25"]:
        res = names.get(name)
        if res is None:
            continue
        curves = np.stack(res["Curves"])
        mn_curve = np.mean(curves, axis=0)
        se = np.std(curves, axis=0)
        label = figure_utils.get_label(name)
        line_color = color_lookup[label]
        if line_color == marker_fill:
            line_color = "k"

        dist = res["Distances"][: len(mn_curve)]
        line_handle = ax.plot(
            dist,
            mn_curve,
            label=name,
            color=line_color,
            mec="k",
            mfc=marker_fill,
            marker=marker_lookup[label],
            markevery=2,
            markersize=3,
            mew=0.5,
            lw=1,
            alpha=0.8,
        )

        ax.fill_between(
            dist,
            mn_curve - se,
            mn_curve + se,
            alpha=0.3,
            facecolor=line_handle[0].get_color(),
        )

    ax.legend().remove()
    plot_utils.remove_spines(ax)

    if contrast_name == "Faces":
        ax.set_ylabel((r"$\Delta$ Selectivity" "\n" "(vs. Chance)"))
        ax.set_yticks([0, 1.0])
        ax.set_ylim([0, 1.5])
    else:
        ax.set_ylabel("")
        ax.set_yticks([])
```

```python
order = ["TDANN", "DNN-SOM", "ITN", "Functional Only", "Unoptimized"]
```

```python
for ax, contrast_name in zip(smoothness_row, contrast_order):
    sub = sr[sr.Contrast == contrast_name]
    if len(sub) == 0:
        continue
    plot_sub = sub[(sub.Model != "Human")]

    sns.barplot(
        data=plot_sub,
        x="Model",
        y="Smoothness",
        palette=color_lookup,
        order=order,
        ax=ax,
    )
    ax.set_xlabel("")

    human_val = np.array(sub[sub.Model == "Human"].Smoothness)

    ax.axhline(
        np.mean(human_val), color=figure_utils.get_color("Human"), linestyle="dashed"
    )

    for i, name in enumerate(order):
        marker = marker_lookup[name]
        if marker is None:
            continue

        ax.scatter(i, 0.2, marker=marker_lookup[name], **mkw, s=6, zorder=5)

    ax.set_xticks([])
    plot_utils.remove_spines(ax)
    ax.set_ylim([0, 1.0])
    if contrast_name == "Faces":
        ax.set_ylabel("Smoothness")
        ax.set_yticks([0, 0.5, 1.0])
    else:
        ax.set_ylabel("")
        ax.set_yticks([])

    ax.set_ylim([0, 1])
```

```python
panel_b_fig
```

```python
%%capture
figure_utils.save(panel_b_fig, "F03/cd_smoothness")
```

# Panels f-g: patch stats

```python
patch_area_results = {
    "Species": [],
    "Contrast": [],
    "Area": [],
    "Seed": [],
}

for subject in subjects:
    hemis = {"Left Hemi": subject.lh_patches, "Right Hemi": subject.rh_patches}
    for hemisphere_name, hemi_patches in hemis.items():
        for patch in hemi_patches:
            patch_area_results["Species"].append("Human")
            patch_area_results["Contrast"].append(patch.contrast.name)
            patch_area_results["Area"].append(np.mean(patch.area))
            patch_area_results["Seed"].append(f"{subject.name}_{hemisphere_name}")

# add the model data
for name, seeds in tissues.items():
    for seed, tissue in seeds.items():
        if len(tissue.patches) == 0:
            patch_area_results["Species"].append(figure_utils.get_label(name))
            patch_area_results["Contrast"].append(patch.contrast.name)
            patch_area_results["Area"].append(0)
            patch_area_results["Seed"].append(seed)
        for patch in tissue.patches:
            patch_area_results["Species"].append(figure_utils.get_label(name))
            patch_area_results["Contrast"].append(patch.contrast.name)
            patch_area_results["Area"].append(np.mean(patch.area))
            patch_area_results["Seed"].append(seed)
```

```python
patch_area_df = pd.DataFrame(patch_area_results)
```

```python
human_data = patch_area_df[patch_area_df.Species == "Human"]
mean_by_subject = human_data.groupby(["Seed", "Contrast"], as_index=False).mean("Area")
```

```python
human_medians = {}

unique_contrasts = mean_by_subject["Contrast"].unique()
for contrast in unique_contrasts:
    contrast_data = np.array(
        mean_by_subject[mean_by_subject["Contrast"] == contrast]["Area"].tolist()
    )[:, np.newaxis]
    diffmat = np.abs(contrast_data - contrast_data.T)
    diffmat_lt = array_utils.lower_tri(diffmat)
    median = np.median(diffmat_lt)
    human_medians[contrast] = median
```

```python
def absolute_pairwise_difference(x, y):
    """computes abs(x_i - y_j) for all i, j"""
    x = np.array(x)[:, np.newaxis]
    y = np.array(y)[:, np.newaxis]
    diffmat = x - y.T
    return np.abs(diffmat)
```

```python
patch_diff_by_species = {}
```

```python
human_data = patch_area_df[patch_area_df.Species == "Human"]

for species in patch_area_df["Species"].unique():
    species_data = patch_area_df[patch_area_df.Species == species]

    for contrast in contrast_order:
        # human
        human_matching = human_data[human_data["Contrast"] == contrast]
        human_values = human_matching["Area"].tolist() if len(human_matching) else [0]

        # model
        matching = species_data[species_data["Contrast"] == contrast]
        seeds = matching["Seed"].unique()
        for seed in seeds:
            identifier = f"{species}_{seed}"
            seed_matching = matching[matching.Seed == seed]
            values = seed_matching["Area"].tolist() if len(matching) else [0]

            # comparison to human
            human_diff = absolute_pairwise_difference(values, human_values).ravel()
            median_diff = np.median(human_diff)

            # get the existing list
            if identifier in patch_diff_by_species.keys():
                patch_diff_by_species[identifier].append(median_diff)
            else:
                patch_diff_by_species[identifier] = []
```

```python
patch_count_results = {
    "Species": [],
    "Contrast": [],
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
            patch_count_results["Count"].append(len(matching_patches))
            patch_count_results["Seed"].append(f"{subject.name}_{hemisphere_name}")

# add the model data
for name, seeds in tissues.items():
    for seed, tissue in seeds.items():
        for contrast in contrasts:
            matching_patches = list(
                filter(lambda p: p.contrast == contrast, tissue.patches)
            )
            patch_count_results["Species"].append(figure_utils.get_label(name))
            patch_count_results["Contrast"].append(contrast.name)
            patch_count_results["Count"].append(len(matching_patches))
            patch_count_results["Seed"].append(seed)
```

```python
patch_count_df = pd.DataFrame(patch_count_results)
```

```python
patch_count_df.groupby("Species").mean()
```

```python
patch_area_df.groupby("Species").mean()
```

```python
def plot_helper(ax, df, key):
    """We use the same function to plot patch count and patch area"""
    # separate out human and Unoptimized
    plot_sub = df[
        (df.Species != "Human")
        & (df.Species != "Unoptimized")
        & (df.Species != "Functional Only")
    ]
    order = ["TDANN", "DNN-SOM", "ITN"]
    human_val = np.array(df[df.Species == "Human"][key])

    # barplot the non-human data
    sns.barplot(
        data=plot_sub, x="Species", y=key, palette=color_lookup, order=order, ax=ax
    )

    # add dashed line for human data
    ax.axhline(
        np.mean(human_val), color=figure_utils.get_color("Human"), linestyle="dashed"
    )

    ax.set_xticks([])
```

```python
fig, axes = plt.subplots(figsize=(2.5, 0.75), ncols=2, gridspec_kw={"wspace": 1.2})

plot_helper(axes[0], patch_count_df, "Count")
plot_helper(axes[1], patch_area_df, "Area")

# pretty
axes[0].set_ylabel("Number of Patches")
axes[1].set_ylabel(r"Patch Area [$mm^2$]")
axes[0].yaxis.set_major_formatter("{x:.1f}")
for ax in axes:
    ax.set_xlabel("")
    ax.set_xticks([])
    plot_utils.remove_spines(ax)
figure_utils.save(fig, "F03/fg_patch_stats")
```

```python
patch_count_by_species = {}
```

```python
human_data = patch_count_df[patch_count_df.Species == "Human"]

for species in patch_count_df["Species"].unique():
    species_data = patch_count_df[patch_count_df.Species == species]

    for contrast in contrast_order:
        # human
        human_matching = human_data[human_data["Contrast"] == contrast]
        human_values = human_matching["Count"].tolist() if len(human_matching) else [0]

        # model
        matching = species_data[species_data["Contrast"] == contrast]
        seeds = matching["Seed"].unique()
        for seed in seeds:
            identifier = f"{species}_{seed}"
            seed_matching = matching[matching.Seed == seed]
            values = seed_matching["Count"].tolist() if len(matching) else [0]

            # comparison to human
            human_diff = absolute_pairwise_difference(values, human_values).ravel()
            median_diff = np.median(human_diff)

            # get the existing list
            if identifier in patch_count_by_species.keys():
                patch_count_by_species[identifier].append(median_diff)
            else:
                patch_count_by_species[identifier] = []
```

```python
mean_patch_diff_by_species = {
    k: np.nanmean(v) for k, v in patch_diff_by_species.items()
}
mean_patch_count_by_species = {
    k: np.nanmean(v) for k, v in patch_count_by_species.items()
}
```

```python
scatter_fig, scatter_axes = plt.subplots(
    figsize=(2.5, 0.75), ncols=2, gridspec_kw={"wspace": 1.2}
)

ax = scatter_axes[0]
for k in mean_patch_count_by_species:
    count = mean_patch_count_by_species[k] + rng.normal(scale=5e-2)
    area = mean_patch_diff_by_species[k] + rng.normal(scale=5e-2)
    if np.isnan(area):
        continue

    species = k.split("_")[0]
    color = color_lookup[species]
    ax.scatter(
        count,
        area,
        c=color,
        s=10,
        edgecolor="k",
        alpha=0.7,
        linewidths=0.5,
        marker=marker_lookup[species],
    )

ax.set_xlabel((r"$\Delta$" " Human" "\n" "Patch Count"))
ax.set_ylabel((r"$\Delta$" " Human" "\n" r"Patch Area [$mm^2$]"))
plot_utils.remove_spines(ax)
```

```python
patch_count_df.head()
```

```python
patch_count_df.anova(dv="Count", between=["Species"])
```

```python
patch_area_df.anova(dv="Area", between=["Species"])
```

```python

```

```python
patch_count_df.pairwise_tukey(dv="Count", between=["Species"])
```

```python
patch_area_df.pairwise_tukey(dv="Area", between=["Species"])
```

### Permutation test for patch count, patch size

```python
def get_patch_profiles(df, model, key):
    categories = sorted(df.Contrast.unique())
    profs = []
    matching = df[df.Species == model]
    for seed in matching.Seed.unique():
        prof = []
        for cat in categories:
            vals = matching[(matching.Seed == seed) & (matching.Contrast == cat)]
            if len(vals) == 0:
                prof.append(0)
            else:
                prof.append(
                    matching[(matching.Seed == seed) & (matching.Contrast == cat)][
                        key
                    ].item()
                )

        profs.append(np.array(prof))

    return profs
```

```python
human_profiles = get_patch_profiles(patch_count_df, "Human", "Count")

for model in sr.Model.unique():
    model_profiles = get_patch_profiles(patch_count_df, model, "Count")
    rand, true, p = permutation_test(human_profiles, model_profiles)

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.hist(rand, bins=100)
    ax.axvline(true, c="red")
    ax.set_title(f"{model}: p = {p:.4f}")
```

```python
patch_area_mn_df = (
    patch_area_df.groupby(["Species", "Contrast", "Seed"]).mean().reset_index()
)
human_profiles = get_patch_profiles(patch_area_mn_df, "Human", "Area")

for model in sr.Model.unique():
    model_profiles = get_patch_profiles(patch_area_mn_df, model, "Area")
    rand, true, p = permutation_test(human_profiles, model_profiles)

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.hist(rand, bins=100)
    ax.axvline(true, c="red")
    ax.set_title(f"{model}: p = {p:.4f}")
```

# Panel h: co-localization

```python
def get_sel(tissue, contrast: floc.Contrast):
    """Helper that gets selectivity from tissue, whether it's a ITMap or an ITN"""
    if hasattr(tissue, "responses"):
        return tissue.responses.selectivity(on_categories=contrast.on_categories)

    return tissue.maps[contrast.name]
```

```python
def model_colocalization(
    tissue,
    con1: floc.Contrast,
    con2: floc.Contrast,
    window_width: float = 1,
):
    T_THRESH = 4

    # split the tissue into non-overlapping 1mm x 1mm windows, reject windows with
    # less than 6 units
    windows = [
        window
        for window in spatial_utils.get_adjacent_windows(tissue.positions, width=7)
        if len(window.indices) > 5
    ]

    # get the selectivity of all units to each contrast
    sel1 = get_sel(tissue, con1)
    sel2 = get_sel(tissue, con2)

    # compute the fraction of units in each window that clear the selectivity threshold
    frac1_list = []
    frac2_list = []
    for window in windows:
        frac1 = np.mean(sel1[window.indices] > T_THRESH)
        frac2 = np.mean(sel2[window.indices] > T_THRESH)

        frac1_list.append(frac1)
        frac2_list.append(frac2)

    # check the correlation of the counts across windows
    r, p = stats.spearmanr(frac1_list, frac2_list)
    distance = 1 - r
    return 1 - distance / 2
```

```python
def human_colocalization(subject, hemi, con1, con2):
    T_THRESH = 4
    full_hemi = "right_hemi" if hemi == "rh" else "left_hemi"

    # get nonoverlapping windows of 10 x 10 mm
    face_data = subject.data["faces"]
    hemi_data = getattr(face_data, full_hemi).T
    mask = ~np.isnan(hemi_data)
    where = np.where(mask)
    positions = np.stack(where).T

    # convert positions into mm
    positions = positions * getattr(subject.xform_info, full_hemi).mm_per_px

    # get windows
    windows = [
        window
        for window in spatial_utils.get_adjacent_windows(positions, width=10)
        if len(window.indices) > 10
    ]

    sel1 = getattr(subject.data[con1], full_hemi).T.ravel()
    sel2 = getattr(subject.data[con2], full_hemi).T.ravel()

    frac1_list = []
    frac2_list = []
    for window in windows:
        frac1 = np.nanmean(sel1[~np.isnan(sel1)][window.indices] > T_THRESH)
        frac2 = np.nanmean(sel2[~np.isnan(sel2)][window.indices] > T_THRESH)

        frac1_list.append(frac1)
        frac2_list.append(frac2)

    r, p = stats.spearmanr(frac1_list, frac2_list)
    dist = 1 - r
    return 1 - dist / 2
```

```python
coloc_res = {"Name": [], "Seed": [], "Pair": [], "Coloc": []}

for name, seeds in tqdm(tissues.items()):
    for seed, tissue in seeds.items():
        fp = model_colocalization(
            tissue, contrast_dict["Faces"], contrast_dict["Places"]
        )
        fb = model_colocalization(
            tissue, contrast_dict["Faces"], contrast_dict["Bodies"]
        )

        coloc_res["Name"].append(figure_utils.get_label(name))
        coloc_res["Seed"].append(seed)
        coloc_res["Pair"].append("Face-Body")
        coloc_res["Coloc"].append(fb)

        coloc_res["Name"].append(figure_utils.get_label(name))
        coloc_res["Seed"].append(seed)
        coloc_res["Pair"].append("Face-Place")
        coloc_res["Coloc"].append(fp)
```

```python
for subject in tqdm(subjects):
    for hemi in ["lh", "rh"]:
        fp = human_colocalization(subject, hemi, "faces", "places")
        fb = human_colocalization(subject, hemi, "faces", "bodies")

        coloc_res["Name"].append("Human")
        coloc_res["Seed"].append(f"{subject.name}_{hemi}")
        coloc_res["Pair"].append("Face-Body")
        coloc_res["Coloc"].append(fb)

        coloc_res["Name"].append("Human")
        coloc_res["Seed"].append(f"{subject.name}_{hemi}")
        coloc_res["Pair"].append("Face-Place")
        coloc_res["Coloc"].append(fp)

coloc_df = pd.DataFrame(coloc_res)
```

```python
coloc_df[coloc_df.Name == "TDANN"]
```

```python
score_results = {"Name": [], "Score": []}

ax = scatter_axes[1]
for name in coloc_df.Name.unique():
    sub = coloc_df[coloc_df.Name == name]
    fb = sub[coloc_df.Pair == "Face-Body"]["Coloc"]
    fp = sub[coloc_df.Pair == "Face-Place"]["Coloc"]

    for x, y in zip(fb, fp):
        score_results["Name"].append(name)
        score_results["Score"].append(x - y)
        ax.scatter(
            x,
            y,
            c=color_lookup[name],
            edgecolor="k",
            alpha=0.7,
            s=10,
            linewidths=0.5,
            marker=marker_lookup[name],
        )

ax.axhline(0.5, c="k", alpha=0.8, lw=0.75)
ax.axvline(0.5, c="k", alpha=0.8, lw=0.75)

ax.set_xlabel("Face-Body Overlap")
ax.set_ylabel("Face-Place Overlap")
ax.set_xlim([0.35, 0.8])
ax.set_ylim([0, 0.8])
ax.set_xticks([0.5])
ax.set_yticks([0.5])

score_df = pd.DataFrame(score_results)
plot_utils.remove_spines(ax)
figure_utils.save(scatter_fig, "F03/h_colocalization")
```

```python
scatter_fig
```

```python
# legend for things we actually used
```

```python
human_fb = coloc_df[(coloc_df.Name == "Human") & (coloc_df.Pair == "Face-Body")][
    "Coloc"
]
human_fp = coloc_df[(coloc_df.Name == "Human") & (coloc_df.Pair == "Face-Place")][
    "Coloc"
]
print(pg.compute_bootci(human_fb, func="mean"))
print(pg.compute_bootci(human_fp, func="mean"))
```

```python
tdann_fb = coloc_df[(coloc_df.Name == "TDANN") & (coloc_df.Pair == "Face-Body")][
    "Coloc"
]
tdann_fp = coloc_df[(coloc_df.Name == "TDANN") & (coloc_df.Pair == "Face-Place")][
    "Coloc"
]
print(pg.compute_bootci(tdann_fb, func="mean"))
print(pg.compute_bootci(tdann_fp, func="mean"))
```

```python
itn_fb = coloc_df[(coloc_df.Name == "ITN") & (coloc_df.Pair == "Face-Body")]["Coloc"]
itn_fp = coloc_df[(coloc_df.Name == "ITN") & (coloc_df.Pair == "Face-Place")]["Coloc"]
print(itn_fb.item(), itn_fp.item())
```

```python
for model in ["Human", "DNN-SOM", "Unoptimized", "Functional Only", "TDANN"]:
    fb = coloc_df[(coloc_df.Name == model) & (coloc_df.Pair == "Face-Body")]["Coloc"]
    fp = coloc_df[(coloc_df.Name == model) & (coloc_df.Pair == "Face-Place")]["Coloc"]
    print(f"{model}\n=====\n")
    display(pg.wilcoxon(fb, fp, alternative="greater"))
```

# not in figure: estimates of tissue size in human VTC
these are going to be strict overestimates, since 1) we don't fill the rectangle and 2) we ignore a good chunk of the anterior region of the ROI

```python
mms = []
for hemi in ["left_hemi", "right_hemi"]:
    for subject in subjects:
        width_px = getattr(subject.data["faces"], hemi).shape[0]
        mm_per_px = getattr(subject.xform_info, hemi).mm_per_px
        total_mm = mm_per_px * width_px
        mms.append(total_mm)
```

```python
fig, ax = plt.subplots(figsize=(1, 1))
ax.hist(mms, bins=10)
```
