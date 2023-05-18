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
    display_name: Python 2
    language: python
    name: python2
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
from spacetorch.analyses.alpha import palette, name_lookup, sw_from_name
from spacetorch.datasets.ringach_2002 import load_ringach_data
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    spatial_utils,
    seed_str,
)
from spacetorch.analyses.sine_gratings import (
    get_sine_tissue,
    get_smoothness_curves,
)
from spacetorch.analyses.floc import get_floc_tissue
import spacetorch.analyses.rsa as rsa
from spacetorch.datasets import floc
from spacetorch.maps import nsd_floc
from spacetorch.maps.v1_map import metric_dict
from spacetorch.maps.pinwheel_detector import PinwheelDetector
from spacetorch.maps.screenshot_maps import NauhausOrientationTissue
from spacetorch.paths import PROJ_DIR
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
plot_con_order = ["Objects", "Characters", "Places", "Faces", "Bodies"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}
pprint.pprint(contrasts)
```

```python
# set up standard line kw args
line_kwargs = {"color": "k", "marker": ".", "markersize": 8, "lw": 2}
```

```python
V1_LAYER, VTC_LAYER = "layer2.0", "layer4.1"
positions = core.get_positions("simclr_swap")
random_positions = core.get_positions("retinotopic")
```

```python
base = "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
SWS = {
    "SW_0": "_lw0",
    "SW_0pt1": "_lw01",
    "SW_0pt25": "",
    "SW_0pt5": "_lwx2",
    "SW_1pt25": "_lwx5",
    "SW_2pt5": "_lwx10",
    "SW_25": "_lwx100",
}
```

```python
seeds = range(5)
tissues = {
    "V1": {
        sw: {
            seed: get_sine_tissue(
                f"{base}{lwmod}{seed_str(seed)}", positions[V1_LAYER], layer=V1_LAYER
            )
            for seed in seeds
        }
        for sw, lwmod in SWS.items()
    },
    "VTC": {
        sw: {
            seed: get_floc_tissue(
                f"{base}{lwmod}{seed_str(seed)}", positions[VTC_LAYER], layer=VTC_LAYER
            )
            for seed in seeds
        }
        for sw, lwmod in SWS.items()
    },
}

tissues["V1"]["Unoptimized"] = {
    seed: get_sine_tissue(
        f"{base}{seed_str(seed)}",
        random_positions[V1_LAYER],
        layer=V1_LAYER,
        step="random",
    )
    for seed in seeds
}
tissues["VTC"]["Unoptimized"] = {
    seed: get_floc_tissue(
        f"{base}{seed_str(seed)}",
        random_positions[VTC_LAYER],
        layer=VTC_LAYER,
        step="random",
    )
    for seed in seeds
}
```

# V1


## Maps

```python
extent = np.ptp(tissues["V1"]["SW_0pt25"][1]._positions)
```

```python
fig, v1_row = plt.subplots(nrows=1, ncols=8, figsize=(6, 6 / 8))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

handle = None
lims = [30, 70]
frac = (lims[1] - lims[0]) / 100
for ax, (name, seeds) in zip(v1_row, tissues["V1"].items()):
    # choose an example seed (#1) and restrict map to a 40% x 40% window
    tissue = seeds[1]
    tissue.set_mask_by_pct_limits([lims, lims])

    # plot pinwheel centers
    pindet = PinwheelDetector(tissue)
    pos, neg = pindet.count_pinwheels()
    pos_centers, neg_centers = pindet.centers
    for x, y in pos_centers:
        ax.scatter(x, y, c="k", s=1)

    for x, y in neg_centers:
        ax.scatter(x, y, c="w", s=1)

    # plot the smoothed OPM
    smoothed = pindet.smoothed
    h = ax.imshow(smoothed, metric_dict["angles"].colormap, interpolation="nearest")
    if name == "SW_0pt25":
        handle = h

    # add scale bar
    total_px = smoothed.shape[0]
    total_mm = extent * frac
    px_per_mm = total_px / total_mm
    plot_utils.add_scale_bar(ax, 2 * px_per_mm, flipud=True)

    # title and axis formatting
    ax.set_title(name_lookup[name], fontsize=7, color=palette[name])
    ax.axis("off")

# add colorbar
cax = fig.add_axes([0.91, 0.2, 0.005, 0.6])
cb = plt.colorbar(
    handle,
    cax=cax,
    label=r"$\theta$",
)
cb.set_ticks([0, 90, 180])
```
```python
figure_utils.save(fig, "F04/a_ori_maps")
```

## V1 Quant

```python
v1_fig, v1_axes = plt.subplots(figsize=(5.5, 1), ncols=3, gridspec_kw={"wspace": 0.65})
v1_axes[0].set_title("Function")
v1_axes[1].set_title("Map Smoothness")
v1_axes[2].set_title("Topographic Phenomena")
```

### % ori sel

```python
CV_THRESH = 0.6
RESP_THRESH = 1.0

ori_results = {"FracSel": [], "Alpha": [], "Seed": []}

for name, seeds in tissues["V1"].items():
    sw = sw_from_name.get(name)
    if sw is None:
        continue

    for seed, tissue in seeds.items():
        if tissue is None:
            continue
        cv = tissue.responses.circular_variance
        mean_responses = tissue.responses._data.mean("image_idx").values

        # get CV of units with sufficiently high mean response
        cv = cv[~np.isnan(cv) & (mean_responses > RESP_THRESH)]

        # count fraction of units with CV below threshold
        frac_sel = np.mean(cv < CV_THRESH)

        ori_results["FracSel"].append(frac_sel)
        ori_results["Alpha"].append(sw)
        ori_results["Seed"].append(seed)
```

```python
# get the value expected for random (Unoptimized) models
random_cv = []
for seed, tissue in tissues["V1"]["Unoptimized"].items():
    cv = tissue.responses.circular_variance
    mean_responses = tissue.responses._data.mean("image_idx").values
    cv = cv[~np.isnan(cv) & (mean_responses > RESP_THRESH)]
    frac_sel = np.mean(cv < CV_THRESH)
    random_cv.append(frac_sel)

random_mean = np.mean(random_cv)
```

```python
# compute value for macaque V1 from Ringach et al., 2022
macaque_mean = np.mean(load_ringach_data(fieldname="orivar") < CV_THRESH)
print(macaque_mean)
```

```python
ori_df = pd.DataFrame(ori_results)
display(ori_df.groupby("Alpha").mean(numeric_only=False))
```

```python
ax = v1_axes[0]

# plot model data
sns.lineplot(data=ori_df, x="Alpha", y="FracSel", ax=ax, **line_kwargs)

# macaque data
ax.axhline(
    macaque_mean, color=figure_utils.get_color("Macaque V1"), linestyle="dashed", lw=1
)
ax.axhline(
    random_mean, color=figure_utils.get_color("Random"), linestyle="dashed", lw=1
)

ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 50])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])
ax.set_ylabel("Fraction of Orientation\nSelective Units")

ax.set_xlabel(r"$\alpha$")
plot_utils.remove_spines(ax)
```

### Smoothness

```python
v1_smoothness_results = {"Alpha": [], "Seed": [], "Iteration": [], "Smoothness": []}

for name, seeds in tqdm(tissues["V1"].items()):
    sw = sw_from_name.get(name)
    if sw is None:
        continue

    for seed, tissue in seeds.items():
        if tissue is None:
            continue
        tissue.reset_unit_mask()
        tissue.set_unit_mask_by_ptp_percentile("angles", 75)
        _, curves = get_smoothness_curves(tissue)
        smoothnesses = [spatial_utils.smoothness(curve) for curve in curves]

        for iteration, smoothness in enumerate(smoothnesses):
            v1_smoothness_results["Alpha"].append(sw)
            v1_smoothness_results["Seed"].append(seed)
            v1_smoothness_results["Iteration"].append(iteration)
            v1_smoothness_results["Smoothness"].append(smoothness)
v1_smoothness_df = pd.DataFrame(v1_smoothness_results)
```

```python
v1_smoothness_df.groupby("Alpha").mean(numeric_only=True)
```

```python
rand_smooth = []
for seed, tissue in tqdm(tissues["V1"]["Unoptimized"].items()):
    if tissue is None:
        continue
    tissue.reset_unit_mask()
    tissue.set_unit_mask_by_ptp_percentile("angles", 75)
    _, curves = get_smoothness_curves(tissue)
    smoothnesses = [spatial_utils.smoothness(curve) for curve in curves]
    rand_smooth.extend(smoothnesses)
rand_mean = np.nanmean(rand_smooth)
```

```python
# get some V1 data
ori_tissue = NauhausOrientationTissue()
neural_bin_edges = np.linspace(0, 0.7, 30)

_, neural_curves = ori_tissue.difference_over_distance(
    num_samples=100, sample_size=1000, bin_edges=neural_bin_edges
)
neural_smoothnesses = [spatial_utils.smoothness(curve) for curve in neural_curves]
neural_mean = np.nanmean(neural_smoothnesses)
```

```python
v1_sm_by_alpha = (
    v1_smoothness_df.groupby("Alpha").mean(numeric_only=False).reset_index()
)
display(v1_sm_by_alpha)
```

```python
ax = v1_axes[1]

# average by iteration so that CI is plotted only over seeds
v1_sm_av = v1_smoothness_df.groupby(["Alpha", "Seed"]).mean().reset_index()
sns.lineplot(data=v1_sm_av, x="Alpha", y="Smoothness", ax=ax, **line_kwargs)
ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 60])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])

# add macaque and random
ax.axhline(
    neural_mean, color=figure_utils.get_color("Macaque V1"), linestyle="dashed", lw=1
)
ax.axhline(
    rand_mean, color=figure_utils.get_color("Unoptimized"), linestyle="dashed", lw=1
)

# axis labels
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("OPM Smoothness")
plot_utils.remove_spines(ax)
```

### Pinwheel Counts

```python
pinwheel_results = {"Density": [], "Alpha": [], "Seed": []}

for name, seeds in tissues["V1"].items():
    sw = sw_from_name.get(name)
    if sw is None:
        continue
    edge_size = extent
    hypercol_width = 3.5
    num_hcol = edge_size / hypercol_width

    for seed, tissue in tqdm(seeds.items(), desc=name):
        if tissue is None:
            continue
        tissue.reset_unit_mask()

        pindet = PinwheelDetector(tissue)
        pos, neg = pindet.count_pinwheels()
        total = pos + neg

        density = total / (num_hcol**2)

        pinwheel_results["Alpha"].append(sw)
        pinwheel_results["Seed"].append(seed)
        pinwheel_results["Density"].append(density)

pinwheel_df = pd.DataFrame(pinwheel_results)
```

```python
ax = v1_axes[2]

sns.lineplot(data=pinwheel_df, x="Alpha", y="Density", ax=ax, **line_kwargs)
ax.axhline(np.pi, linestyle="dashed", c=figure_utils.get_color("Macaque V1"), lw=1)
ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 30])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(("Pinwheels /" "\n" r"Column Spacing$^2$"))
plot_utils.remove_spines(ax)
```

```python
pinwheel_df.groupby("Alpha").mean()
```

```python
# add vertical lines at our default value of alpha = 0.25
for ax in v1_axes:
    ax.axvline(0.25, c="k", alpha=0.1)
```

```python
figure_utils.save(v1_fig, "F04/bcd_v1_quant")
v1_fig
```

# VTC

```python
mat_dir = PROJ_DIR / "nsd_tvals"
subjects = nsd_floc.load_data(mat_dir, domains=contrasts, find_patches=True)
```

## VTC Maps

```python
fig, vtc_row = plt.subplots(nrows=1, ncols=8, figsize=(6, 6 / 8))
plt.subplots_adjust(hspace=0.05, wspace=0.05)
for ax, (name, seeds) in zip(vtc_row, tissues["VTC"].items()):
    tissue = seeds[1]
    tissue.make_selectivity_map(
        ax,
        marker=".",
        contrasts=[contrast_dict[n] for n in plot_con_order],
        size_mult=2e-3,
        foreground_alpha=0.8,
        selectivity_threshold=12,
        rasterized=True,
        linewidths=0,
    )
    ax.axis("off")
    plot_utils.add_scale_bar(ax, 10)
    ax.set_title(name_lookup[name], fontsize=7, color=palette[name])
```

```python
figure_utils.save(fig, "F04/e_vtc_sel_maps")
```

## VTC Quant

```python
vtc_fig, vtc_axes = plt.subplots(figsize=(5.5, 1), ncols=3, gridspec_kw={"wspace": 0.65})
vtc_axes[0].set_title("")
vtc_axes[1].set_title("")
vtc_axes[2].set_title("")
```

### Similarity to Selectivity Distribution

```python

```

```python
human_rsms = []

for subject in subjects:
    for hemi in ["left_hemi", "right_hemi"]:
        human_rsms.append(rsa.get_human_rsm(subject, hemi))
```

```python
rsa_results = {"Alpha": [], "Sim": [], "Seed": [], "Human": []}

for name, seeds in tqdm(tissues["VTC"].items()):
    sw = figure_utils.get_sw(name)
    if sw is None:
        continue

    for seed, tissue in seeds.items():
        rsm = rsa.get_model_rsm(tissue)

        for hidx, human_rsm in enumerate(human_rsms):
            sim = rsa.rsm_similarity(rsm, human_rsm)
            rsa_results["Alpha"].append(sw)
            rsa_results["Sim"].append(sim)
            rsa_results["Seed"].append(seed)
            rsa_results["Human"].append(hidx)

rsa_df = pd.DataFrame(rsa_results)
```

```python
human2human = []
for i, rsm_a in enumerate(human_rsms):
    for j, rsm_b in enumerate(human_rsms):
        if i == j:
            continue
        human2human.append(rsa.rsm_similarity(rsm_a, rsm_b))
```

```python
unopt = []
for seed, tissue in tissues["VTC"]["Unoptimized"].items():
    rsm_a = rsa.get_model_rsm(tissue)
    for j, rsm_b in enumerate(human_rsms):
        unopt.append(rsa.rsm_similarity(rsm_a, rsm_b))
```

```python
ax = vtc_axes[0]

sns.lineplot(data=rsa_df, x="Alpha", y="Sim", ax=ax, **line_kwargs)
ax.set_ylabel(("Similarity to Human" "\n" r"VTC [Kendall's $\tau$]"))
ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 50])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])
ax.set_xlabel(r"$\alpha$")
ax.axhline(
    np.mean(human2human), ls="dashed", lw=1, color=figure_utils.get_color("Human")
)
ax.axhline(
    np.mean(unopt), ls="dashed", lw=1, color=figure_utils.get_color("Unoptimized")
)
plot_utils.remove_spines(ax)
```

### Smoothness

```python
N_BINS = 10
bin_edges = np.linspace(0, 60, N_BINS)
vtc_smoothness = {contrast.name: [] for contrast in contrasts}
for contrast in tqdm(contrasts):
    for subj in subjects:
        data = subj.data[contrast.name.lower()]

        for hemi in ["left_hemi", "right_hemi"]:
            _, curves = subj.smoothness(contrast, hemi, bin_edges)
            vtc_smoothness[contrast.name].extend(
                [spatial_utils.smoothness(curve) for curve in curves]
            )
```

```python
vtc_means = {}
for contrast_name, vals in vtc_smoothness.items():
    vtc_means[contrast_name] = np.nanmean(vals)
```

```python
vtc_smoothness_results = {
    "Alpha": [],
    "Seed": [],
    "Iteration": [],
    "Smoothness": [],
    "Category": [],
}

for name, seeds in tqdm(tissues["VTC"].items()):
    sw = sw_from_name.get(name)
    if sw is None:
        continue

    for seed, tissue in seeds.items():
        if tissue is None:
            continue

        for contrast in contrasts:
            _, curves = tissue.category_smoothness(
                contrast, num_samples=25, bin_edges=bin_edges
            )
            smoothnesses = [spatial_utils.smoothness(curve) for curve in curves]

            for iteration, smoothness in enumerate(smoothnesses):
                vtc_smoothness_results["Alpha"].append(sw)
                vtc_smoothness_results["Seed"].append(seed)
                vtc_smoothness_results["Iteration"].append(iteration)
                vtc_smoothness_results["Smoothness"].append(smoothness)
                vtc_smoothness_results["Category"].append(contrast.name)
```

```python
vtc_smoothness_df = pd.DataFrame(vtc_smoothness_results)
```

```python
df = (
    vtc_smoothness_df.groupby(["Alpha", "Category", "Seed"])
    .mean()
    .reset_index()
    .drop(columns=["Iteration"])
)

dist_to_human = {"Alpha": [], "Category": [], "Dist": []}


for category in df.Category.unique():
    vtc_mean = vtc_means[category]
    for alpha in df.Alpha.unique():
        matching = df[(df.Alpha == alpha) & (df.Category == category)]
        vals = np.array(matching["Smoothness"])
        human_dists = np.abs(vals - vtc_mean)

        for d in human_dists:
            dist_to_human["Alpha"].append(alpha)
            dist_to_human["Category"].append(category)
            dist_to_human["Dist"].append(d)

dist_to_human = pd.DataFrame(dist_to_human)
```

```python
dist_to_human.groupby("Alpha").mean()
```

```python
vtc_rand_smooth = {contrast.name: [] for contrast in contrasts}
for seed, tissue in tqdm(tissues["VTC"]["Unoptimized"].items()):
    for contrast in contrasts:
        _, curves = tissue.category_smoothness(contrast, num_samples=20)
        smoothnesses = [spatial_utils.smoothness(curve) for curve in curves]
        vtc_rand_smooth[contrast.name].extend(smoothnesses)
```

```python
vtc_rand_means = {}
for contrast_name, vals in vtc_rand_smooth.items():
    vtc_rand_means[contrast_name] = np.nanmean(vals)
```

```python
ax = vtc_axes[1]
for contrast, mean in vtc_means.items():
    ax.axhline(mean, color=contrast_colors[contrast], linestyle="dashed", lw=1)

sns.lineplot(
    data=vtc_smoothness_df,
    x="Alpha",
    y="Smoothness",
    color="k",
    hue="Category",
    palette=contrast_colors,
    marker=".",
    markersize=8,
    lw=2,
    ax=ax,
)
ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 60])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Selectivity Smoothness")
ax.legend().remove()
plot_utils.remove_spines(ax)
```

### Patch Size

```python
for name, seeds in tissues["VTC"].items():
    for seed, tissue in tqdm(seeds.items()):
        if tissue is None:
            continue
        tissue.patches = []
        for contrast in contrasts:
            tissue.find_patches(contrast)
```

```python
patch_results = {"Alpha": [], "Seed": [], "Category": [], "Count": [], "Size": []}

for name, seeds in tissues["VTC"].items():
    sw = sw_from_name.get(name)
    if sw is None:
        continue

    for seed, tissue in seeds.items():
        if tissue is None:
            continue

        for contrast in contrasts:
            matching_patches = list(
                filter(
                    lambda patch: patch.contrast.name == contrast.name, tissue.patches
                )
            )
            count = len(matching_patches)
            mean_size = np.mean([patch.area for patch in matching_patches])
            if np.isnan(mean_size):
                mean_size = 0
            patch_results["Alpha"].append(sw)
            patch_results["Seed"].append(seed)
            patch_results["Category"].append(contrast.name)
            patch_results["Count"].append(count)
            patch_results["Size"].append(mean_size)

patch_df = pd.DataFrame(patch_results)
```

```python
human_patch_sizes = {contrast.name: [] for contrast in contrasts}
human_patch_counts = {contrast.name: [] for contrast in contrasts}

for subject in subjects:
    hemis = {"Left Hemi": subject.lh_patches, "Right Hemi": subject.rh_patches}
    for hemisphere_name, hemi_patches in hemis.items():
        for contrast in contrasts:
            matching_patches = list(
                filter(lambda p: p.contrast == contrast, hemi_patches)
            )
            mean_size = np.mean([patch.area for patch in matching_patches])
            human_patch_sizes[contrast.name].append(mean_size)
            human_patch_counts[contrast.name].append(len(matching_patches))
```

```python
vtc_size_mean = {}
vtc_count_mean = {}
for contrast in contrasts:
    vtc_size_mean[contrast.name] = np.nanmean(human_patch_sizes[contrast.name])
    vtc_count_mean[contrast.name] = np.nanmean(human_patch_counts[contrast.name])
```

```python
df = (
    patch_df.groupby(["Alpha", "Category", "Seed"])
    .mean()
    .reset_index()
    .drop(columns=["Size"])
)

dist_to_human = {"Alpha": [], "Category": [], "Dist": []}


for category in df.Category.unique():
    vtc_mean = vtc_count_mean[category]

    for alpha in df.Alpha.unique():
        matching = df[(df.Alpha == alpha) & (df.Category == category)]
        vals = np.array(matching["Count"])
        human_dists = np.abs(vals - vtc_mean)

        for d in human_dists:
            dist_to_human["Alpha"].append(alpha)
            dist_to_human["Category"].append(category)
            dist_to_human["Dist"].append(d)

dist_to_human = pd.DataFrame(dist_to_human)
```

```python
dist_to_human.groupby("Alpha").mean()
```

```python
ax = vtc_axes[2]
for contrast, mean in vtc_count_mean.items():
    if np.isnan(mean):
        continue

    ax.axhline(mean, color=contrast_colors[contrast], linestyle="dashed", lw=1)

sns.lineplot(
    data=patch_df,
    x="Alpha",
    y="Count",
    hue="Category",
    palette=contrast_colors,
    marker=".",
    markersize=8,
    errorbar=None,
    lw=2,
    ax=ax,
)

ax.set_xscale("symlog", linthresh=0.09)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.set_xlim([-0.01, 50])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("# Patches")
ax.legend().remove()
plot_utils.remove_spines(ax)
```

```python
for ax in vtc_axes:
    ax.axvline(0.25, c="k", alpha=0.1)
```

```python
figure_utils.save(vtc_fig, "F04/fgh_vtc_sel_maps")
vtc_fig
```

```python

```

```python

```
