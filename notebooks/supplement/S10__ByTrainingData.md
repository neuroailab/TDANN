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
import pandas as pd
import seaborn as sns
import pingouin as pg  # not directly used, but importing this lets us do things like df.anova
from tqdm import tqdm
```

```python
import spacetorch.analyses.core as core
from spacetorch.datasets import floc
from spacetorch.utils import figure_utils, plot_utils, spatial_utils
from spacetorch.analyses.sine_gratings import (
    get_sine_tissue,
    add_sine_colorbar,
    METRIC_DICT,
    get_smoothness_curves,
)
from spacetorch.paths import POSITION_DIR, PROJ_DIR
from spacetorch.analyses.floc import (
    get_floc_tissue,
)
from spacetorch.models.positions import NetworkPositions
from spacetorch.maps import nsd_floc
```

```python
figure_utils.set_text_sizes()
```

# 1. Maps grid

```python
V1_LAYER = "layer2.0"
positions = core.get_positions("simclr_swap")[V1_LAYER]
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
v1_tissues = {
    "ImageNet": get_sine_tissue(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
        positions,
        layer=V1_LAYER,
    ),
    "Ecoset": get_sine_tissue(
        "by_training_data/simclr_ecoset",
        sup_pos,
        layer=V1_LAYER,
    ),
    "Sine Gratings": get_sine_tissue(
        "by_training_data/simclr_spatial_resnet18_trained_sine_gratings",
        sup_pos,
    ),
    "White Noise": get_sine_tissue(
        "by_training_data/white_noise_trained",
        positions,
        layer=V1_LAYER,
    ),
}
```

```python
SCALING = 1.5
ncols = len(v1_tissues)
fig, ax_rows = plt.subplots(ncols=ncols, nrows=3, figsize=(4, 3))

# plot models
for axes, (name, tissue) in zip(ax_rows.T, v1_tissues.items()):
    tissue.set_mask_by_pct_limits([[30, 60], [30, 60]])

    for (metric_name, metric), ax in zip(METRIC_DICT.items(), axes):
        scatter_handle = tissue.make_parameter_map(
            ax,
            metric=metric,
            scale_points=True,
            final_psm=0.07,
            linewidths=0,
            rasterized=True,
        )
        ax.axis("off")

        # add a colorbar if we're in the last column
        if axes[0] == ax_rows[0, -1]:
            cbar = add_sine_colorbar(fig, ax, metric, label=metric.xlabel)
            cbar.ax.set_yticklabels([metric.xticklabels[0], metric.xticklabels[-1]])

    plot_utils.add_scale_bar(axes[-1], width=2)
    plt.subplots_adjust(hspace=0.05, wspace=0.01)
    axes[0].set_title(name)

for ax in ax_rows[:, 0]:
    ax.axis("off")

figure_utils.save(fig, "S07/ori_maps")
```

```python tags=[]
# some quantification of smoothness
v1_smoothness_results = {"Dataset": [], "Iteration": [], "Smoothness": []}

for name, tissue in v1_tissues.items():
    tissue.reset_unit_mask()
    tissue.set_unit_mask_by_ptp_percentile("angles", 75)
    _, curves = get_smoothness_curves(tissue)
    smoothnesses = [spatial_utils.smoothness(curve) for curve in curves]

    for iteration, smoothness in enumerate(smoothnesses):
        v1_smoothness_results["Dataset"].append(name)
        v1_smoothness_results["Iteration"].append(iteration)
        v1_smoothness_results["Smoothness"].append(smoothness)

v1_sm_df = pd.DataFrame(v1_smoothness_results)
```

```python
v1_sm_df.groupby("Dataset").mean()
```

```python
v1_sm_df.anova(dv="Smoothness", between="Dataset")
```

```python
v1_sm_df.pairwise_tukey(dv="Smoothness", between="Dataset")
```

```python
# ori sel
CV_THRESH = 0.6
RESP_THRESH = 1.0

for name, tissue in v1_tissues.items():
    cv = tissue.responses.circular_variance
    mean_responses = tissue.responses._data.mean("image_idx").values
    cv = cv[~np.isnan(cv) & (mean_responses > RESP_THRESH)]
    frac_sel = np.mean(cv < CV_THRESH)
    print(name, frac_sel)
```

```python
VTC_LAYER = "layer4.1"
positions = core.get_positions("simclr_swap")[VTC_LAYER]
sup_pos = NetworkPositions.load_from_dir(
    (
        f"{POSITION_DIR}"
        "/supervised_resnet18"
        "/resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019"
    )
).layer_positions[VTC_LAYER]

# scaling
sup_pos.coordinates = sup_pos.coordinates * 7
sup_pos.neighborhood_width = sup_pos.neighborhood_width * 7
```

```python
tissues = {
    "ImageNet": get_floc_tissue(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
        positions,
        layer=VTC_LAYER,
    ),
    "Ecoset": get_floc_tissue(
        "by_training_data/simclr_ecoset",
        sup_pos,
        layer=VTC_LAYER,
    ),
    "Sine Gratings": get_floc_tissue(
        "by_training_data/simclr_spatial_resnet18_trained_sine_gratings",
        sup_pos,
    ),
    "White Noise": get_floc_tissue(
        "by_training_data/white_noise_trained",
        positions,
        layer=VTC_LAYER,
    ),
    "Unoptimized": get_floc_tissue(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_1",
        positions,
        layer=VTC_LAYER,
        step="random",
    ),
}
```

```python
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
plot_con_order = ["Objects", "Characters", "Places", "Faces", "Bodies"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}
ordered_contrasts = [contrast_dict[c] for c in contrast_order]
```

```python
mat_dir = PROJ_DIR / "nsd_tvals"
subjects = nsd_floc.load_data(mat_dir, domains=contrasts, find_patches=True)
```

```python
import spacetorch.analyses.rsa as rsa
```

```python
human_rsms = []

for subject in subjects:
    for hemi in ["left_hemi", "right_hemi"]:
        human_rsms.append(rsa.get_human_rsm(subject, hemi))
```

```python
rsa_results = {"Name": [], "Sim": [], "Human": []}

for name, tissue in tqdm(tissues.items()):
    rsm = rsa.get_model_rsm(tissue)

    for hidx, human_rsm in enumerate(human_rsms):
        sim = rsa.rsm_similarity(rsm, human_rsm)
        rsa_results["Name"].append(name)
        rsa_results["Sim"].append(sim)
        rsa_results["Human"].append(hidx)

rsa_df = pd.DataFrame(rsa_results)
```

```python
rsa_df.Name.unique()
```

```python
fig, ax = plt.subplots(figsize=(2, 1))
sns.barplot(data=rsa_df, x="Name", y="Sim", color="#444", ax=ax)
plot_utils.remove_spines(ax)
ax.set_xticks([])
ax.set_ylabel(("Similarity to Human" "\n" r"VTC [Kendall's $\tau$]"))
ax.set_xlabel("")
figure_utils.save(fig, "S07/RSA_results")
```

```python
# don't need Unoptimized past this point
tissues.pop("Unoptimized")
```

```python
contrasts = floc.DOMAIN_CONTRASTS[::-1]
```

```python
%%capture
mappable = plt.imshow(np.linspace(-30, 30, 100).reshape(10, 10), cmap='seismic')
```

```python
fig, axes = plt.subplots(ncols=4, nrows=6, figsize=(3, 4.5))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

for col_idx, (ax_col, (name, tissue)) in enumerate(zip(axes.T, tissues.items())):
    for ax, contrast in zip(ax_col, contrasts):
        tissue.make_single_contrast_map(
            ax,
            contrast=contrast,
            final_psm=2e-4,
            vmin=-30,
            vmax=30,
            linewidths=0,
            rasterized=True,
        )
        if col_idx == 0:
            ax.text(
                -0.1,
                0.5,
                contrast.name,
                rotation=90,
                color=contrast.color,
                verticalalignment="center",
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=5,
            )

    ax_col[0].set_title(name, fontsize=5)
    tissue.make_selectivity_map(
        ax_col[-1],
        selectivity_threshold=12,
        foreground_alpha=0.8,
        size_mult=3e-3,
        marker=".",
        linewidths=0,
        rasterized=True,
    )

cax = fig.add_axes([.9, .77, .01, .08])
cb = plt.colorbar(mappable, cax=cax)
cb.set_label(r'$t$-value');
for ax in axes.ravel():
    ax.axis("off")
    plot_utils.add_scale_bar(ax, 10)
figure_utils.save(fig, "S07/sel_maps")
```

```python
vtc_smoothness_results = {
    "Dataset": [],
    "Iteration": [],
    "Smoothness": [],
    "Category": [],
}

for name, tissue in tissues.items():
    for contrast in contrasts:
        _, curves = tissue.category_smoothness(contrast, num_samples=20)
        smoothnesses = [spatial_utils.smoothness(curve) for curve in curves]

        for iteration, smoothness in enumerate(smoothnesses):
            vtc_smoothness_results["Dataset"].append(name)
            vtc_smoothness_results["Iteration"].append(iteration)
            vtc_smoothness_results["Smoothness"].append(smoothness)
            vtc_smoothness_results["Category"].append(contrast.name)


vtc_sm_df = pd.DataFrame(vtc_smoothness_results)
```

```python
vtc_sm_df.groupby("Dataset").mean()
```

```python
for name, tissue in tqdm(tissues.items(), total=len(tissues)):
    tissue.patches = []
    for contrast in contrasts:
        tissue.find_patches(contrast)
```

```python
fig, axes = plt.subplots(ncols=4, figsize=(8, 2))
for ax, (name, tissue) in zip(axes, tissues.items()):
    print(name, len(tissue.patches))
    ax.scatter(*tissue.positions.T, alpha=0.01, c="k", s=1)
    for patch in tissue.patches:
        ax.add_patch(patch.to_mpl_poly())

    ax.set_title(name)
    ax.axis("off")
```

```python
SEL_THRESH = 5

sel_results = {
    "Dataset": [],
    "Category": [],
    "Fraction Selective": [],
}

for name, tissue in tissues.items():
    for contrast in contrasts:
        sel = tissue.responses.selectivity(
            on_categories=contrast.on_categories, off_categories=contrast.off_categories
        )
        frac_sel = np.mean(sel > SEL_THRESH)
        sel_results["Dataset"].append(name)
        sel_results["Category"].append(contrast.name)
        sel_results["Fraction Selective"].append(frac_sel)

sel_df = pd.DataFrame(sel_results)
```

```python
sel_df.groupby("Dataset").mean()
```

```python

```
