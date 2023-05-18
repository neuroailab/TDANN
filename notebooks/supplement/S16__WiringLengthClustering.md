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
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pandas as pd
import pingouin as pg
from scipy.spatial.distance import pdist
from scipy.stats import scoreatpercentile
import seaborn as sns
from tqdm import tqdm
```

```python
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    array_utils,
    generic_utils,
    seed_str,
)

import spacetorch.analyses.core as core
from spacetorch.analyses.alpha import name_lookup, palette
from spacetorch.analyses.imagenet import create_imnet_tissue
from spacetorch.constants import RNG_SEED
from spacetorch.paths import CACHE_DIR
```

```python
figure_utils.set_text_sizes()
rng = default_rng(seed=RNG_SEED)
```

# distance of activated units to natural stimuli

```python
# load from cache if we can
cache_probe = CACHE_DIR / "wl_dist_tissues.pkl"
if cache_probe.is_file():
    dist_tissues = generic_utils.load_pickle(cache_probe)
else:
    VTC_LAYER = "layer4.1"

    models = {
        "SW_0": "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0",
        "SW_0pt1": "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw01",
        "SW_0pt25": "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
        "SW_0pt5": "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx2",
        "SW_1pt25": "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx5",
        "SW_2pt5": "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx10",
        "SW_25": "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx100",
        "Unoptimized": "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx2",
    }

    dist_tissues = {}
    for name, base_model_name in models.items():
        dist_tissues[name] = {}
        for seed in range(5):
            model_name = f"{base_model_name}{seed_str(seed)}"

            features = core.get_features_from_model(
                model_name,
                "ImageNet",
                layers=[VTC_LAYER],
                max_batches=2,
                batch_size=32,
                step="random" if name == "Unoptimized" else "latest",
            )

            position_dict = core.get_positions(
                "retinotopic" if name == "Unoptimized" else "simclr_swap"
            )

            tissue = create_imnet_tissue(features, position_dict[VTC_LAYER].coordinates)
            dist_tissues[name][seed] = tissue
    generic_utils.write_pickle(cache_probe, dist_tissues)
```

```python
dist_results = {"Model": [], "Image": [], "Distance": [], "Seed": []}

for name, seeds in tqdm(dist_tissues.items()):
    for seed, tissue in seeds.items():
        features = tissue.responses._data.values
        pos = tissue._positions

        for image_idx, image_features in enumerate(features):
            # get positions of the top 5% most active units
            p95 = scoreatpercentile(image_features, 95)
            passing = np.nonzero(image_features >= p95)[0]
            passing = np.random.choice(passing, size=(200,), replace=False)
            passing_pos = pos[passing]

            distances = pdist(passing_pos)
            for d in distances:
                dist_results["Distance"].append(d)
                dist_results["Image"].append(image_idx)
                dist_results["Model"].append(name)
                dist_results["Seed"].append(seed)

dist_df = pd.DataFrame(dist_results)
```

```python
order = [
    "SW_0",
    "SW_0pt1",
    "SW_0pt25",
    "SW_0pt5",
    "SW_1pt25",
    "SW_2pt5",
    "SW_25",
    "Unoptimized",
]
```

```python
# this takes a bit of time since there's a ton of data

fig, ax = plt.subplots(figsize=(1.75, 1))
ax = sns.kdeplot(
    data=dist_df,
    hue="Model",
    x="Distance",
    lw=1,
    hue_order=order,
    palette=palette,
    fill=False,
    ax=ax,
)
ax.set_xlabel("Distance of Activated Units (mm)")
ax.set_ylabel("Frequency", labelpad=-5)
ax.set_xlim([-1, 70])
ax.set_yticks([0, 0.006])
ax.legend().remove()
plot_utils.remove_spines(ax)
```

## Silhouette of Activated Units

```python
%%capture
# create fake mappable
mappable = plt.imshow(np.linspace(0, 1, 25).reshape(5, 5), cmap="gray_r")
```

```python
PCTILE = 99
IMAGE_IDX = 0

fig, axes = plt.subplots(figsize=(4.5, 4.2 / 7), ncols=len(dist_tissues) - 1)
cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])

for idx, (ax, (name, seeds)) in enumerate(zip(axes, dist_tissues.items())):
    tissue = seeds[0]
    features = tissue.responses._data.values
    pos = tissue._positions

    # find the positions of the most active units
    image_features = features[IMAGE_IDX]
    p99 = scoreatpercentile(image_features, PCTILE)
    passing = np.nonzero(image_features >= p99)[0]
    passing_pos = pos[passing]
    passing_feat = image_features[passing]

    ax.scatter(
        *pos.T,
        c="k",
        alpha=array_utils.norm(image_features) ** 2,
        s=1,
        rasterized=True,
        linewidths=0,
    )
    ax.set_xlim([-1, 71])
    ax.set_ylim([-1, 71])
    ax.axis("off")
    plot_utils.add_scale_bar(ax, 10)

    ax.set_title(name_lookup[name], fontdict={"color": palette[name], "size": 6})

cb = plt.colorbar(mappable=mappable, cax=cax, ticks=[0, 1])
cb.ax.set_yticklabels(["min", "max"])
cb.set_label("Response\nMagnitude (a.u.)")
```

# Clusterness

```python
def ripley(pos, thresh):
    """
    Clusterness is the total number of near neighbors around any point

    Given a pairwise distance matrix D, we go row by row and count the number of
    entries in each row that are below a specified threshold.
    This reduces to a very simple count over the whole matrix.

    Intuitively, if units are uniformly spaced, there will be no very-near-neighbors.
    If units are highly clumped, then most of the distance matrix will be tiny values
    """
    dp = pdist(pos).ravel()
    return np.count_nonzero(dp < thresh)
```

```python
ripley_results = {
    "Model": [],
    "Image": [],
    "Clusterness": [],
    "Random Iter": [],
    "Seed": [],
}

max_extent = max(np.ptp(dist_tissues["SW_0"][0]._positions, axis=0))
bin_width = 10

for name, seeds in tqdm(dist_tissues.items()):
    for seed, tissue in seeds.items():
        features = tissue.responses._data.values
        pos = tissue._positions

        for image_idx, image_features in enumerate(features):
            # get positions of the top 5% most active units
            p95 = scoreatpercentile(image_features, 95)
            passing = np.nonzero(image_features >= p95)[0]

            for random_iter in range(10):
                random = np.random.choice(
                    np.arange(len(pos)), size=(200,), replace=False
                )
                passing = np.random.choice(passing, size=(200,), replace=False)

                passing_pos = pos[passing]
                random_pos = pos[random]

                true_rip = ripley(passing_pos, bin_width)
                shuf_rip = ripley(random_pos, bin_width)

                if shuf_rip == 0:
                    continue
                ratio = true_rip / shuf_rip

                ripley_results["Clusterness"].append(ratio)
                ripley_results["Image"].append(image_idx)
                ripley_results["Model"].append(name)
                ripley_results["Random Iter"].append(random_iter)
                ripley_results["Seed"].append(seed)
```

```python
ripley_df = pd.DataFrame(ripley_results)
```

```python
fig, ax = plt.subplots(figsize=(1.5, 1))
ax = sns.barplot(
    data=ripley_df, x="Model", y="Clusterness", order=order, palette=palette, ax=ax
)
plot_utils.remove_spines(ax)
ax.set_xticklabels([0, "", 0.25, "", 1.25, "", 25, "U"])
ax.set_xlabel(r"$\alpha$")
ax.axhline(1.0, linestyle="dashed", c="gray")
```

```python
# average_over_random_iters
sub_avg = (
    ripley_df.groupby(["Model", "Seed"])
    .mean()
    .reset_index()
    .drop(columns=["Random Iter", "Image"])
)
```

```python
sub_avg.anova(dv="Clusterness", between="Model")
```

```python
pg.pairwise_tukey(data=sub_avg, dv="Clusterness", between="Model").round(7)
```

```python

```
