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
import seaborn as sns
```

```python
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    seed_str,
)
from spacetorch.utils.vissl.performance import Performance

import spacetorch.analyses.core as core
from spacetorch.analyses.alpha import palette_float
from spacetorch.analyses.wiring_length import load_wiring_length_results, LoadSpec
from spacetorch.constants import RNG_SEED
from spacetorch.paths import CHECKPOINT_DIR
from spacetorch.wiring_length import WireLengthExperiment, Shifts
```

```python
figure_utils.set_text_sizes()
rng = default_rng(seed=RNG_SEED)
```

```python
sws = [
    ("_lw0", 0),
    ("_lw01", 0.1),
    ("", 0.25),
    ("_lwx2", 0.5),
    ("_lwx5", 1.25),
    ("_lwx10", 2.5),
    ("_lwx100", 25.0),
]
```

```python
specs = []
for lw_mod, sw in sws:
    specs.extend(
        [
            LoadSpec(
                full_name=f"simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3{lw_mod}",
                df_name_key="Alpha",
                df_name_val=sw,
                seed=seed,
            )
            for seed in range(5)
        ]
    )
```

```python
wl_df = load_wiring_length_results(specs)
```

```python
v1_df = wl_df[wl_df.Layer == "V1"]
it_df = wl_df[wl_df.Layer == "VTC"]

layer_palette = {
    "V1": figure_utils.get_color("layer2.0"),
    "VTC": figure_utils.get_color("layer4.1"),
}
```

```python
fig, axes = plt.subplots(figsize=(1.5, 0.75), ncols=2)
plt.subplots_adjust(wspace=0.6)

sns.lineplot(
    data=v1_df,
    x="Alpha",
    y="Wiring Length",
    marker=".",
    markersize=8,
    color="k",
    ax=axes[0],
)

sns.lineplot(
    data=it_df,
    x="Alpha",
    y="Wiring Length",
    marker=".",
    markersize=8,
    color="k",
    ax=axes[1],
)

for ax in axes:
    ax.set_xscale("symlog", linthresh=0.09)
    ax.set_xticks([], minor=True)
    ax.set_xlim([-0.01, 60])
    ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
    ax.set_xticklabels([0, "", "", 0.5, "", "", 25])
    ax.set_xlabel(r"$\alpha$")
    plot_utils.remove_spines(ax)


axes[0].set_ylabel("Wiring Length (mm)")
axes[1].set_ylabel("")

axes[0].set_title("V1")
axes[1].set_title("VTC")
```

```python
# stats
simple_v1_df = (
    v1_df.groupby(["Seed", "Alpha", "Pattern"]).mean(numeric_only=False).reset_index()
)
display(simple_v1_df.groupby("Alpha").mean(numeric_only=False))
display(simple_v1_df.anova(dv="Wiring Length", between="Alpha"))
display(simple_v1_df.pairwise_tukey(dv="Wiring Length", between="Alpha"))
```

```python
# stats
simple_it_df = it_df.groupby(["Seed", "Alpha", "Pattern"]).mean().reset_index()
display(simple_it_df.groupby("Alpha").mean())
display(simple_it_df.anova(dv="Wiring Length", between="Alpha"))
display(simple_it_df.pairwise_tukey(dv="Wiring Length", between="Alpha"))
```

# Performance

```python
def parse(name):
    spatial_weight = 0.25
    if "lw01" in name:
        spatial_weight = 0.1
    elif "lw0_" in name:
        spatial_weight = 0
    elif "lwx2" in name:
        spatial_weight = 0.5
    elif "lwx5" in name:
        spatial_weight = 1.25
    elif "lwx100" in name:
        spatial_weight = 25
    elif "lwx10" in name:
        spatial_weight = 2.5

    seed = 0
    if "seed" in name:
        seed = int(name.split("_linear_eval_checkpoints")[0].split("seed_")[-1])

    return spatial_weight, seed
```

```python
performance_results = {"Alpha": [], "Seed": [], "Accuracy": []}

base_dir = CHECKPOINT_DIR / "linear_eval"
for sub in base_dir.iterdir():
    if "isoswap_3" not in str(sub):
        continue

    spatial_weight, seed = parse(sub.stem)
    metrics_path = sub / "metrics.json"
    if not metrics_path.exists():
        continue

    perf = Performance(spatial_weight, metrics_path, "SimCLR")

    performance_results["Seed"].append(seed)
    performance_results["Alpha"].append(spatial_weight)
    performance_results["Accuracy"].append(perf.best("top_1"))

performance_df = pd.DataFrame(performance_results)
```

```python
fig, ax = plt.subplots(figsize=(0.75, 0.75))
for sw in sorted(performance_df["Alpha"].unique()):
    for seed in range(5):
        mn_perf = performance_df[
            (performance_df.Alpha == sw) & (performance_df.Seed == seed)
        ]["Accuracy"].mean()
        if np.isnan(mn_perf):
            continue
        mn_v1_wl = v1_df[(v1_df.Alpha == sw) & (v1_df.Seed == seed)][
            "Wiring Length"
        ].mean()
        mn_it_wl = it_df[(it_df.Alpha == sw) & (it_df.Seed == seed)][
            "Wiring Length"
        ].mean()

        total_wl = mn_v1_wl + mn_it_wl

        ax.scatter(
            total_wl,
            mn_perf,
            c=palette_float[sw],
            s=10,
            edgecolor="k",
            lw=0.5,
            alpha=0.8,
            linewidths=0.5,
        )

plot_utils.remove_spines(ax)

ax.set_xlabel("Wiring Length (mm)")
ax.set_ylabel("Categorization\nAccuracy (%)")
```

# Wiring Length Schematic

```python
# load all models for further analysis
def get_models(base, seeds=None, step="latest"):
    ret = {}
    if seeds is None:
        seeds = range(5)

    for seed in seeds:
        try:
            model = core.load_model_from_analysis_config(
                f"{base}{seed_str(seed)}", step=step
            )
            ret[seed] = model
        except Exception:
            continue

    return ret
```

```python
models = {
    "SW_0": get_models(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0"
    ),
    "SW_0pt1": get_models(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw01"
    ),
    "SW_0pt25": get_models(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
    ),
    "SW_0pt5": get_models(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx2"
    ),
    "SW_1pt25": get_models(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx5"
    ),
    "SW_2pt5": get_models(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx10"
    ),
    "SW_25": get_models(
        "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx100"
    ),
}
positions = core.get_positions("simclr_swap")
```

```python
wle = WireLengthExperiment(
    model=models["SW_2pt5"][0],
    layer_positions=positions,
    source_layer="layer4.0",
    target_layer="layer4.1",
    num_patterns=5,
)
```

```python
# for visualization only: custom params

# increase extent of tissue a bit to force a gap between unit in each layer
wle.source_extent = 85
pattern_idx = 4

wl = wle.compute_wl(
    pattern_idx,
    kmeans_dist_thresh=10,
    active_pctile=99.5,
    lims=None,
    direction=Shifts.BOTTOM,
)

source_color = "#AA6D39"
target_color = "#236467"
```

```python
fig, ax = plt.subplots(figsize=(0.4, 1))

# background units
ax.scatter(*wle.tissues["source"].positions.T, c="#DDD", s=1, rasterized=True)
ax.scatter(*wle.tissues["target"].positions.T, c="#DDD", s=1, rasterized=True)

# active units
common = {"s": 2, "rasterized": True, "alpha": 0.5}
ax.scatter(*wle.tissues["source"].passing_pos.T, c=source_color, **common)
ax.scatter(*wle.tissues["target"].passing_pos.T, c=target_color, **common)

# distance to centroids
for layer in ["source", "target"]:
    tiss = wle.tissues[layer]
    lab = tiss.labels

    for curr in np.unique(lab):
        ctr = tiss.centroids[curr]
        match = np.nonzero(lab == curr)[0]

        for idx in match:
            ax.plot(
                [tiss.passing_pos[idx, 0], ctr[0]],
                [tiss.passing_pos[idx, 1], ctr[1]],
                c="k",
                alpha=0.8,
                lw=0.5,
            )

    ax.scatter(*tiss.centroids.T, c="k", s=1, marker="s", zorder=10, rasterized=True)


# plot optimal inter-layer mapping
for row, col in zip(*wle.assignment):
    ax.plot(
        [
            wle.tissues["source"].centroids[row, 0],
            wle.tissues["target"].centroids[col, 0],
        ],
        [
            wle.tissues["source"].centroids[row, 1],
            wle.tissues["target"].centroids[col, 1],
        ],
        c="k",
        lw=1,
    )

# axis beautification
plot_utils.remove_spines(ax)
plot_utils.add_scale_bar(ax, 10)
ax.axis("off")
ax.axhline(-7, linestyle="dashed", c="gray", lw=1)
ax.text(
    -25.5,
    40,
    "Source\n(Layer 8)",
    verticalalignment="center",
    horizontalalignment="center",
    rotation=90,
)
ax.text(
    -25.5,
    -45,
    "Target\n(Layer 9)",
    verticalalignment="center",
    horizontalalignment="center",
    rotation=90,
)
```

# WL by Objective

```python

```

```python
specs = []

# add self-sup
specs.extend(
    [
        LoadSpec(
            full_name="simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
            df_name_key="Model",
            df_name_val="TDANN",
            seed=seed,
        )
        for seed in range(5)
    ]
)

# add categorization
specs.extend(
    [
        LoadSpec(
            full_name="supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019",
            df_name_key="Model",
            df_name_val="Categorization",
            seed=seed,
        )
        for seed in range(5)
    ]
)

specs.extend(
    [
        LoadSpec(
            full_name="simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl",
            df_name_key="Model",
            df_name_val="Absolute SL",
            seed=seed,
        )
        for seed in range(5)
    ]
)
```

```python
wl_df = load_wiring_length_results(specs)
```

```python
bar_palette = {
    "TDANN": figure_utils.get_color("TDANN"),
    "Categorization": figure_utils.get_color("Categorization"),
    "Absolute SL": figure_utils.get_color("Absolute SL"),
}
```

```python
fig, axes = plt.subplots(figsize=(1.5, .75), ncols=2, gridspec_kw={"wspace": 1})
for layer, ax in zip(["V1", "VTC"], axes):
    sub = wl_df[wl_df.Layer == layer].reset_index()
    sns.barplot(sub, x="Model", y="Wiring Length", palette=bar_palette, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.legend().remove()
    plot_utils.remove_spines(ax)
    ax.set_title(layer)

axes[0].set_ylabel("Wiring Length (mm)")
figure_utils.save(fig, "F07/wiring_length")
```

```python
wl_df.head()
```

```python
v1_df = wl_df[wl_df.Layer == "V1"]
vtc_df = wl_df[wl_df.Layer == "VTC"]
```

```python
display(
    pg.mwu(
        v1_df[v1_df.Model == "TDANN"].groupby(["Seed"]).mean()["Wiring Length"],
        v1_df[v1_df.Model == "Categorization"]
        .groupby(["Seed"])
        .mean()["Wiring Length"],
    )
)

display(
    pg.mwu(
        v1_df[v1_df.Model == "TDANN"].groupby(["Seed"]).mean()["Wiring Length"],
        v1_df[v1_df.Model == "Absolute SL"].groupby(["Seed"]).mean()["Wiring Length"],
    )
)
```

```python
display(
    pg.mwu(
        vtc_df[vtc_df.Model == "TDANN"].groupby(["Seed"]).mean()["Wiring Length"],
        vtc_df[vtc_df.Model == "Categorization"]
        .groupby(["Seed"])
        .mean()["Wiring Length"],
    )
)
display(
    pg.mwu(
        vtc_df[vtc_df.Model == "TDANN"].groupby(["Seed"]).mean()["Wiring Length"],
        vtc_df[vtc_df.Model == "Absolute SL"].groupby(["Seed"]).mean()["Wiring Length"],
    )
)
```

```python

```
