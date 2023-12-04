---
jupyter:
  jupytext:
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
from pathlib import Path
import pickle
import pprint
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    get_smoothness_curves,
)
from spacetorch.analyses.floc import get_floc_tissue
from spacetorch.analyses.wiring_length import LoadSpec, load_wiring_length_results
from spacetorch.datasets import floc, DatasetRegistry
from spacetorch.maps.pinwheel_detector import PinwheelDetector
from spacetorch.maps import nsd_floc

from spacetorch.paths import (
    CHECKPOINT_DIR,
    PROJ_DIR,
    SUP_CHECKPOINT_DIR,
    RESULTS_DIR,
    _base_fs,
)
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}
pprint.pprint(contrasts)
```

```python
face_contrast = contrast_dict["Faces"]
```

```python
# setting various params
SEEDS = range(5)
LW_MODS = ["_lw01", "", "_lwx2", "_lwx5", "_lwx10", "_lwx100"]

score_dir = RESULTS_DIR / "neural_fits"
bases = {
    "TDANN": "simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
    "Categorization": "supervised_spatial_resnet18_swappedon_SineGrating2019",
    "Absolute SL": "simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl",
}
checkpoint_dirs = {
    "TDANN": CHECKPOINT_DIR,
    "Categorization": SUP_CHECKPOINT_DIR,
    "Absolute SL": CHECKPOINT_DIR,
}

pos_lookup = {
    "TDANN": core.get_positions("simclr_swap"),
    "Categorization": core.get_positions("supervised_swap"),
    "Absolute SL": core.get_positions("simclr_swap"),
}

cfg_lookup = {
    "TDANN": "simclr",
    "Categorization": "supswap_supervised",
    "Absolute SL": "simclr",
}
regions = [("V1", "layer2.0"), ("IT", "layer4.1")]

bin_edges = np.linspace(0, 60, 10)
midpoints = array_utils.midpoints_from_bin_edges(bin_edges)
```

```python
def load_score(model, region):
    benchmark = (
        "dicarlo.MajajHong2015.IT-pls" if region == "IT" else "tolias.Cadena2017-pls"
    )
    pth = score_dir / benchmark / f"{model}.pkl"
    if pth.exists():
        with pth.open("rb") as stream:
            score = pickle.load(stream)

        return score
    return None


def parse_ve(varexp, layer):
    ves = []
    raw = varexp.raw.raw
    med = raw.median("neuroid")

    for split in np.unique(raw.split):
        ve = float(med.sel(split=split, layer=f"base_model.{layer}"))
        ves.append(ve)

    return np.mean(ves)
```

```python
results = {
    "Objective": [],
    "Smoothness": [],
    "Variance Explained": [],
    "Region": [],
    "Model Seed": [],
    "Alpha": [],
}
```

```python
def alpha_lookup(alpha_str):
    if alpha_str == "":
        return 0.25
    lookup = {"_lw01": 0.1, "_lwx2": 0.5, "_lwx5": 1.25, "_lwx10": 2.5, "_lwx100": 25.0}
    return lookup[alpha_str]
```

```python
for region, layer in regions:
    for base_name, base in bases.items():
        positions = pos_lookup[base_name][layer]
        checkpoint_dir = checkpoint_dirs[base_name]

        for lw_idx, lw_modifier in enumerate(LW_MODS):
            for seed_idx, seed in enumerate(SEEDS):
                seed_str = f"_seed_{seed}" if seed > 0 else ""
                name = f"{base}{lw_modifier}{seed_str}"
                full = checkpoint_dir / f"{name}_checkpoints"

                # load lin reg
                varexp = load_score(name, region)
                if varexp is None:
                    continue
                ve = parse_ve(varexp, layer)

                if region == "IT":
                    tissue = get_floc_tissue(
                        f"{cfg_lookup[base_name]}/{name}", positions, layer="layer4.1"
                    )
                    _, curves = tissue.category_smoothness(
                        face_contrast, num_samples=20, bin_edges=bin_edges
                    )
                elif region == "V1":
                    tissue = get_sine_tissue(
                        f"{cfg_lookup[base_name]}/{name}", positions, layer="layer2.0"
                    )

                    _, curves = get_smoothness_curves(tissue)

                mean_smoothness = np.mean(
                    [spatial_utils.smoothness(curve) for curve in curves]
                )

                results["Region"].append(region)
                results["Objective"].append(base_name)
                results["Smoothness"].append(mean_smoothness)
                results["Variance Explained"].append(ve)
                results["Model Seed"].append(seed)
                results["Alpha"].append(alpha_lookup(lw_modifier))
```

```python
# add nonspatial simclr
for region, layer in regions:
    for seed_idx, seed in enumerate(range(5)):
        seed_str = f"_seed_{seed}" if seed > 0 else ""
        name = (
            f"simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0{seed_str}"
        )

        # load lin reg
        varexp = load_score(name, region)
        if varexp is None:
            continue

        # ve = varexp.sel(layer=f"base_model.{layer}")[0]
        ve = parse_ve(varexp, layer)
        if region == "IT":
            tissue = get_floc_tissue(
                f"simclr/{name}",
                pos_lookup["TDANN"][layer],
                layer="layer4.1",
            )
            _, curves = tissue.category_smoothness(
                face_contrast, num_samples=20, bin_edges=bin_edges
            )
        elif region == "V1":
            tissue = get_sine_tissue(
                f"simclr/{name}",
                pos_lookup["TDANN"][layer],
                layer="layer2.0",
            )

            _, curves = get_smoothness_curves(tissue)

        mean_smoothness = np.mean([spatial_utils.smoothness(curve) for curve in curves])

        results["Region"].append(region)
        results["Objective"].append(
            "TDANN"
        )  # this isn't totally true, but it gives us the alpha = 0 data points
        results["Smoothness"].append(mean_smoothness)
        results["Variance Explained"].append(ve)
        results["Model Seed"].append(seed)
        results["Alpha"].append(0)
```

```python
df = pd.DataFrame(results)
```

```python
df = df.rename(columns={"Alpha": r"$\alpha$"})
palette = {
    "TDANN": figure_utils.get_color("TDANN"),
    "Categorization": figure_utils.get_color("Categorization"),
    "Absolute SL": figure_utils.get_color("Absolute SL"),
}
```

```python
xlabel_lookup = {"V1": "OPM Smoothness", "IT": "Face Selectivity Smoothness"}
```

```python
# actually plot log alpha as the sizes to make the range smaller
df["Log Alpha"] = np.log(1 + df[r"$\alpha$"])

# make a scatterplot for each region
for region, layer in regions:
    region_df = df[df.Region == region]

    fig = plt.figure(figsize=(1, 1))
    main_width = 0.8
    main_ax = fig.add_axes([0, 0, main_width, main_width])
    smooth_joint_ax = fig.add_axes([0, main_width + 0.01, main_width, 0.1])
    ve_ax = fig.add_axes([main_width, 0, 0.1, main_width])

    # plot the points
    sns.scatterplot(
        data=region_df,
        x="Smoothness",
        y="Variance Explained",
        hue="Objective",
        palette=palette,
        style="Objective",
        legend=None,
        size="Log Alpha",
        sizes=(10, 30),
        rasterized=True,
        alpha=0.7,
        ax=main_ax,
    )

    # plot the joint KDE plots
    joint_kwargs = {
        "data": region_df,
        "linewidth": 1,
        "fill": False,
        "legend": None,
        "palette": palette,
        "hue": "Objective",
        "common_norm": True,
        "bw_adjust": 0.8,
    }
    sns.kdeplot(x="Smoothness", ax=smooth_joint_ax, **joint_kwargs)
    sns.kdeplot(y="Variance Explained", ax=ve_ax, **joint_kwargs)
    smooth_joint_ax.set_xlim(main_ax.get_xlim())
    ve_ax.set_ylim(main_ax.get_ylim())
    ve_ax.spines["bottom"].set_visible(False)
    smooth_joint_ax.spines["left"].set_visible(False)

    # axis formatting
    for ax in [smooth_joint_ax, ve_ax]:
        plot_utils.remove_spines(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    plot_utils.remove_spines(main_ax)
    main_ax.set_xlabel(xlabel_lookup[region])
    main_ax.set_ylabel(f"Var Exp ({region})")
    figure_utils.save(fig, f"S16/scatter_{region}")
```

```python
v1_df = df[df.Region == "V1"]
it_df = df[df.Region == "IT"]
```

```python
tdann_passing = v1_df[(v1_df.Objective == "TDANN") & (v1_df[r"$\alpha$"] != 25)]
categorization_passing = v1_df[
    (v1_df.Objective == "Categorization") & (v1_df[r"$\alpha$"] != 25)
]
abs_sl_passing = v1_df[(v1_df.Objective == "Absolute SL") & (v1_df[r"$\alpha$"] != 25)]

display(
    pg.mwu(
        tdann_passing.groupby(["Model Seed"]).mean()["Variance Explained"],
        categorization_passing.groupby(["Model Seed"]).mean()["Variance Explained"],
    )
)
display(
    pg.mwu(
        tdann_passing.groupby(["Model Seed"]).mean()["Variance Explained"],
        abs_sl_passing.groupby(["Model Seed"]).mean()["Variance Explained"],
    )
)
```

```python
tdann_passing = it_df[(it_df.Objective == "TDANN") & (it_df[r"$\alpha$"] != 25)]
categorization_passing = it_df[
    (it_df.Objective == "Categorization") & (it_df[r"$\alpha$"] != 25)
]
abs_sl_passing = it_df[(it_df.Objective == "Absolute SL") & (it_df[r"$\alpha$"] != 25)]

display(
    pg.mwu(
        tdann_passing.groupby(["Model Seed"]).mean()["Variance Explained"],
        categorization_passing.groupby(["Model Seed"]).mean()["Variance Explained"],
    )
)
display(
    pg.mwu(
        tdann_passing.groupby(["Model Seed"]).mean()["Variance Explained"],
        abs_sl_passing.groupby(["Model Seed"]).mean()["Variance Explained"],
    )
)
```

```python

```
