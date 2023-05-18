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
from typing import Optional
from pathlib import Path
import pickle
```

```python
from matplotlib.cm import BuPu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import linregress
import seaborn as sns
from tqdm import tqdm
```

```python
from spacetorch.analyses.alpha import palette_float
from spacetorch.analyses.dimensionality import effective_dim
from spacetorch.models.trunks.resnet import LAYER_ORDER
from spacetorch.paths import (
    CHECKPOINT_DIR,
    SUP_CHECKPOINT_DIR,
    RESULTS_DIR,
    _base_fs,
)
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    seed_str,
)

```

```python
figure_utils.set_text_sizes()
MARKERSIZE = 6
```

```python
palette = {
    "TDANN": figure_utils.get_color("TDANN"),
    "Categorization": figure_utils.get_color("Categorization"),
    "Absolute SL": figure_utils.get_color("Absolute SL"),
    'tdann': figure_utils.get_color('TDANN'),
    'categorization': figure_utils.get_color('Categorization'),
    'absolute_sl': figure_utils.get_color('Absolute SL'),
}
```

## Panel A: Linear regression

```python
# setting various params
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

cfg_lookup = {
    "TDANN": "simclr",
    "Categorization": "supswap_supervised",
    "Absolute SL": "simclr",
}
regions = [("V1", "layer2.0"), ("IT", "layer4.1")]
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
        checkpoint_dir = checkpoint_dirs[base_name]

        for lw_idx, lw_modifier in enumerate(LW_MODS):
            for seed in range(5):
                name = f"{base}{lw_modifier}{seed_str(seed)}"
                full = checkpoint_dir / f"{name}_checkpoints"

                # load lin reg
                varexp = load_score(name, region)
                if varexp is None:
                    continue
                ve = parse_ve(varexp, layer)

                results["Region"].append(region)
                results["Objective"].append(base_name)
                results["Variance Explained"].append(ve)
                results["Model Seed"].append(seed)
                results["Alpha"].append(alpha_lookup(lw_modifier))
```

```python
# add TDANN alpha = 0
for region, layer in regions:
    for seed in range(5):
        name = f"simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0{seed_str(seed)}"

        # load lin reg
        varexp = load_score(name, region)
        if varexp is None:
            continue

        ve = parse_ve(varexp, layer)

        results["Region"].append(region)
        results["Objective"].append("TDANN")
        results["Variance Explained"].append(ve)
        results["Model Seed"].append(seed)
        results["Alpha"].append(0)
```

```python
linreg_df = pd.DataFrame(results)
```

```python
vtc_df = linreg_df[linreg_df.Region == 'IT']
vtc_df = vtc_df[vtc_df.Alpha > 0]

fig, ax = plt.subplots(figsize=(1, 1))
sns.lineplot(
    data=vtc_df,
    x='Alpha',
    y='Variance Explained',
    hue='Objective',
    hue_order=['Categorization', 'Absolute SL', 'TDANN'],
    lw=0.5,
    palette=palette,
    marker=".",
    markersize=MARKERSIZE,
    mew=.4
)
ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([0.08, 30])
ax.set_xticks([], minor=True)
ax.set_xticks([ 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([ 0.1, "", "", 1.25, "", 25])
ax.set_xlabel(r"$\alpha$")
ax.axvline(0.25, c="k", alpha=0.1)
ax.set_ylim([0, 0.65])
ax.legend().remove()
plot_utils.remove_spines(ax)
ax.set_ylabel('Variance Explained (IT)')
figure_utils.save(fig, "F06/it_linreg")
```

## Panel B: One-to-one corrs

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
ax.set_ylabel("1-to-1 Unit-to-Voxel\nCorrelation")
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_ylim([0, 0.5])
ax.legend().remove()

plot_utils.remove_spines(ax)
figure_utils.save(fig, "F06/sub2sub")
```

## Panel C: Effective dim

```python
bases = {
    "tdann": "NSD/simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
    "categorization": "NSD/supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019",
    "absolute_sl": "NSD/simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl",
}
```

```python
def load(npz):
    file = np.load(npz)
    return {
        "eigvals": file["eigvals"],
        "layer": file["layer"][()],
        "seed": file["seed"][()],
        "spatial_weight": file["spatial_weight"][()],
        "model_name": file["model_name"][()],
    }
```

```python
results = {
    "Spatial Weight": [],
    "Layer": [],
    "Layer Index": [],
    "Seed": [],
    "Effective Dim": [],
    "Model Type": [],
}

for model_type, base in bases.items():
    res_dir = RESULTS_DIR / "eigvals__10000_images__spatial_mp" / base
    paths = list(res_dir.rglob("*.npz"))

    for path in paths:
        res = load(path)
            
        results["Spatial Weight"].append(res["spatial_weight"])
        results["Layer"].append(res["layer"])
        results["Layer Index"].append(LAYER_ORDER.index(res["layer"]) + 1)
        results["Seed"].append(res["seed"])
        results["Effective Dim"].append(effective_dim(res["eigvals"]))
        results["Model Type"].append(model_type)

df = pd.DataFrame(results)
```

```python
sub_df = df[(df['Spatial Weight'] == 0) | (df['Spatial Weight'] == 0.25)]
sub_df = sub_df[sub_df.Layer == "layer4.1"]
```

```python
lh_vtc_human_values = [20.81495681, 15.47835614, 17.94187346, 14.36792843, 12.5535002, 25.1976215, 17.25583616, 21.00962163]
rh_vtc_human_values = [14.76887836, 15.98228787, 13.13239104, 13.81466669, 16.38861376,
 16.18196947, 16.26466094, 16.27532075]
human_mean = np.mean(lh_vtc_human_values + rh_vtc_human_values)
print(human_mean)
```

```python
model_order = ['tdann', 'categorization', 'absolute_sl']
```

```python
fig, ax = plt.subplots(figsize=(1, 1))
sns.barplot(
    data=sub_df,
    x='Spatial Weight',
    hue='Model Type',
    palette=palette,
    y='Effective Dim',
    ax=ax
)

sns.stripplot(
    data=sub_df,
    x='Spatial Weight',
    hue='Model Type',
    dodge=True,
    y='Effective Dim',
    palette={'tdann': '#ccc', 'categorization': '#ccc', 'absolute_sl': '#ccc'},
    edgecolor='k',
    linewidth=.5,
    jitter=.1,
    size=1.5,
    ax=ax,
)
ax.axhline(human_mean, ls='dashed', c=figure_utils.get_color('Human'), lw=1)
ax.set_xticks([0, 1])
ax.set_xticklabels([
    r'$\alpha = 0$',
    r'$\alpha = 0.25$',
])
ax.set_xlabel('')
ax.set_ylabel('Effective Dimensionality')
plot_utils.remove_spines(ax)
ax.legend().remove()
figure_utils.save(fig, "F06/effdim_bars")
```

```python
fig, ax = plt.subplots(figsize=(1, 1))
sns.lineplot(
    data=df[df.Layer == "layer4.1"],
    x='Spatial Weight',
    y='Effective Dim',
    hue='Model Type',
    hue_order=["absolute_sl", "categorization", "tdann"],
    palette=palette,
    marker=".",
    markersize=MARKERSIZE,
    ax=ax
)
ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 30])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel('Effective Dimensionality')
ax.axvline(0.25, c="k", alpha=0.1)
ax.legend().remove()
plot_utils.remove_spines(ax)

ax.axhline(human_mean, ls='dashed', c=figure_utils.get_color('Human'), lw=1)
figure_utils.save(fig, "F06/effdim_by_model_by_alpha")
```

```python
76.8 / 27.7
```

```python
print(df[df.Layer == 'layer4.1'].groupby(["Model Type", "Spatial Weight", "Seed"]).mean())
```

```python
# stats
sub_df.anova(dv='Effective Dim', between=['Model Type', 'Spatial Weight'])
```

```python
layer_palette = {
    LAYER_ORDER[0]: BuPu(0.1),
    LAYER_ORDER[1]: BuPu(0.22),
    LAYER_ORDER[2]: BuPu(0.34),
    LAYER_ORDER[3]: BuPu(0.46),
    LAYER_ORDER[4]: BuPu(0.58),
    LAYER_ORDER[5]: BuPu(0.70),
    LAYER_ORDER[6]: BuPu(0.82),
    LAYER_ORDER[7]: BuPu(0.94),
}
```

```python
%%capture
mappable = plt.imshow(np.linspace(0, 1, 25).reshape(5, 5), cmap="BuPu")
```

```python
fig, ax = plt.subplots(figsize=(1, 1))
sns.lineplot(
    data=df[df['Model Type'] == 'tdann'],
    x='Spatial Weight',
    y='Effective Dim',
    hue='Layer',
    palette=layer_palette,
    marker=".",
    markersize=MARKERSIZE,
    ax=ax
)
ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 30])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])
ax.set_xlabel(r"$\alpha$")
ax.set_yscale("log")
ax.set_ylabel('Effective Dimensionality')
ax.set_yticks([1, 10])
ax.axvline(0.25, c="k", alpha=0.1)
ax.legend().remove()
plot_utils.remove_spines(ax)

cax = fig.add_axes([0.95, 0.15, 0.04, 0.65])
clab = fig.colorbar(mappable, cax=cax)
clab.set_ticks([0, 1])
clab.set_ticklabels(["Layer 1", "Layer 8"])
figure_utils.save(fig, "F06/effdim_by_layer_by_alpha")
```

```python
# maybe to supplement? scree plots and stuff
```

```python
scree_results = {
    "Spatial Weight": [],
    "Layer": [],
    "Layer Index": [],
    "Seed": [],
    "PC Index": [],
    "Variance": [],
    "Model Type": [],
}

for model_type, base in bases.items():
    res_dir = RESULTS_DIR / "eigvals__10000_images__spatial_mp" / base
    paths = list(res_dir.rglob("*.npz"))

    for path in tqdm(paths):
        res = load(path)
        eigvals = res["eigvals"]

        if res["layer"] != "layer4.1":
            continue
            
        if res["spatial_weight"] != 0.25:
            continue

        for i, v in enumerate(eigvals[:128]):
            scree_results["Spatial Weight"].append(res["spatial_weight"])
            scree_results["Layer"].append(res["layer"])
            scree_results["Layer Index"].append(LAYER_ORDER.index(res["layer"]) + 1)
            scree_results["Seed"].append(res["seed"])
            scree_results["PC Index"].append(i + 1)
            scree_results["Variance"].append(v)
            scree_results["Model Type"].append(model_type)

scree_df = pd.DataFrame(scree_results)
```

```python
fig, ax= plt.subplots(figsize=(1, 1))
sns.lineplot(data=scree_df, ax=ax, x='PC Index', y='Variance', lw=1, hue='Model Type', palette=palette)

ax.set_xscale('log')
ax.set_yscale('log')
plot_utils.remove_spines(ax)
ax.legend().remove()
figure_utils.save(fig, "SYY/scree_by_model_type")
```

```python
res_dir = RESULTS_DIR / "eigvals__10000_images__spatial_mp" / bases["tdann"]
paths = list(res_dir.rglob("*.npz"))

scree_results = {
    "Spatial Weight": [],
    "Seed": [],
    "PC Index": [],
    "Variance": [],
    "Layer": [],
}

for path in tqdm(paths):
    res = load(path)
    eigvals = res["eigvals"]

    for i, v in enumerate(eigvals[:128]):
        scree_results["Spatial Weight"].append(res["spatial_weight"])
        scree_results["Seed"].append(res["seed"])
        scree_results["PC Index"].append(i + 1)
        scree_results["Variance"].append(v)
        scree_results["Layer"].append(res["layer"])

scree_df = pd.DataFrame(scree_results)
```

```python
fig, axes = plt.subplots(figsize=(5, 2.5), ncols=4, nrows=2, gridspec_kw={'hspace': 0.8, 'wspace': 0.5})
for ax, (layer_idx, layer) in zip(axes.ravel(), enumerate(LAYER_ORDER)):
    matching = scree_df[scree_df.Layer == layer]
    sns.lineplot(data=matching, ax=ax, x='PC Index', y='Variance', lw=1, hue='Spatial Weight', palette=palette_float)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([1, None])
    ax.legend().remove()
    plot_utils.remove_spines(ax)
    ax.set_title(f"Layer {layer_idx + 1}")
figure_utils.save(fig, "SYY/scree_by_alpha_by_layer_type")
```

### Supplement: PLE

```python
def estimate_ple(eigvals: np.ndarray, component_start: int = 0, components_keep: Optional[int] = None) -> float:
    fitted_eigvals = eigvals[component_start:components_keep]
    xs = np.log(np.arange(1, len(fitted_eigvals) + 1))
    ys = np.log(fitted_eigvals)
    
    fit_result = linregress(xs, ys)
    return abs(fit_result.slope)
```

```python
ple_results = {
    "Spatial Weight": [],
    "Seed": [],
    "PLE": [],
    "Model Type": [],
}

for model_type, base in bases.items():
    res_dir = RESULTS_DIR / "eigvals__10000_images__spatial_mp" / base
    paths = list(res_dir.rglob("*.npz"))

    for path in paths:
        res = load(path)
        if res["layer"] != "layer4.1":
            continue
            
        eigvals = res["eigvals"]
        ple = estimate_ple(eigvals, component_start=1, components_keep=50)
            
        ple_results["Spatial Weight"].append(res["spatial_weight"])
        ple_results["Seed"].append(res["seed"])
        ple_results["PLE"].append(ple)
        ple_results["Model Type"].append(model_type)

ple_df = pd.DataFrame(ple_results)
```

```python
fig, ax = plt.subplots(figsize=(1, 1))
sns.lineplot(
    data=ple_df,
    x='Spatial Weight',
    y='PLE',
    hue='Model Type',
    hue_order=["absolute_sl", "categorization", "tdann"],
    palette=palette,
    marker=".",
    markersize=MARKERSIZE,
    ax=ax
)
ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([-0.01, 30])
ax.set_xticks([], minor=True)
ax.set_xticks([0, 0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0, 0.1, "", "", 1.25, "", 25])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel('Power Law Exponent')
ax.axvline(0.25, c="k", alpha=0.1)
ax.legend().remove()
plot_utils.remove_spines(ax)

ax.axhline(1.0, ls='dashed', c=figure_utils.get_color('Human'), lw=1)
figure_utils.save(fig, "F06/ple_by_model")
```

```python

```
