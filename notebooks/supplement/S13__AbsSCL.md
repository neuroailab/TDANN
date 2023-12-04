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
import pprint
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
```

```python
from spacetorch.datasets import floc
from spacetorch.utils import figure_utils, plot_utils
from spacetorch.utils.vissl.performance import Performance
from spacetorch.paths import CHECKPOINT_DIR, RESULTS_DIR
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
results_path = RESULTS_DIR / "self-sup_v_categorization_loss_results.pkl"
abs_results_path = RESULTS_DIR / "old_scl_loss_results.pkl"
loss_df = pd.read_pickle(results_path)
abs_loss_df = pd.read_pickle(abs_results_path)

combined_df = pd.concat([loss_df, abs_loss_df])
combined_df["Model"].replace("Self-Supervision", "TDANN", inplace=True)
palette = {model: figure_utils.get_color(model) for model in combined_df.Model.unique()}
```

```python
fig, ax = plt.subplots(figsize=(1, 1))

sns.lineplot(
    data=combined_df, x="Iteration", y="Loss", hue="Model", palette=palette, ax=ax
)
ax.set_ylabel("VTC-like Layer\nSpatial Loss")
ax.set_xlabel("Training Epoch")
plot_utils.remove_spines(ax)
ax.legend().remove()
figure_utils.save(fig, "S15/a_loss_curves")
```

# Performance

```python
def parse(name):
    model_type = "unknown"
    if "isoswap_3" in name:
        model_type = "Relative SL"
    elif "old_scl" in name:
        model_type = "Absolute SL"

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

    return spatial_weight, model_type
```

```python
performance_results = {"Alpha": [], "Accuracy": [], "Type": []}

base_dir = CHECKPOINT_DIR / "linear_eval"

for sub in base_dir.iterdir():
    spatial_weight, model_type = parse(sub.stem)
    perf = Performance(spatial_weight, sub / "metrics.json", model_type)
    if model_type == "unknown":
        print(spatial_weight, model_type, perf.best("top_1"))
        continue

    if spatial_weight == 0:
        continue
        for typ in ["Absolute SL", "Relative SL"]:
            performance_results["Alpha"].append(0)
            performance_results["Accuracy"].append(perf.best("top_1"))
            performance_results["Type"].append(typ)
        continue

    performance_results["Alpha"].append(spatial_weight)
    performance_results["Accuracy"].append(perf.best("top_1"))
    performance_results["Type"].append(model_type)
performance_df = pd.DataFrame(performance_results)
```

```python
performance_df.groupby(["Type", "Alpha"]).mean()
```

```python
rel = np.array(
    performance_df.query("Type == 'Relative SL'")
    .groupby("Alpha")
    .mean()
    .reset_index()
    .Accuracy
)
ab = np.array(
    performance_df.query("Type == 'Absolute SL'")
    .groupby("Alpha")
    .mean()
    .reset_index()
    .Accuracy
)
```

```python
pg.wilcoxon(rel, ab)
```

```python
fig, ax = plt.subplots(figsize=(1, 1))

sns.lineplot(
    data=performance_df,
    x="Alpha",
    y="Accuracy",
    hue="Type",
    hue_order=["Absolute SL", "Relative SL"],
    palette={
        "Absolute SL": figure_utils.get_color("Absolute SL"),
        "Relative SL": figure_utils.get_color("Relative SL"),
    },
    marker='.',
    markersize=MARKERSIZE,
    mew=0.5,
    lw=1,
    ax=ax,
)

ax.set_xscale("symlog", linthresh=0.09)
ax.set_xlim([.08, 50])
ax.set_xticks([], minor=True)
ax.set_xticks([0.1, 0.25, 0.5, 1.25, 2.5, 25])
ax.set_xticklabels([0.1, "", "", 1.25, "", 25])
ax.axvline(0.25, c='k', alpha=0.1)

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Categorization\nAccuracy (%)")
ax.legend().remove()
plot_utils.remove_spines(ax)
figure_utils.save(fig, "S15/performance")
```

```python

```

```python
def parse(name):
    core = name.split("_checkpoints")[0]
    if "lw0" in core and "lw01" not in core:
        if core.endswith("lw0"):
            return 0, 0
        seed = int(core.split("_")[-1])
        return 0, seed

    spatial_weight = 0.25
    seed = 0

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

    if "seed" in name:
        seed = int(core.split("_")[-1])

    return spatial_weight, seed
```

```python
matching = sorted(
    SUP_CHECKPOINT_DIR.glob("supervised_spatial_resnet18_swappedon_SineGrating2019*")
) + sorted(SUP_CHECKPOINT_DIR.glob("supervised_resnet18_lw0*"))

pprint.pprint(matching)
```

```python
performance_results = {"Alpha": [], "Accuracy": [], "Seed": []}

for sub in matching:
    if "lfw" in sub.stem or "xswap" in sub.stem:
        continue

    spatial_weight, seed = parse(sub.stem)
    perf = Performance(spatial_weight, sub / "metrics.json", "Supervised")

    performance_results["Alpha"].append(spatial_weight)
    performance_results["Accuracy"].append(perf.best("top_1"))
    performance_results["Seed"].append(seed)
performance_df = pd.DataFrame(performance_results)
```

```python
performance_df
```

```python
fig, ax = plt.subplots(figsize=(1, 1))

sns.barplot(
    data=performance_df,
    x="Alpha",
    y="Accuracy",
    ax=ax,
    color=figure_utils.get_color("Categorization"),
)

ax.set_xticks([0, 2, 4, 6])
ax.set_xticklabels([0, 0.25, 1.25, 25])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Categorization\nAccuracy (%)")
plot_utils.remove_spines(ax)
figure_utils.save(fig, "S15/c_sup_performance")
```
