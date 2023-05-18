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
import pickle
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
```

```python
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    seed_str,
)
import spacetorch.analyses.core as core
from spacetorch.datasets import floc
from spacetorch.paths import RESULTS_DIR
from spacetorch.models.trunks.resnet import LAYER_ORDER as layers
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_colors = {c.name: c.color for c in contrasts}
```

```python
positions = core.get_positions("simclr_swap")
random_positions = core.get_positions("retinotopic")
```

## Linear Regression

```python
score_dir = RESULTS_DIR / "neural_fits"


def load_score(model, benchmark):
    pth = score_dir / benchmark / f"{model}.pkl"
    if pth.exists():
        with pth.open("rb") as stream:
            score = pickle.load(stream)

        return score
    return None
```

```python
base_name = "simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"

benchmarks = {
    "V1": "tolias.Cadena2017-pls",
    "V4": "dicarlo.MajajHong2015.V4-pls",
    "IT": "dicarlo.MajajHong2015.IT-pls",
}

ve_res = {"Layer": [], "VE": [], "Split": [], "Region": [], "Seed": []}
for region, bmark in benchmarks.items():
    for seed in range(5):
        name = f"{base_name}{seed_str(seed)}"
        varexp = load_score(name, bmark)
        if varexp is None:
            continue
        raw = varexp.raw.raw
        med = raw.median("neuroid")

        for split in np.unique(raw.split):
            for layer in layers:
                ve = float(med.sel(split=split, layer=f"base_model.{layer}"))
                ve_res["Region"].append(region)
                ve_res["Seed"].append(seed)
                ve_res["Split"].append(split)
                ve_res["Layer"].append(layer)
                ve_res["VE"].append(ve)

ve_df = pd.DataFrame(ve_res)
```

```python
# load data from scripts
cv_df = pd.read_pickle(RESULTS_DIR / "supp_layer_selection_cv_df.pkl")
cat_df = pd.read_pickle(RESULTS_DIR / "supp_layer_selection_cat_df.pkl")
```

```python
mosaic = """AAB
            AAC
            """

fig = plt.figure(figsize=(4, 2))
ax_dict = fig.subplot_mosaic(
    mosaic,
    gridspec_kw={
        "wspace": 1.2,
        "hspace": 1.2,
        "width_ratios": [1, 1, 3],
        "height_ratios": [1, 1],
    },
)
axes = {
    "fits": ax_dict["A"],
    "cv": ax_dict["B"],
    "sel": ax_dict["C"],
}

for name, ax in axes.items():
    plot_utils.remove_spines(ax)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_xticklabels([2, "4\nV1-like", 6, "9\nVTC-like"])
```

```python
ax = axes["fits"]
sns.lineplot(data=ve_df, x="Layer", y="VE", hue="Region", ax=ax)
ax.legend(title="", frameon=False)
ax.set_ylabel("Noise-Corrected Variance Explained")
```

```python
ax = axes["cv"]
sns.lineplot(data=cv_df, x="Layer", y="% Selective", color="k", ax=ax)
ax.set_ylabel("% Orientation\nSelective")
```

```python
ax = axes["sel"]
sns.lineplot(
    data=cat_df,
    x="Layer",
    y="% Selective",
    hue="Contrast",
    palette=contrast_colors,
    ax=ax,
)
plot_utils.remove_spines(ax)
ax.set_ylabel("% Category\nSelective")
ax.set_xlabel("Layer")
ax.legend().remove()
```

```python
figure_utils.save(fig, "S02/layer_sel")
fig
```

```python

```
