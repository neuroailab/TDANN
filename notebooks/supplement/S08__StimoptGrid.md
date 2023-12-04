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
from tqdm import tqdm
```

```python
import spacetorch.analyses.core as core

from spacetorch.utils import (
    figure_utils,
    generic_utils,
    seed_str,
)
from spacetorch.datasets import floc
from spacetorch.paths import CACHE_DIR
from spacetorch.stimulation import StimulationExperiment
```

```python
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
model_name = "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
model_seed = 0
```

```python
cache_loc = CACHE_DIR / "stimopt_images" / f"{model_name}_seed_{model_seed}"
outputs = generic_utils.load_pickle(cache_loc)
```

```python
models = get_models(model_name, seeds=[model_seed])
```

```python
positions = core.get_positions("simclr_swap")

experiments = {}
for seed, model in models.items():
    experiments[seed] = StimulationExperiment(
        "IT", model, positions, "layer4.0", "layer4.1"
    )

for seed, exp in experiments.items():
    exp.source_tissue.cache_id = f"{model_name}{seed_str(seed)}_layer4.0"
    exp.target_tissue.cache_id = f"{model_name}{seed_str(seed)}"  # not optimal, but the cache assumes layer4.1 so we can omit layer here

    exp.source_tissue.patches = []
    exp.target_tissue.patches = []

    for contrast in tqdm(floc.DOMAIN_CONTRASTS):
        exp.source_tissue.find_patches(contrast)
        exp.target_tissue.find_patches(contrast)
```

```python
# get back experiments
```

```python
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}
```

```python
anchors = np.linspace(1.5 * 7, 8.5 * 7, 5)
cx, cy = np.meshgrid(anchors, anchors)
cx = cx.ravel()
cy = cy.ravel()
```

```python
fig, ax_rows = plt.subplots(figsize=(4, 4), ncols=5, nrows=5)
ax_rows = ax_rows[::-1]
plt.subplots_adjust(wspace=0.01, hspace=0.1)

numbers = [
    21,
    22,
    23,
    24,
    25,
    16,
    17,
    18,
    19,
    20,
    11,
    12,
    13,
    14,
    15,
    6,
    7,
    8,
    9,
    10,
    1,
    2,
    3,
    4,
    5,
]

for idx, (im, ax) in enumerate(zip(outputs, ax_rows.ravel())):
    x, y = cx[idx], cy[idx]
    membership = set()
    for patch in experiments[0].target_tissue.patches:
        if patch.contains(x, y):
            membership.add(patch.contrast.name)

    counter = 0
    for c in membership:
        color = contrast_dict[c].color
        ax.scatter(
            0.1 + (0.1 * counter),
            0.1,
            c=color,
            s=20,
            edgecolor="white",
            lw=1,
            transform=ax.transAxes,
        )
        counter += 1

    t = ax.text(0.1, 0.78, numbers[idx], transform=ax.transAxes, fontsize=6)
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="black"))

    ax.imshow(im)
    ax.axis("off")

figure_utils.save(fig, "S13/stimopt_grid")
```

```python
tissue = experiments[0].target_tissue
```

```python
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(*tissue.positions.T, s=1, c="gray", alpha=0.1, rasterized=True)

for patch in tissue.patches:
    ax.add_patch(patch.to_mpl_poly(alpha=0.3))

for i, (x, y) in enumerate(zip(cx, cy[::-1])):
    text = i + 1
    ax.text(
        x,
        y,
        text,
        alpha=1,
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )

ax.axis("off")
figure_utils.save(fig, "S13/patch_numbers")
```

```python

```
