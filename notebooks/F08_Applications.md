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
import pprint
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pingouin as pg
from scipy.stats import scoreatpercentile
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
```

```python
from spacetorch.utils import (
    figure_utils,
    plot_utils,
    array_utils,
    seed_str,
    gpu_utils,
    generic_utils,
)

import spacetorch.analyses.core as core
from spacetorch.bmi.pattern import Pattern
from spacetorch.bmi.opt import optimize, OptParams, ObjectiveParams
from spacetorch.datasets import floc
from spacetorch.models.trunks.resnet import LAYER_ORDER
from spacetorch.paths import CACHE_DIR, RESULTS_DIR
from spacetorch.stimulation import StimulationExperiment
from spacetorch.feature_extractor import FeatureExtractor
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}

# redefine in a custom order
plot_con_order = ["Objects", "Characters", "Places", "Bodies", "Faces"]
contrasts = [contrast_dict[n] for n in plot_con_order]
pprint.pprint(contrasts)
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
models = get_models(model_name)
```

# Functional Alignment

```python
VTC_LAYER = "layer4.1"
positions = core.get_positions("simclr_swap")
```

```python
experiments = {}
for seed, model in models.items():
    experiments[seed] = StimulationExperiment(
        "IT", model, positions, "layer4.0", "layer4.1"
    )
```

```python
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
def norm_bump(x):
    """For visualization, normalize a bump activity to [0, 1], exaggerating higher values"""
    if np.max(x) == 0:
        return x

    x_bump = (x * np.max(x)) ** 1.2
    return x_bump / np.max(x_bump)
```

```python
line_kwargs = {"linestyle": "dashed", "c": "gray", "lw": 1}
point_size = 1
exp = experiments[0]
stim_sites = [[49, 45.5], [35, 28]]
sigma = 5.25


for site_idx, stim_site in enumerate(stim_sites):
    fig, axes = plt.subplots(figsize=(3, 1.5), ncols=2)
    exp.inject_activity(stim_site, sigma=sigma)

    # add stim lines
    axes[0].axvline(stim_site[0], **line_kwargs)
    axes[0].axhline(stim_site[1], **line_kwargs)

    # add patches
    for patch in exp.source_tissue.patches:
        axes[0].add_patch(patch.to_mpl_poly(hollow=True, lw=1))

    for patch in exp.target_tissue.patches:
        axes[1].add_patch(patch.to_mpl_poly(hollow=True, lw=1))

    # add selectivity
    for contrast in contrasts:
        sel = exp.source_tissue.responses.selectivity(
            contrast.on_categories, selectivity_fn=array_utils.tstat
        )
        sel_ind = sel > 10.0
        if sel_ind.sum() == 0:
            continue
        normed_inp = norm_bump(exp.input_activity)

        axes[0].scatter(
            exp.source_tissue.positions[sel_ind, 0],
            exp.source_tissue.positions[sel_ind, 1],
            c=contrast.color,
            s=point_size,
            alpha=normed_inp[sel_ind],
            rasterized=True,
        )

        sel = exp.target_tissue.responses.selectivity(
            contrast.on_categories, selectivity_fn=array_utils.tstat
        )
        sel_ind = sel > 10.0
        normed_out = norm_bump(exp.output_flat[sel_ind])
        axes[1].scatter(
            exp.target_tissue.positions[sel_ind, 0],
            exp.target_tissue.positions[sel_ind, 1],
            c=contrast.color,
            s=point_size,
            alpha=normed_out,
            rasterized=True,
        )

    for ax in axes:
        ax.axis("off")
        plot_utils.add_scale_bar(ax, 10)

    axes[0].set_title("Source Layer")
    axes[1].set_title("Target Layer")
    figure_utils.save(fig, f"F08/a_site_{site_idx}")
```

```python
def get_sel_profile(tissue, indices):
    stacked = np.stack(
        [
            tissue.responses.selectivity_category_average(
                on_categories=contrast.on_categories
            )[indices]
            for contrast in contrasts
        ]
    )
    best = np.argmax(stacked, axis=0)
    mx = np.max(stacked, axis=0)
    mask = mx >= 0
    masked = best[mask]
    tuning_profile = np.bincount(masked)
    return tuning_profile / tuning_profile.sum()


def get_activated_indices(exp, percentile=95):
    source_thresh = scoreatpercentile(exp.input_activity, percentile)
    target_thresh = scoreatpercentile(exp.output_flat, percentile)

    return (
        np.nonzero(exp.input_activity >= source_thresh)[0],
        np.nonzero(exp.output_flat >= target_thresh)[0],
    )
```

# estimate true vs. shuffled for many seeds

```python
centers = np.linspace(14, 56, 10)
cx, cy = np.meshgrid(centers, centers[::-1])
cx = cx.ravel()
cy = cy.ravel()
centers = np.stack([cx, cy]).T

# get eccentricity
xdiff = cx - 5
ydiff = cy - 5
eccentricities = np.sqrt(xdiff**2 + ydiff**2)
```

```python
cache_loc = CACHE_DIR / "f8_stim_df_rescale.csv"
```

```python
if cache_loc.is_file():
    df = pd.read_csv(cache_loc)
else:
    sim_results = {"Seed": [], "True": [], "Random": [], "Loc": [], "Ecc": []}

    for seed, exp in experiments.items():
        for cidx, (center, ecc) in tqdm(
            enumerate(zip(centers, eccentricities)), total=len(centers)
        ):
            exp.inject_activity(center, sigma=3.5)

            source_indices, target_indices = get_activated_indices(exp, 95)
            random_indices = np.random.choice(
                np.arange(len(exp.target_tissue._positions)),
                size=(len(target_indices),),
                replace=False,
            )

            source_sel_profile = get_sel_profile(exp.source_tissue, source_indices)
            target_sel_profile = get_sel_profile(exp.target_tissue, target_indices)
            random_sel_profile = get_sel_profile(exp.target_tissue, random_indices)
            try:
                true_sim = array_utils.chisq_sim(target_sel_profile, source_sel_profile)
                random_sim = array_utils.chisq_sim(
                    random_sel_profile, source_sel_profile
                )
            except Exception:
                continue

            sim_results["Seed"].append(seed)
            sim_results["True"].append(true_sim)
            sim_results["Random"].append(random_sim)
            sim_results["Loc"].append(cidx)
            sim_results["Ecc"].append(ecc)

    df = pd.DataFrame(sim_results)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.to_csv(cache_loc)
```

```python
pg.ttest(df["True"], df["Random"], paired=True)
```

```python
trues = df["True"]
randoms = df["Random"]
eccs = df["Ecc"]
eccs = array_utils.norm(eccs)
combined = trues + randoms
diffs = np.array(trues) - np.array(randoms)

fig, ax = plt.subplots(figsize=(1.5, 1.5))
cax = fig.add_axes([0.94, 0.15, 0.05, 0.7])

handle = ax.scatter(
    trues,
    randoms,
    s=5,
    c=eccs,
    edgecolor="k",
    linewidths=0.3,
    rasterized=True,
    alpha=0.8,
    cmap="inferno",
)
ax.set_xlabel(("True Similarity" "\n" r"[-log($\chi^2$)]"))
ax.set_ylabel("Shuffled Similarity")
cb = fig.colorbar(handle, cax=cax)
cb.set_label("Site Location", labelpad=-20)
cb.set_ticks([0, 1])
cb.set_ticklabels(["Central", "Peripheral"])

print(np.mean(df["True"] < df["Random"]))
plot_utils.remove_spines(ax)
ax.set_xlim([-5, 6])
ax.set_ylim([-5, 6])
ax.set_xticks([-5, 0, 5])
ax.set_yticks([-5, 0, 5])

ax.plot(
    [min(combined), max(combined)],
    [min(combined), max(combined)],
    linestyle="dashed",
    c="gray",
    lw=0.75,
)
figure_utils.save(fig, "F08/b_func_align")
```

# BMI/Eye Chart

```python
# load image, transform as training examples would be
image_path = "/share/kalanit/users/eshedm/spacetorch/paper_figures/assets/snellen.png"
img = Image.open(image_path).convert("RGB")
xforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

img_tensor = torch.unsqueeze(xforms(img), dim=0)
```

```python
# load model
model_name = "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
positions = core.get_positions("simclr_swap")
model_seed = 0

model = core.load_model_from_analysis_config(
    f"{model_name}{seed_str(model_seed)}", step="latest"
)
model = model.to(gpu_utils.DEVICE).eval()
```

```python
# choose layers
layers = LAYER_ORDER
model_layer_strings = [f"base_model.{layer}" for layer in layers]
```

```python
# extract features
dataset = TensorDataset(img_tensor, torch.Tensor([-1]))
dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

extractor = FeatureExtractor(dataloader, 1, verbose=True)
features, inputs, labels = extractor.extract_features(
    model, model_layer_strings, return_inputs_and_labels=True
)
```

```python
# view the image
IDX = 0
raw = inputs[IDX].squeeze().transpose(1, 2, 0)
normed = (raw - np.min(raw)) / np.ptp(raw)
fig, ax = plt.subplots(figsize=(1, 1))
ax.imshow(normed)
ax.axis("off")
figure_utils.save(fig, "F08/c_clean_image")
```

```python
# generate patterns
patterns = {}
for ls, mls in tqdm(zip(layers, model_layer_strings), total=8):
    feats = features[mls][IDX].ravel()
    pos = positions[ls].coordinates
    pattern = Pattern(pos=pos, feats=feats)

    pattern.smudge(sigma_mm=0.1)

    patterns[ls] = pattern
```

```python
title_lookup = {
    "layer1.0": "Layer 2",
    "layer2.0": "Layer 4",
    "layer3.0": "Layer 6",
    "layer4.0": "Layer 8",
}
```

```python
# plot patterns
fig, axes = plt.subplots(
    ncols=4, nrows=2, figsize=(3, 2.25), gridspec_kw={"hspace": 0.2, "wspace": 0.1}
)
cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.65])

for ax in axes.ravel():
    ax.axis("off")

clean_row, smudged_row = axes

to_plot = [("layer1.0", 1), ("layer2.0", 9), ("layer3.0", 6), ("layer4.0", 14)]

for clean_ax, smudged_ax, (ls, ext) in zip(clean_row, smudged_row, to_plot):
    pattern = patterns[ls]
    common = {
        "mm_to_show": ext,
        "cmap": "magma",
        "origin": "lower",
        "vmin": 0,
    }
    pattern.plot(clean_ax, smudged=False, **common, vmax=0.5)
    h = pattern.plot(smudged_ax, smudged=True, **common, vmax=1)

    clean_ax.set_title(title_lookup[ls])

    # add scale bars
    for ax in [clean_ax, smudged_ax]:
        total = np.ptp(h.get_extent())
        plot_utils.add_scale_bar(
            ax, width=pattern.px_per_mm * 0.5, y_start=-0.04 * total
        )

cb = plt.colorbar(
    mappable=h,
    cax=cbar_ax,
    ticks=[0, 1],
)
cb.set_ticklabels(["min", "max"])
cb.set_label(label="Response Magnitude (a.u.)", labelpad=-10)


axes[0, 0].text(
    -0.1,
    0.5,
    "Recorded Activity",
    verticalalignment="center",
    horizontalalignment="center",
    rotation=90,
    transform=axes[0, 0].transAxes,
    fontsize=4,
)

axes[1, 0].text(
    -0.15,
    0.5,
    "Activity Achievable\nw/ Coarse Stimulation",
    verticalalignment="center",
    horizontalalignment="center",
    rotation=90,
    transform=axes[1, 0].transAxes,
    fontsize=4,
)
figure_utils.save(fig, "F08/c_patterns")
```

```python
opt_params = OptParams(
    objective_params=ObjectiveParams(layers=layers, smudged=True),
    num_steps=3_000,
)

res = optimize(model, patterns, opt_params, gpu_utils.DEVICE)
```

```python
# grid with different devices
sigmas = [0, 0.0025, 0.01, 0.05, 0.1]

layer_sets = {
    "First Layer": [layers[0]],
    "First 2 Layers": layers[:2],
    "First 4 Layers": layers[:4],
    "First 6 Layers": layers[:6],
    "All 8 Layers": layers,
}
```

```python
save_dir = RESULTS_DIR / "figure_8_stim_patterns_snellen_nocrop_rescale"
save_dir.mkdir(exist_ok=True)
```

```python
# iterate over different combinations of sigma and layers, generating optimal image each time
results = []
for sigma in sigmas:
    print(f"Sigma: {sigma}")
    patterns = {}

    for ls, mls in zip(layers, model_layer_strings):
        feats = features[mls][IDX].ravel()
        pos = positions[ls].coordinates
        pattern = Pattern(pos=pos, feats=feats)
        pattern.smudge(sigma_mm=sigma)
        patterns[ls] = pattern

    for ls_name, layer_set in layer_sets.items():
        save_name = f"sig_{sigma}__{ls_name.replace(' ', '_').lower()}"
        save_path = save_dir / f"{save_name}.pkl"

        if save_path.exists():
            res = generic_utils.load_pickle(save_path)
            results.append(res)
            continue

        opt_params = OptParams(
            objective_params=ObjectiveParams(layers=layer_set, smudged=True),
            num_steps=3_000,
        )

        res = optimize(model, patterns, opt_params, gpu_utils.DEVICE)
        res.sigma = sigma
        res.ls_name = ls_name
        results.append(res)
        generic_utils.write_pickle(save_path, res)
```

```python
# for the figure:
match = [
    res for res in results if res.sigma == 0.0025 and res.ls_name == "All 8 Layers"
][0]
fig, ax = plt.subplots(figsize=(1, 1))
ax.imshow(match.image)
ax.axis("off")
figure_utils.save(fig, "F08/c_image_recon")
```

```python
# plot the final grid
fig, axes = plt.subplots(
    figsize=(4 / 5 * 5, 4 / 5 * len(sigmas)),
    ncols=len(layer_sets),
    nrows=len(sigmas),
    gridspec_kw={"wspace": 0.01, "hspace": 0.1},
)

for i, sigma in enumerate(sigmas):
    sig_mic = sigma * 1000
    axes[i, 0].text(
        -0.1,
        0.5,
        rf"$\sigma = ${sig_mic} $\mu$m",
        transform=axes[i, 0].transAxes,
        rotation=90,
        fontsize=5,
        verticalalignment="center",
        horizontalalignment="center",
    )
    for j, (ls_name, layer_set) in enumerate(layer_sets.items()):
        ax = axes[i, j]
        if i == 0:
            ax.set_title(ls_name, fontsize=5)
        match = [
            res for res in results if res.sigma == sigma and res.ls_name == ls_name
        ][0]
        ax.imshow(match.image)
        ax.axis("off")

figure_utils.save(fig, "F08/e_recon_grid")
```

```python

```
