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
```

```python
from spacetorch.datasets import floc
from spacetorch.maps import nsd_floc
from spacetorch.utils import figure_utils, plot_utils
from spacetorch.paths import PROJ_DIR
```

```python
figure_utils.set_text_sizes()
contrasts = floc.DOMAIN_CONTRASTS
contrast_order = ["Faces", "Bodies", "Characters", "Places", "Objects"]
contrast_colors = {c.name: c.color for c in contrasts}
contrast_dict = {c.name: c for c in contrasts}
```

```python
mat_dir = PROJ_DIR / "nsd_tvals"
subjects = nsd_floc.load_data(mat_dir, domains=contrasts, find_patches=True)
the_chosen_one = subjects[3]
```

```python
mosaic = """AABBCCDD
            AABBCCDD
            EFGHIJKL
            MNOPQRST
            """
fig = plt.figure(figsize=(6, 3))
ax_dict = fig.subplot_mosaic(
    mosaic,
    gridspec_kw={
        "wspace": 0.7,
        "hspace": 0.2,
    },
)
```

```python
data = the_chosen_one.data["faces"].right_hemi[:, 300:]
```

```python
ax_dict["A"].axis("off")
mappable = ax_dict["A"].imshow(
    data.T,
    vmin=-20,
    vmax=20,
    cmap="seismic",
)
cb = fig.colorbar(mappable, ax=ax_dict["A"], ticks=[20, -20], shrink=0.4, pad=0.1)
cb.set_label(label=r"$t$-value", labelpad=-10)
nsd_floc.add_anatomy_labels(ax_dict["A"])
```

```python
ax = ax_dict["B"]
ax.axis("off")

mask = ~np.isnan(data)
mm_per_px = the_chosen_one.xform_info.right_hemi.mm_per_px
lims = mm_per_px * np.array(np.shape(mask))
extent = [0, lims[0], lims[1], 0]

# raw
ax.imshow(mask.T, alpha=0.2, cmap="gray_r", extent=extent)
contrast = contrast_dict["Faces"]
hemi = the_chosen_one.data[contrast.name.lower()].right_hemi[:, 300:]
passing_mask = hemi > 6
passing_x, passing_y = np.nonzero(passing_mask)
passing_x_mm = passing_x * mm_per_px
passing_y_mm = passing_y * mm_per_px

ax.scatter(
    passing_x_mm,
    passing_y_mm,
    c=contrast.color,
    s=2,
    alpha=1,
    label=contrast.name,
    marker="s",
    rasterized=True,
)

plot_utils.add_scale_bar(ax, 2 * the_chosen_one.xform_info.right_hemi.px_per_mm)
```

```python
ax = ax_dict["C"]
ax.axis("off")

mask = ~np.isnan(data)
mm_per_px = the_chosen_one.xform_info.right_hemi.mm_per_px
lims = mm_per_px * np.array(np.shape(mask))
extent = [0, lims[0], lims[1], 0]

# raw
ax.imshow(mask.T, alpha=0.2, cmap="gray_r", extent=extent)
for contrast in contrasts:
    hemi = the_chosen_one.data[contrast.name.lower()].right_hemi[:, 300:]
    passing_mask = hemi > 6
    passing_x, passing_y = np.nonzero(passing_mask)
    passing_x_mm = passing_x * mm_per_px
    passing_y_mm = passing_y * mm_per_px

    ax.scatter(
        passing_x_mm,
        passing_y_mm,
        c=contrast.color,
        s=2,
        alpha=1,
        label=contrast.name,
        marker="s",
        rasterized=True,
    )

plot_utils.add_scale_bar(ax, 2 * the_chosen_one.xform_info.right_hemi.px_per_mm)
```

```python
ax = ax_dict["D"]
ax.axis("off")

mask = ~np.isnan(the_chosen_one.data["faces"].right_hemi)
mm_per_px = the_chosen_one.xform_info.right_hemi.mm_per_px
lims = mm_per_px * np.array(np.shape(mask))
extent = [0, lims[0], lims[1], 0]

# raw
ax.imshow(mask.T, alpha=0.2, cmap="gray_r", extent=extent)
for contrast in contrasts:
    matchy_patchy = [
        patch
        for patch in the_chosen_one.rh_patches
        if patch.contrast.name == contrast.name
    ]

    hemi = the_chosen_one.data[contrast.name.lower()].right_hemi
    passing_mask = hemi > 6
    passing_x, passing_y = np.nonzero(passing_mask)
    passing_x_mm = passing_x * mm_per_px
    passing_y_mm = passing_y * mm_per_px
    ax.scatter(
        passing_x_mm,
        passing_y_mm,
        c=contrast.color,
        s=1,
        alpha=0.0,
        label=contrast.name,
        rasterized=True,
    )
    for patch in matchy_patchy:
        ax.add_patch(patch.to_mpl_poly(alpha=0.8, lw=1))

plot_utils.add_scale_bar(ax, 2 * the_chosen_one.xform_info.right_hemi.px_per_mm)
yl = ax.get_ylim()
ax.set_ylim([yl[0], yl[0] / 2])
```

```python
ax_dict["A"].set_title("Face Selectivity", pad=10)
ax_dict["B"].set_title("Thresholded")
ax_dict["C"].set_title("All Categories")
ax_dict["D"].set_title("Detected Patches")
```

```python
ax_lookup = {
    (0, "left_hemi"): "E",
    (1, "left_hemi"): "F",
    (2, "left_hemi"): "G",
    (3, "left_hemi"): "H",
    (4, "left_hemi"): "I",
    (5, "left_hemi"): "J",
    (6, "left_hemi"): "K",
    (7, "left_hemi"): "L",
    (0, "right_hemi"): "M",
    (1, "right_hemi"): "N",
    (2, "right_hemi"): "O",
    (3, "right_hemi"): "P",
    (4, "right_hemi"): "Q",
    (5, "right_hemi"): "R",
    (6, "right_hemi"): "S",
    (7, "right_hemi"): "T",
}
```

```python
# all hemis all subjects

for (subj_idx, hemi_name), ax_letter in ax_lookup.items():
    ax = ax_dict[ax_letter]
    subject = subjects[subj_idx]

    if hemi_name == "left_hemi":
        ax.set_title(subject.name)

    patch_indexer = "lh_patches" if hemi_name == "left_hemi" else "rh_patches"
    patches = getattr(subject, patch_indexer)
    ax.axis("off")

    mask = ~np.isnan(getattr(subject.data["faces"], hemi_name))
    mm_per_px = getattr(subject.xform_info, hemi_name).mm_per_px
    px_per_mm = getattr(subject.xform_info, hemi_name).px_per_mm
    lims = mm_per_px * np.array(np.shape(mask))
    extent = [0, lims[0], lims[1], 0]

    # raw
    ax.imshow(mask.T, alpha=0.2, cmap="gray_r", extent=extent)
    for contrast in contrasts:
        matchy_patchy = [
            patch for patch in patches if patch.contrast.name == contrast.name
        ]

        hemi = getattr(subject.data[contrast.name.lower()], hemi_name)
        passing_mask = hemi > 6
        passing_x, passing_y = np.nonzero(passing_mask)
        passing_x_mm = passing_x * mm_per_px
        passing_y_mm = passing_y * mm_per_px
        ax.scatter(
            passing_x_mm,
            passing_y_mm,
            c=contrast.color,
            s=1,
            alpha=0.0,
            label=contrast.name,
            rasterized=True,
        )
        for patch in matchy_patchy:
            ax.add_patch(patch.to_mpl_poly(alpha=0.8, lw=0.5))

    plot_utils.add_scale_bar(ax, 2 * px_per_mm)
    yl = ax.get_ylim()
    ax.set_ylim([yl[0], yl[0] / 2.5])
```

```python
ax_dict["E"].text(
    -0.3, 0.4, "LH", rotation=90, transform=ax_dict["E"].transAxes, fontsize=6
)

ax_dict["M"].text(
    -0.3, 0.4, "RH", rotation=90, transform=ax_dict["M"].transAxes, fontsize=6
)
```

```python
plot_contrasts = [con for con in contrasts if con.name != "Objects"]
for cidx, contrast in enumerate(plot_contrasts):
    fig.text(
        0.62 + (cidx * 0.08),
        0.53,
        contrast.name,
        color=contrast.color,
        fontdict={"weight": 700},
        horizontalalignment="center",
    )
```

```python
figure_utils.save(fig, "S05/human_data")
fig
```

```python

```
