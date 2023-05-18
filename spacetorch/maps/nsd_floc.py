"""
Classes and utils for dealing with NSD fLoc data
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from typing_extensions import Literal

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial.distance import pdist, squareform
import skimage.measure
from spacetorch.constants import RNG_SEED

from spacetorch.datasets import floc
from spacetorch.maps.patch import Patch
from spacetorch.utils import array_utils, generic_utils, spatial_utils
from spacetorch.paths import CACHE_DIR


def add_anatomy_labels(ax: plt.Axes):
    """Add text labels indicating anterior, posterior, lateral, and medial for the
    human data
    """
    common = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "fontdict": {"fontstyle": "italic", "fontsize": 6},
        "transform": ax.transAxes,
    }

    ax.text(0.5, 1.07, "A", **common)
    ax.text(0.5, -0.05, "P", **common)
    ax.text(-0.01, 0.5, "L", **common)
    ax.text(1.01, 0.5, "M", **common)


def trim(x: np.ndarray) -> np.ndarray:
    """Given a 2D matrix `x` with some rows and columns all NaN, trim to the section of
    the image that actually contains data.
    """
    nan_rows = np.isnan(x).all(axis=1)
    trimmed = x[~nan_rows]

    nan_cols = np.isnan(trimmed).all(axis=0)
    trimmed = trimmed[:, ~nan_cols]

    return trimmed


# write an adapter to know which floc domain we're referring to in the brain data folder
def nsd2floc_domain_names(nsd_name: str) -> str:

    # these two need to be pluralized
    if nsd_name == "car":
        return "Cars"

    if nsd_name == "word":
        return "Words"

    # everything else just needs to be capitalized
    return nsd_name.capitalize()


@dataclass
class ContrastData:
    """ContrastData tracks data in both hemispheres for a given subject"""

    name: str
    left_hemi: np.ndarray
    right_hemi: np.ndarray
    contrast: floc.Contrast


@dataclass
class HemisphereTransformInfo:
    px_per_mm: float
    extent: np.ndarray  # (2,)

    @property
    def mm_per_px(self) -> float:
        return 1 / self.px_per_mm


@dataclass
class SubjectTransformInfo:
    left_hemi: HemisphereTransformInfo
    right_hemi: HemisphereTransformInfo

    @classmethod
    def from_mat(cls, matfile: Path) -> "SubjectTransformInfo":
        _data = scipy.io.loadmat(matfile)
        xform_info = _data["transform_info"][0][0]
        lh_px_per_mm, rh_px_per_mm, lh_extent, rh_extent = xform_info

        left_hemi = HemisphereTransformInfo(
            px_per_mm=lh_px_per_mm[0][0], extent=lh_extent[0]
        )

        right_hemi = HemisphereTransformInfo(
            px_per_mm=rh_px_per_mm[0][0], extent=rh_extent[0]
        )
        return cls(left_hemi=left_hemi, right_hemi=right_hemi)


@dataclass
class Subject:
    name: str
    xform_info: SubjectTransformInfo
    data: Dict[str, ContrastData]
    lh_patches: List[Patch] = field(default_factory=list)
    rh_patches: List[Patch] = field(default_factory=list)

    def find_patches(
        self,
        hemisphere: Literal["lh", "rh"],
        contrast: floc.Contrast,
        threshold: float = 4,
        minimum_size: float = 100,
        verbose: bool = False,
    ) -> None:
        """
        Arguments:
            hemisphere: one of 'lh' (left hemi) or 'rh' (right hemi)
            contrast: a Contrast to find patches for
            threshold: the threshold, in whatever units the data are defined in
            minimum_size: minimum size of the patches kept in square mm
        """

        cache_probe = (
            f"{self.name}"
            f"_hem{hemisphere}"
            f"_{contrast.name}"
            f"_thr{threshold:.2f}"
            f"_min{minimum_size:.2f}"
        )
        cache_loc = CACHE_DIR / "human_patches" / f"{cache_probe}.pkl"

        if cache_loc.exists():
            patches = generic_utils.load_pickle(cache_loc)
        else:
            patches = []
            contrast_data = self.data[contrast.name.lower()]

            if hemisphere == "lh" or hemisphere == "left_hemi":
                data = contrast_data.left_hemi
                xform_info = self.xform_info.left_hemi
            elif hemisphere == "rh" or hemisphere == "right_hemi":
                data = contrast_data.right_hemi
                xform_info = self.xform_info.right_hemi
            else:
                raise ValueError("hemisphere must be one of 'lh' or 'rh'")

            failing = np.nonzero(data <= threshold)
            cp = np.copy(data)

            # zero out NaN and sub-threshold pixels
            cp[np.isnan(cp)] = 0
            cp[failing] = 0

            # find and label contiguous islands
            labels = skimage.measure.label(cp > 0)
            unique_labels = np.unique(labels)

            # for each unique label, create the patch
            for lab in unique_labels:
                # skip the background pixels
                if lab == 0:
                    continue

                # get rows and columns
                matching = np.where(lab == labels)
                points_px = list(zip(*matching))
                indices = [
                    np.ravel_multi_index(point, data.shape) for point in points_px
                ]

                points_mm = np.array(points_px) * xform_info.mm_per_px

                # if there aren't even 100 voxels in the entire tissue selective for the
                # category, bail eraly
                if len(points_mm) < 100:
                    continue

                # convert points from px to mm
                patch = Patch(
                    points_mm,
                    unit_indices=np.array(indices),
                    selectivities=data[matching],
                    contrast=contrast,
                )

                if patch.area >= minimum_size:
                    patches.append(patch)

            if verbose:
                print(f"Writing to cache {cache_loc}")
            generic_utils.write_pickle(cache_loc, patches)

        for patch in patches:
            if hemisphere == "lh" or hemisphere == "left_hemi":
                self.lh_patches.append(patch)
            else:
                self.rh_patches.append(patch)

    def smoothness(
        self,
        contrast: floc.Contrast,
        hemi: Literal["left_hemi", "right_hemi"],
        bin_edges: np.ndarray,
        distance_cutoff: float = 60,
        num_samples: int = 500,
        sample_size: int = 500,
        shuffle: bool = False,
    ):
        rng = default_rng(seed=RNG_SEED)
        midpoints = array_utils.midpoints_from_bin_edges(bin_edges)
        data = self.data[contrast.name.lower()]

        hemi_data = getattr(data, hemi).T
        where = np.where(~np.isnan(hemi_data))
        positions = np.stack(where).T

        # convert positions into mm
        positions = positions * getattr(self.xform_info, hemi).mm_per_px

        # flatten 2d map into vector
        sel = hemi_data.ravel()
        sel = sel[~np.isnan(sel)]

        # compute the curve
        curves = []
        for _ in range(num_samples):
            indices = rng.choice(
                np.arange(len(positions)), size=(sample_size,), replace=False
            )
            subset_sel = sel[indices]
            subset_sel_expanded = subset_sel[:, np.newaxis]
            sel_diff = np.abs(subset_sel_expanded - subset_sel_expanded.T)
            diff = array_utils.lower_tri(sel_diff)
            distances = array_utils.lower_tri(squareform(pdist(positions[indices])))
            if shuffle:
                rng.shuffle(distances)
            mask = np.nonzero(distances <= distance_cutoff)[0]
            means, _, _ = spatial_utils.agg_by_distance(
                distances[mask], diff[mask], num_bins=20, bin_edges=bin_edges
            )
            curves.append(means)

        return midpoints, np.stack(curves)


def load_data(mat_dir: Path, domains=floc.DOMAIN_CONTRASTS, find_patches: bool = True):
    # load data in a sort of unstructured way
    subject_data: Dict[str, Dict[str, ContrastData]] = {}
    subject_xforms = {}

    for pth in mat_dir.glob("*.mat"):
        subj, category = pth.stem.split("_")
        floc_category = nsd2floc_domain_names(category)
        try:
            contrast = next(
                filter(lambda c: c.name.lower() == floc_category.lower(), domains)
            )
        except Exception:
            continue

        data = scipy.io.loadmat(pth)["raw"]
        subject_xforms[subj] = SubjectTransformInfo.from_mat(pth)

        # separate into hemispheres
        midpoint = data.shape[1] // 2
        left_hemi = data[:, :midpoint]
        right_hemi = data[:, midpoint:]

        # create contrast data
        contrast_data = ContrastData(
            name=category,
            left_hemi=trim(left_hemi),
            right_hemi=np.fliplr(trim(right_hemi)),
            contrast=contrast,
        )

        if subj not in subject_data.keys():
            subject_data[subj] = {}

        if category == "car":
            category = "cars"
        elif category == "word":
            category = "words"
        subject_data[subj][category] = contrast_data

    subjects = []
    for subj_name, subj_data in subject_data.items():
        subject = Subject(
            name=subj_name, data=subj_data, xform_info=subject_xforms[subj_name]
        )
        if find_patches:
            for contrast in domains:
                for hemi in ["lh", "rh"]:
                    subject.find_patches(
                        hemisphere=hemi,  # type: ignore
                        contrast=contrast,
                    )
        subjects.append(subject)

    return sorted(subjects, key=lambda sub: sub.name)
