"""
Path management to deal with code being run on different machines. For convenience,
most of the code should be run from within this repository. Lots of stuff will fail
otherwise.
"""

import os
from pathlib import Path

from git.repo import Repo

try:
    working_dir = Repo(".", search_parent_directories=True).working_tree_dir
    git_root = Path(working_dir)
except Exception as e:
    print("Couldn't find git repo; please run code from inside spacetorch/")
    raise e


# try to find the base filesystem
_base_fs = Path(os.environ.get("ST_BASE_FS", "/oak/stanford/groups/kalanit/biac2/kgs"))
assert _base_fs.exists(), f"could not reach base filesystem {_base_fs}"


# standard locations for things:

DS_DIR = _base_fs / "datasets"
NSD_PATH = DS_DIR / "nsd_stimuli.hdf5"
SINE_GRATING_2019_DIR = DS_DIR / "sine_grating_images_20190507"
FLOC_DIR = DS_DIR / "fLoc_stimuli"
IMAGENET_DIR = DS_DIR / "imagenet"
DEFAULT_IMAGENET_VAL_DIR = IMAGENET_DIR / "validation"
RWAVE_CONTAINER_PATH = DS_DIR / "rwave_python_images"

# project outputs
PROJ_DIR = _base_fs / "tdann"

BASE_CHECKPOINT_DIR = PROJ_DIR
CHECKPOINT_DIR = PROJ_DIR / "checkpoints"
SUP_CHECKPOINT_DIR = PROJ_DIR / "checkpoints"

FEATURE_DIR = PROJ_DIR / "features"
POSITION_DIR = PROJ_DIR / "positions"
FIGURE_DIR = PROJ_DIR / "figures"
RESULTS_DIR = PROJ_DIR / "results"
CACHE_DIR = PROJ_DIR / "cache"

analysis_config_dir = git_root / "configs" / "analysis_configs"
