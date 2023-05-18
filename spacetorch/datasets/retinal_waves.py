import itertools
from typing import List, Optional, Iterable
from pathlib import Path
import warnings

import numpy as np
from numpy.random import default_rng
from skimage import io

import torch
from torch.utils.data import Dataset
import xarray as xr

from spacetorch.paths import RWAVE_CONTAINER_PATH

# types
PathType = type(Path())

DEFAULT_RWAVE_DIRS = [
    RWAVE_CONTAINER_PATH / "big_retina_ad_01",
    RWAVE_CONTAINER_PATH / "big_retina_ad_02",
]


class FramePath(PathType):  # type: ignore
    @property
    def wave_index(self) -> int:
        return int(self.stem.split("_")[1])

    @property
    def step(self) -> int:
        return int(self.stem.split("_")[-1])


class Wave:
    def __init__(
        self, wave_index: int, frame_paths: np.ndarray, random_seed: int = 424
    ):
        """
        frame_paths should be a numpy array of PosixPath objects
        """
        assert isinstance(
            frame_paths, np.ndarray
        ), "frame_paths must be a numpy array of `PosixPath`s to allow fancy indexing"

        self.wave_index = wave_index

        # sort frame paths by step
        self.frame_paths: List[FramePath] = sorted(
            [FramePath(p) for p in frame_paths], key=lambda fp: fp.step
        )

        self.rng = default_rng(seed=random_seed)

    def __getitem__(self, key) -> FramePath:
        return self.frame_paths[key]

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __repr__(self) -> str:
        return f"Wave {self.wave_index}: {len(self)} frames"

    def drop_frames(self, filter_mode: Optional[str] = None, n_keep: int = 8) -> None:
        """
        Permanently remove some images from the wave using `filter_mode`

        Inputs:
            filter_mode:
                drop_from_end: remove frames starting from the end of the movie
                drop_from_start: remove frames starting from the start of the movie
                drop_from_both_ends: remove frames on either end, retaining the middle
                drop_random: does what it says on the tin
                keep_from_random_start: keeps `n_keep` frames starting from a random
                    point
            n_keep: number of frames to keep after dropping
        """
        if filter_mode is None:
            return
        if len(self) <= n_keep:
            warnings.warn(
                (
                    f"Can't drop frames, number of frames({len(self)}) <= number to "
                    f"keep ({n_keep})"
                )
            )

        elif filter_mode == "drop_from_end":
            self.frame_paths = self.frame_paths[:n_keep]
        elif filter_mode == "drop_from_start":
            self.frame_paths = self.frame_paths[-n_keep:]
        elif filter_mode == "drop_from_both_ends":
            n_drop = len(self) - n_keep
            margin = n_drop // 2
            if len(self) % 2 == 0:  # even
                self.frame_paths = self.frame_paths[margin:-margin]
            else:
                self.frame_paths = self.frame_paths[(margin + 1) : -margin]
        elif filter_mode == "drop_random":
            random_indices = np.sort(
                self.rng.choice(len(self), size=(n_keep,), replace=False)
            )
            self.frame_paths = [self.frame_paths[idx] for idx in random_indices]
        elif filter_mode == "keep_from_random_start":
            latest_possible_start = len(self) - n_keep
            start_point = self.rng.choice(latest_possible_start)
            self.frame_paths = self.frame_paths[start_point : (start_point + n_keep)]


class WaveDir:
    def __init__(
        self,
        directory: Path,
        file_pattern: str = "*.png",
        max_number_of_waves: Optional[int] = None,
    ):
        self.directory = directory

        # make this an np array to allow fancy indexing
        self.paths = np.array(
            [FramePath(pth) for pth in self.directory.glob(file_pattern)]
        )

        wave_indices = [path.wave_index for path in self.paths]
        self.waves: List[Wave] = [
            Wave(wave_index, self.paths[wave_index == wave_indices])
            for wave_index in np.unique(wave_indices)[:max_number_of_waves]
        ]

    @property
    def name(self) -> str:
        return str(self.directory.stem)

    def __len__(self) -> int:
        return len(self.waves)

    def __getitem__(self, key: int) -> "Wave":
        return self.waves[key]

    def __repr__(self) -> str:
        return f"Wave Directory: {self.directory} ({len(self)} waves)"

    @property
    def all_paths(self) -> List[FramePath]:
        """
        returns a concatenated list of all paths from all waves (in order?)
        """

        return list(
            itertools.chain.from_iterable([wave.frame_paths for wave in self.waves])
        )


class WavePool:
    """
    Helper class that yields one wave at a time from a pool of WaveDirs
    """

    def __init__(
        self, wave_dir_paths: List[Path], max_waves_per_dir: Optional[int] = None
    ):
        self._wave_dir_paths = wave_dir_paths

        wave_dirs: List[WaveDir] = [
            WaveDir(pth, max_number_of_waves=max_waves_per_dir)
            for pth in self._wave_dir_paths
        ]

        self.waves: Iterable[Wave] = itertools.chain.from_iterable(
            [wave_dir.waves for wave_dir in wave_dirs]
        )

    def __iter__(self) -> Iterable[Wave]:
        return self.waves

    def __next__(self) -> Wave:
        return next(self.waves)  # type: ignore


class RetinalWaveData(Dataset):
    """RetinalWaveData"""

    def __init__(self, wave: Wave, transforms=None):
        if transforms is not None:
            self.transforms = transforms

        self.wave = wave
        self.file_list = self.wave.frame_paths

        steps = [fp.step for fp in self.file_list]
        self.labels = steps

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(self.file_list[idx])

        # remove fourth channel (alpha/opacity)
        img = img[:, :, :3]

        target = self.labels[idx]

        if hasattr(self, "transforms"):
            img = self.transforms(img)

        return img, target


class RetinalWaveResponses:
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self._data = xr.DataArray(
            data=features,
            coords={"wave_step": ("image_idx", labels)},
            dims=["image_idx", "unit_idx"],
        )

    @property
    def mean_per_wave(self) -> np.ndarray:
        """
        Computes the mean respones of each unit to all frames of a wave
        .groupby().mean() is really slow, see
        https://github.com/pydata/xarray/issues/659,
        """
        return self._data.mean(dim="image_idx")

    @property
    def max_per_wave(self) -> np.ndarray:
        """
        Computes the mean respones of each unit to all frames of a wave
        """
        return self._data.max(dim="image_idx")

    def __len__(self) -> int:
        return self._data.sizes["unit_idx"]
