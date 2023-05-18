"""
Utility classes for extracting features and saving them to disk in the HDF5 format.
Only really used to save features on disk for later computing unit-to-unit correlations
"""
import math
from pathlib import Path
from typing import Dict, Optional, List, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from spacetorch.feature_extractor import FeatureExtractor
from spacetorch.utils.generic_utils import make_iterable

# types
FeatureDict = Dict[str, np.ndarray]


class FeatureSaver:
    def __init__(
        self,
        model: nn.Module,
        layers: List[str],
        dataset: Dataset,
        save_path: Path,
        max_images: Optional[int] = None,
        layer_prefix: str = "base_model.",
    ):
        self.model = model
        self.layers = layers
        self.dataset = dataset
        self.save_path = save_path
        self.dataset_len = len(dataset)  # type: ignore
        self.max_images = max_images or self.dataset_len
        self.layer_prefix = layer_prefix

    def compute_features(self, batch_size: int = 32):
        n_images: int = min(self.max_images, self.dataset_len)
        n_batches: int = math.ceil(n_images / batch_size)

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        feature_extractor = FeatureExtractor(
            dataloader, n_batches, vectorize=False, verbose=True
        )

        features = feature_extractor.extract_features(
            self.model, [f"{self.layer_prefix}{layer}" for layer in self.layers]
        )
        assert isinstance(features, dict)
        self._features = {k.split("base_model.")[-1]: v for k, v in features.items()}

    def save_features(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.save_path, "w") as f:
            for k, v in self.features.items():
                f.create_dataset(k, data=v)

    @staticmethod
    def load_features(
        load_path: Path, keys: Optional[Union[List[str], str]] = None
    ) -> FeatureDict:

        features = {}
        with h5py.File(load_path, "r") as f:
            if keys is None:
                keys = f.keys()
            else:
                keys = make_iterable(keys)  # type: ignore

            assert keys is not None
            for k in keys:
                features[k] = f[k][:]

        return features

    @property
    def features(self):
        if not hasattr(self, "_features"):
            raise Exception(
                "No features computed. Run the compute_features method first"
            )

        return self._features
