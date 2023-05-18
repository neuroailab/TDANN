from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from spacetorch.feature_extractor import FeatureExtractor
from spacetorch.paths import DEFAULT_IMAGENET_VAL_DIR
import xarray as xr


@dataclass
class ImageNetResponses:
    _data: xr.DataArray

    def __len__(self):
        return self._data.sizes["unit_idx"]


IMAGENET_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

NUM_TRAIN_IMAGES = 1_281_167
NUM_VALIDATION_IMAGES = 50_000


class ImageNetData(torchvision.datasets.ImageFolder):
    """ImageNet data"""

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def imagenet_validation_performance(
    model: nn.Module,
    output_layer: str,
    batch_size: int = 16,
    n_batches: Optional[int] = None,
    verbose: bool = True,
    imagenet_val_dir=DEFAULT_IMAGENET_VAL_DIR,
) -> float:

    assert Path(imagenet_val_dir).exists()
    dataset = ImageNetData(imagenet_val_dir, IMAGENET_TRANSFORMS)
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True
    )

    if n_batches is None:
        n_batches = NUM_VALIDATION_IMAGES // batch_size

    feature_extractor = FeatureExtractor(
        data_loader, n_batches, vectorize=True, verbose=verbose
    )
    logits, _, labels = feature_extractor.extract_features(
        model,
        output_layer,
        return_inputs_and_labels=True,
    )

    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == labels)
