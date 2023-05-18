import numpy as np
from numpy.random import default_rng
from torch.utils.data import Dataset
import torchvision

from spacetorch.constants import RNG_SEED

NOISE_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


class NoiseImages(Dataset):
    def __init__(self, num_images, transforms):
        self.num_images = num_images
        self.transforms = transforms
        self.rng = default_rng(seed=RNG_SEED)

    def __len__(self):
        return self.num_images

    def __getitem__(self, _):
        img = self.rng.normal(size=(224, 224, 3)).astype(np.float32)
        target = -1

        if self.transforms:
            img = self.transforms(img)

        return img, target
