import h5py
import numpy as np
from PIL import Image
import torch
import torchvision

NSD_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

DUMMY_LABEL = 0

class NSDImages(torch.utils.data.Dataset):
    """NSD stimuli."""

    def __init__(self, stim_path, transform=None):
        """
        Args:
            stim_path (string): Path to the hdf file with stimuli.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        stim = h5py.File(stim_path, "r")  # 73k images
        self.data = stim["imgBrick"]
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)

        return x, DUMMY_LABEL

    def __len__(self):
        return len(self.data)
