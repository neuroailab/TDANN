from .imagenet import ImageNetData, IMAGENET_TRANSFORMS
from .sine_gratings import SineGrating2019
from .floc import fLocData
from .retinal_waves import RetinalWaveData, DEFAULT_RWAVE_DIRS
from .noise import NoiseImages, NOISE_TRANSFORMS
from .nsd import NSDImages, NSD_TRANSFORMS

import torchvision
from spacetorch import paths

DEFAULT_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

_DATASETS = {
    "SineGrating2019": (
        SineGrating2019,
        (paths.SINE_GRATING_2019_DIR, DEFAULT_TRANSFORMS),
    ),
    "fLoc": (fLocData, (paths.FLOC_DIR, DEFAULT_TRANSFORMS)),
    "NSD": (NSDImages, (paths.NSD_PATH, NSD_TRANSFORMS)),
    "ImageNet": (
        ImageNetData,
        (f"{paths.IMAGENET_DIR}/validation", IMAGENET_TRANSFORMS),
    ),
    "RetinalWaves": (RetinalWaveData, (DEFAULT_RWAVE_DIRS, DEFAULT_TRANSFORMS)),
    "noise": (NoiseImages, (640, NOISE_TRANSFORMS)),
}


class DatasetRegistry:
    def __init__(self):
        self._DATASETS = _DATASETS

    @staticmethod
    def get(dataset_name: str):
        dataset = _DATASETS.get(dataset_name)
        if dataset is None:
            raise ValueError(
                f"Sorry, {dataset_name} not in registry. Try one of {_DATASETS.keys()}"
            )
        dataset_cls, dataset_args = dataset
        return dataset_cls(*dataset_args)

    @staticmethod
    def list() -> None:
        """
        Print all datasets in the registry
        """
        print("Available datasets:")
        for dataset in _DATASETS:
            print(f"\t {dataset}")
