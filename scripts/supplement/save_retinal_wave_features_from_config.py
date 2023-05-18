import argparse
import math

from spacetorch.datasets.retinal_waves import (
    RetinalWaveResponses,
    RetinalWaveData,
    DEFAULT_RWAVE_DIRS,
    WavePool,
)
from spacetorch.feature_extractor import FeatureExtractor
from spacetorch.models import ModelRegistry
from spacetorch.paths import FEATURE_DIR
from spacetorch.utils.generic_utils import load_config_from_yaml

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm
from vissl.utils.hydra_config import AttrDict

starting_image_size = 224
radius = starting_image_size / 2
square_edge_size = math.floor(math.sqrt(2 * math.pow(radius, 2)))

RWAVE_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.CenterCrop(square_edge_size),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser


def make_feature_extractor(dataset: Dataset, batch_size: int = 32) -> FeatureExtractor:
    n_images: int = len(dataset)  # type: ignore
    n_batches: int = math.ceil(n_images / batch_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    return FeatureExtractor(dataloader, n_batches, vectorize=True, verbose=False)


def main():
    args = get_parser().parse_args()
    cfg: AttrDict = load_config_from_yaml(args.config)
    model = ModelRegistry.get_from_config(cfg.model)
    assert model is not None, "model could not be loaded"

    # push model to cuda if we can
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)

    save_path = FEATURE_DIR / cfg.name / "square_retinal_wave_features_mean_per_wave.h5"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    wave_pool = WavePool(DEFAULT_RWAVE_DIRS, max_waves_per_dir=None)

    mean_features = {k: [] for k in cfg.model.layers}
    layer_names = [f"base_model.{layer}" for layer in cfg.model.layers]

    for wave in tqdm(wave_pool, desc="waves"):
        dataset = RetinalWaveData(wave, RWAVE_TRANSFORMS)
        feature_extractor = make_feature_extractor(dataset, args.batch_size)

        features, _, labels = feature_extractor.extract_features(
            model, layer_names, return_inputs_and_labels=True
        )
        assert isinstance(features, dict)

        for layer_name, layer_features in features.items():
            responses = RetinalWaveResponses(layer_features, labels)
            mean_response_per_wave = responses.mean_per_wave.values
            mean_features[layer_name.split("base_model.")[-1]].append(
                mean_response_per_wave
            )

    mean_features = {k: np.stack(v) for k, v in mean_features.items()}

    with h5py.File(save_path, "a") as f:
        for layer, per_wave_aggregation in mean_features.items():
            f.create_dataset(layer, data=per_wave_aggregation)


if __name__ == "__main__":
    main()
