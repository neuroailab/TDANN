import argparse
from pathlib import Path

import torch

from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_saver import FeatureSaver
from spacetorch.models import ModelRegistry
from spacetorch.paths import FEATURE_DIR
from spacetorch.utils.generic_utils import load_config_from_yaml

from vissl.utils.hydra_config import AttrDict


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def main():
    args = get_parser().parse_args()
    cfg: AttrDict = load_config_from_yaml(args.config)
    model = ModelRegistry.get_from_config(cfg.model)
    assert model is not None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dataset = DatasetRegistry.get(args.dataset_name)

    log("Constructing feature saver")
    save_path: Path = FEATURE_DIR / cfg.name / f"{args.dataset_name}.h5"
    feature_saver: FeatureSaver = FeatureSaver(
        model, cfg.model.layers, dataset, save_path, max_images=args.max_images
    )

    log("Extracting features")
    feature_saver.compute_features(batch_size=args.batch_size)

    log("Saving features")
    feature_saver.save_features()

    log("All done!")


if __name__ == "__main__":
    main()
