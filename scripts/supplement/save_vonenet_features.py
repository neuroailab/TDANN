import argparse
from pathlib import Path

import torch

from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_saver import FeatureSaver
from spacetorch.paths import FEATURE_DIR

import vonenet


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def main():
    args = get_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = vonenet.VOneNet(
        simple_channels=64, complex_channels=64, model_arch=None, stride=8
    )
    model.to(device)

    # load dataset
    dataset = DatasetRegistry.get(args.dataset_name)

    log("Constructing feature saver")
    save_path: Path = FEATURE_DIR / "VOneNet" / f"{args.dataset_name}.h5"
    feature_saver: FeatureSaver = FeatureSaver(
        model,
        ["output"],
        dataset,
        save_path,
        max_images=args.max_images,
        layer_prefix="",
    )

    log("Extracting features")
    feature_saver.compute_features(batch_size=args.batch_size)

    log("Saving features")
    feature_saver.save_features()

    log("All done!")


if __name__ == "__main__":
    main()
