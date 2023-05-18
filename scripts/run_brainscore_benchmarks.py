import argparse
import functools
from typing import List
from pathlib import Path

import brainscore.benchmarks as bs_benchmarks
from brainscore.metrics import Score
from model_tools.brain_transformation import LayerScores
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from model_tools.activations.pca import LayerPCA

import torch
from vissl.utils.hydra_config import AttrDict

from spacetorch.paths import BASE_CHECKPOINT_DIR, POSITION_DIR, RESULTS_DIR
from spacetorch.models import ModelRegistry, BRAIN_MAPPING
from spacetorch import constants
from spacetorch.utils.generic_utils import (
    load_config_from_yaml,
    string_to_list,
    disable_result_caching,
    write_pickle,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--benchmark_identifiers",
        type=str,
        required=True,
        help="comma-separated list of benchmarks to run",
    )
    return parser


def save_score(score: Score, save_path: Path) -> None:
    write_pickle(save_path, score)


def main():
    args = get_parser().parse_args()
    cfg: AttrDict = load_config_from_yaml(args.config)
    benchmark_identifiers: List[str] = string_to_list(args.benchmark_identifiers)

    # see if we can short-circuit
    all_exist = True
    for bmi in benchmark_identifiers:
        save_path = RESULTS_DIR / "neural_fits" / bmi / f"{cfg.name}.pkl"
        if not save_path.is_file():
            all_exist = False
            break

    if all_exist:
        print("All benchmarks already run; skipping")
        return

    disable_result_caching()
    cfg.model.weights.checkpoint_dir = (
        BASE_CHECKPOINT_DIR / cfg.model.weights.checkpoint_dir
    )
    cfg.model.model_config.TRUNK.TRUNK_PARAMS.position_dir = (
        POSITION_DIR / cfg.model.model_config.TRUNK.TRUNK_PARAMS.position_dir
    )

    base_model = ModelRegistry.get_from_config(cfg.model)
    assert base_model is not None
    base_model.eval()

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(
        identifier=cfg.name, model=base_model, preprocessing=preprocessing
    )

    LayerPCA.hook(activations_model, n_components=1000)
    layers = [f"base_model.{layer}" for layer in BRAIN_MAPPING["resnet18"].keys()]

    with torch.no_grad():
        for benchmark_identifier in benchmark_identifiers:
            save_path = (
                RESULTS_DIR / "neural_fits" / benchmark_identifier / f"{cfg.name}.pkl"
            )
            save_path.parent.mkdir(exist_ok=True, parents=True)
            if save_path.exists():
                print("Already exists; skipping")
                continue

            layer_scores = LayerScores(
                cfg.name,
                activations_model=activations_model,
                visual_degrees=constants.DVA_PER_IMAGE,
            )
            scores = layer_scores(
                benchmark=bs_benchmarks.load(benchmark_identifier), layers=layers
            )

            save_score(scores, save_path)


if __name__ == "__main__":
    main()
