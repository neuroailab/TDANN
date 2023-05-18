"""
Many analyses require either loading a set of positions, loading a model, or accessing
plotting utilities specific to V1-like and VTC-like maps. This module handles the most
common issues
"""
from typing import List, Dict, Optional

import torch.nn as nn

from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.models import ModelRegistry
from spacetorch.models.positions import LayerPositions, NetworkPositions
from spacetorch.models.trunks.resnet import LAYER_ORDER
from spacetorch.paths import (
    BASE_CHECKPOINT_DIR,
    POSITION_DIR,
    analysis_config_dir,
)
from spacetorch.utils import generic_utils, gpu_utils

from spacetorch.types import PositionType

POS_LOOKUP: Dict[PositionType, str] = {
    "simclr_swap": (
        "simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0/"
        "resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019_NBVER2"
    ),
    "supervised_swap": (
        "supervised_resnet18_lw0/"
        "resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019"
    ),
    "retinotopic": "resnet18_retinotopic_init_fuzzy",
    "resnet50": (
        "supervised_resnet50/resnet50_retinotopic_init_swappedon_SineGrating2019"
    ),
    "chanx2": (
        "supervised_resnet18_chanx2/"
        "resnet18_chanx2_retinotopic_init_swappedon_SineGrating2019"
    ),
}


def get_positions(
    pos_type: PositionType, rescale: bool = True
) -> Dict[str, LayerPositions]:
    load_dir = POSITION_DIR / POS_LOOKUP[pos_type]
    netpos = NetworkPositions.load_from_dir(load_dir)
    layer_pos = dict(sorted(netpos.layer_positions.items()))

    # at the the time models were trained, the positions were meant to match closer
    # to estimates in macaque. We scale those initial positions by a fixed amount
    # to make them a closer match to human data
    if getattr(netpos, "version", 1.0) == 1.0:
        rescale_lookup = {
            "layer1.0": 0.6,
            "layer1.1": 0.6,
            "layer2.0": 1.5,
            "layer2.1": 1.5,
            "layer3.0": 1.75,
            "layer3.1": 1.4,
            "layer4.0": 7,
            "layer4.1": 7,
        }

        # perform rescaling
        for k, v in layer_pos.items():
            if rescale:
                v.coordinates = v.coordinates * rescale_lookup[k]
                v.neighborhood_width = v.neighborhood_width * rescale_lookup[k]
            layer_pos[k] = v

    return layer_pos


def load_model_from_analysis_config(
    model_name: str, step="latest", exclude_fc: bool = True
) -> nn.Module:
    config_path = analysis_config_dir / f"{model_name}.yaml"
    assert config_path.exists(), config_path

    config = generic_utils.load_config_from_yaml(config_path)

    ckpt_root = BASE_CHECKPOINT_DIR

    config.model.weights.checkpoint_dir = (
        ckpt_root / config.model.weights.checkpoint_dir
    )

    if hasattr(config.model.model_config.TRUNK.TRUNK_PARAMS, "position_dir"):
        config.model.model_config.TRUNK.TRUNK_PARAMS.position_dir = (
            POSITION_DIR / config.model.model_config.TRUNK.TRUNK_PARAMS.position_dir
        )

    model = ModelRegistry.get_from_config(
        config.model, step=step, exclude_fc=exclude_fc
    )
    assert model is not None
    return model


def get_features_from_model(
    model_name: str,
    dataset_name: str = "ImageNet",
    layers: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
    step="latest",
    exclude_fc=True,
    **kwargs,
):
    layers = layers or LAYER_ORDER
    dataset = DatasetRegistry.get(dataset_name)
    model = load_model_from_analysis_config(
        model_name, step=step, exclude_fc=exclude_fc
    )
    model = model.to(gpu_utils.DEVICE)
    layers = [f"base_model.{layer}" for layer in layers]
    return get_features_from_layer(
        model, dataset, layers, max_batches=max_batches, **kwargs
    )
