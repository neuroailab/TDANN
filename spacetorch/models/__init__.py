import logging
from pathlib import Path
import re
from typing import Callable, Mapping, Optional, Union, List, Type
from typing_extensions import Literal

import torch
import torch.nn as nn
from .trunks.resnet import VisslResNet, SpatialResNet18, SpatialResNet50
from .trunks.resnet_chanx import ResNet18_ChanX, SpatialResNet18_ChanX

from vissl.utils.hydra_config import AttrDict
from vissl.models.heads.mlp import MLP as VisslMLP

from spacetorch.utils.generic_utils import castable_to_type

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s %(levelname)s:%(message)s"
)
logger = logging.getLogger("model init")

# types
ModelConfig = Type[AttrDict]
ModelName = str
ModelConstructor = Callable[[ModelConfig, ModelName], nn.Module]

ModelPart = Literal["heads", "trunk"]
ModelParts = Mapping[ModelPart, ModelConstructor]

_MODELS: Mapping[str, ModelParts] = {
    "VisslResNet": {"trunk": VisslResNet, "heads": VisslMLP},
    "SpatialResNet18": {"trunk": SpatialResNet18, "heads": VisslMLP},
    "SpatialResNet50": {"trunk": SpatialResNet50, "heads": VisslMLP},
    "ResNet18_ChanX": {"trunk": ResNet18_ChanX, "heads": VisslMLP},
    "SpatialResNet18_ChanX": {"trunk": SpatialResNet18_ChanX, "heads": VisslMLP},
}

# brain mappings
BRAIN_MAPPING = {
    "resnet18": {
        "layer1.0": "retina",
        "layer1.1": "retina",
        "layer2.0": "V1",
        "layer2.1": "V1",
        "layer3.0": "V2",
        "layer3.1": "V4",
        "layer4.0": "IT",
        "layer4.1": "IT",
    },
    "resnet50": {
        "layer1.0": "retina",
        "layer1.1": "retina",
        "layer1.2": "retina",
        "layer2.0": "V1",
        "layer2.1": "V1",
        "layer2.2": "V1",
        "layer2.3": "V2",
        "layer3.0": "V4",
        "layer3.1": "V4",
        "layer3.2": "V4",
        "layer3.3": "V4",
        "layer3.4": "V4",
        "layer3.5": "V4",
        "layer4.0": "IT",
        "layer4.1": "IT",
        "layer4.2": "IT",
    },
}

OUTPUT_DIMS_FOR_224_INPUTS = {
    "resnet18": {
        "layer1.0": (64, 56, 56),
        "layer1.1": (64, 56, 56),
        "layer2.0": (128, 28, 28),
        "layer2.1": (128, 28, 28),
        "layer3.0": (256, 14, 14),
        "layer3.1": (256, 14, 14),
        "layer4.0": (512, 7, 7),
        "layer4.1": (512, 7, 7),
    },
    "resnet50": {
        "layer1.0": (256, 56, 56),
        "layer1.1": (256, 56, 56),
        "layer1.2": (256, 56, 56),
        "layer2.0": (512, 28, 28),
        "layer2.1": (512, 28, 28),
        "layer2.2": (512, 28, 28),
        "layer2.3": (512, 28, 28),
        "layer3.0": (1024, 14, 14),
        "layer3.1": (1024, 14, 14),
        "layer3.2": (1024, 14, 14),
        "layer3.3": (1024, 14, 14),
        "layer3.4": (1024, 14, 14),
        "layer3.5": (1024, 14, 14),
        "layer4.0": (2048, 7, 7),
        "layer4.1": (2048, 7, 7),
        "layer4.2": (2048, 7, 7),
    },
}


def load_weights(
    model: nn.Module,
    weight_path: Path,
    model_part: ModelPart = "trunk",
    cpu_only: bool = False,
    to_exclude=None,
):
    load_kwargs = {"map_location": torch.device("cpu")} if cpu_only else {}
    ckpt = torch.load(weight_path, **load_kwargs)
    model_params = ckpt["classy_state_dict"]["base_model"]["model"][model_part]
    if to_exclude is not None:
        model_params = {k: v for k, v in model_params.items() if k not in to_exclude}

    if model_part == "heads":
        model_params = {
            "clf.0.weight": model_params["0.clf.0.weight"],
            "clf.0.bias": model_params["0.clf.0.bias"],
        }

    model.load_state_dict(model_params)


def resolve_weight_path(
    weight_config: AttrDict, file_pattern: str = "*phase*.torch"
) -> Optional[Path]:
    """
    weight_config should have two keys:
        checkpoint_dir: str or None
        step: str or int or None. If str, only "latest" is supported
    """

    def phase_from_path(path: Path) -> int:
        """Returns the first integer in the stem"""
        return int(re.findall(r"\d+", path.stem)[0])

    # return None if we either part of the config is None
    if weight_config.checkpoint_dir is None or weight_config.step is None:
        logger.info("Either checkpoint step or directory was null, returning None")
        return None

    # if we made it this far, there should be a checkpoint to return somewhere
    checkpoint_dir = Path(weight_config.checkpoint_dir)
    assert checkpoint_dir.is_dir(), checkpoint_dir

    # if the user wants the latest checkpoint, sort by phase and return the last element
    if weight_config.step == "latest":
        sorted_paths: List[Path] = sorted(
            checkpoint_dir.glob(file_pattern), key=phase_from_path
        )
        the_chosen_one = sorted_paths[-1]
        logger.info(f"Step 'latest' resolved to {the_chosen_one}")
        return the_chosen_one

    # if the user thinks they know which phase to grab, reconstruct the path
    if castable_to_type(weight_config.step, int):
        step = int(weight_config.step)
        filtered_paths = list(
            filter(
                lambda pth: phase_from_path(pth) == step,
                checkpoint_dir.glob(file_pattern),
            )
        )
        assert (
            len(filtered_paths) == 1
        ), f"Couldn't find the checkpoint you requested: {weight_config}"
        return filtered_paths[-1]

    return None


class ModelRegistry:
    def __init__(self):
        self._MODELS = _MODELS

    @staticmethod
    def get(
        model_name: ModelName,
        model_config: ModelConfig,
        weight_path: Optional[Union[str, Path]] = None,
        model_part: ModelPart = "trunk",
        cpu_only: bool = False,
        exclude_fc: bool = True,
    ) -> Optional[nn.Module]:
        assert model_part in ["heads", "trunk"], "model part must be 'trunk' or 'heads'"
        parts: Optional[ModelParts] = _MODELS.get(model_name)
        if parts is None:
            return None

        model_cls = parts[model_part]
        if model_part == "heads":
            model = model_cls(model_config, model_config.HEAD.DIMS)
        else:
            model = model_cls(model_config, model_name)

        if weight_path is not None:
            weight_path = Path(weight_path)
            assert weight_path.exists(), weight_path

            to_exclude = (
                ["base_model.fc.weight", "base_model.fc.bias"] if exclude_fc else None
            )
            load_weights(
                model,
                weight_path,
                model_part=model_part,
                cpu_only=cpu_only,
                to_exclude=to_exclude,
            )

        return model

    @classmethod
    def get_from_config(
        cls, config: ModelConfig, step="latest", exclude_fc: bool = True
    ) -> Optional[nn.Module]:
        if step == "random":
            config.weights.step = None
        elif step is not None:
            config.weights.step = step

        return cls.get(
            config.name,
            config.model_config,
            resolve_weight_path(config.weights),
            exclude_fc=exclude_fc,
        )

    @staticmethod
    def list() -> None:
        """
        Print all models in the registry
        """
        print("Available models:")
        for model in _MODELS:
            print(f"\t {model}")
