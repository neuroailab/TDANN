"""
Stimulation classes and utils
"""
from typing import Tuple, Any, Union, Dict
from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn
from spacetorch.datasets import sine_gratings, floc, DatasetRegistry
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.maps.v1_map import V1Map
from spacetorch.maps.it_map import ITMap
from spacetorch.models.positions import LayerPositions
from spacetorch.models import OUTPUT_DIMS_FOR_224_INPUTS
from spacetorch.utils import array_utils, torch_utils
from spacetorch.utils.gpu_utils import DEVICE


def get_sine_responses(
    model: nn.Module, batch_size: int = 32, layer: str = "layer2.0"
) -> sine_gratings.SineResponses:
    model = model.to(DEVICE)
    features, _, labels = get_features_from_layer(
        model,
        DatasetRegistry.get("SineGrating2019"),
        f"base_model.{layer}",
        batch_size=batch_size,
    )
    return sine_gratings.SineResponses(features, labels)


def get_floc_responses(
    model: nn.Module, batch_size: int = 32, layer: str = "layer4.1"
) -> floc.fLocResponses:
    model = model.to(DEVICE)
    features, _, labels = get_features_from_layer(
        model,
        DatasetRegistry.get("fLoc"),
        f"base_model.{layer}",
        batch_size=batch_size,
        return_inputs_and_labels=True,
    )
    return floc.fLocResponses(features, labels)


class BaseStimulationExperiment:
    def __init__(
        self,
        model: nn.Module,
        layer_positions: Dict[str, LayerPositions],
        source_layer: str,
        target_layer: str,
    ):
        self.source_layer = source_layer
        self.target_layer = target_layer

        self.source_positions = layer_positions[self.source_layer].coordinates
        self.target_positions = layer_positions[self.target_layer].coordinates

        self.model = model.to(DEVICE)

        self.input_activity = None
        self.reshaped_input_activity = None
        self.raw_output = None
        self.output_flat = None

    def forward(self, input_activity):
        """
        Run activity pattern forward
        """

        self.input_activity = input_activity

        self.reshaped_input_activity = self.input_activity.reshape(
            OUTPUT_DIMS_FOR_224_INPUTS["resnet18"][self.source_layer]
        )
        inp = torch.Tensor(self.reshaped_input_activity[np.newaxis, :]).to(DEVICE)

        module = torch_utils.resolve_sequential_module_from_str(
            self.model, f"base_model.{self.target_layer}"
        )
        self.raw_output = module(inp)
        self.output_flat = self.raw_output.detach().cpu().numpy().squeeze().ravel()


class StimulationExperiment(BaseStimulationExperiment):
    def __init__(
        self,
        map_type: Literal["V1", "IT"],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.map_type = map_type
        self.map_class = Union[V1Map, ITMap]
        self.responses_getter: Any

        if self.map_type.lower() == "v1":
            self.map_class = V1Map
            self.responses_getter = get_sine_responses
        elif self.map_type.lower() == "it":
            self.map_class = ITMap
            self.responses_getter = get_floc_responses

        self._create_tissue_maps()

    def _create_tissue_maps(self):
        self.source_tissue = self.map_class(
            np.array(self.source_positions),
            self.responses_getter(self.model, layer=self.source_layer),
        )

        self.target_tissue = self.map_class(
            np.array(self.target_positions),
            self.responses_getter(self.model, layer=self.target_layer),
        )

    def inject_activity(
        self, center: Tuple[float, float], sigma: float, magnitude: float = 1e9
    ):
        self._source_tmp_mask = np.copy(self.source_tissue.unit_mask)
        self._target_tmp_mask = np.copy(self.target_tissue.unit_mask)
        self.source_tissue.reset_unit_mask()
        self.target_tissue.reset_unit_mask()
        input_activity = (
            array_utils.gaussian_2d(self.source_tissue.positions, center, sigma)
            * magnitude
        )

        self.forward(input_activity)
        self.source_tissue.unit_mask = self._source_tmp_mask
        self.target_tissue.unit_mask = self._target_tmp_mask
