from typing import List, Any

import torch
from torch import nn
from classy_vision.losses import ClassyLoss, register_loss
from vissl.utils.hydra_config import AttrDict

from spacetorch.losses.losses_torch import SpatialCorrelationLossModule


@register_loss("cross_entropy_spatial_correlation_loss")
class CrossEntropySpatialCorrelationLoss(ClassyLoss):
    """
    Config params:
        spatial_weight: amount by which to scale spatial loss relative to cross
            entropy loss
        ignore_index: sample should be ignored for loss, optional
        temperature: specify temperature for softmax. Default 1.0
    """

    def __init__(self, loss_config: AttrDict):
        super(CrossEntropySpatialCorrelationLoss, self).__init__()
        self._ignore_index = -1
        self._temperature = loss_config["temperature"]
        self._neighborhoods_per_batch = loss_config["neighborhoods_per_batch"]
        self._layer_weights = loss_config["layer_weights"]

        if "ignore_index" in loss_config:
            self._ignore_index = loss_config["ignore_index"]

        # initialize the loss modules
        self.categorization_loss = nn.modules.CrossEntropyLoss(
            ignore_index=self._ignore_index
        )

        self.use_old_version = loss_config.use_old_version
        self.spatial_loss = SpatialCorrelationLossModule(
            self._neighborhoods_per_batch, use_old_version=self.use_old_version
        )

    # VISSL requires this signature to return a class instance
    @classmethod
    def from_config(cls, loss_config: AttrDict):
        return cls(loss_config)

    def forward(self, output: List[Any], target: torch.Tensor):
        flat_outputs, spatial_outputs = output

        loss = self.categorization_loss(flat_outputs, target)

        for layer, layer_output in spatial_outputs.items():
            features, pos = layer_output
            layer_loss = self.spatial_loss(
                features, pos.coordinates, pos.neighborhood_indices
            )
            loss += self._layer_weights[layer] * layer_loss

        return loss
