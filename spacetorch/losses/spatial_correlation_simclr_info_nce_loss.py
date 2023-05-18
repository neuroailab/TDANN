from typing import List, Any

import torch
from torch import nn
from classy_vision.losses import ClassyLoss, register_loss
from vissl.utils.hydra_config import AttrDict
from vissl.losses.simclr_info_nce_loss import SimclrInfoNCECriterion

from spacetorch.losses.losses_torch import SpatialCorrelationLossModule


@register_loss("spatial_correlation_simclr_info_nce_loss")
class SpatialCorrelationSimCLRInfoNCE(ClassyLoss):
    """
    Config params:
        spatial_weight: amount by which to scale spatial loss relative to simclr loss
        ignore_index: sample should be ignored for loss, optional
        temperature: specify temperature for softmax. Default 1.0
    """

    def __init__(self, loss_config: AttrDict):
        super(SpatialCorrelationSimCLRInfoNCE, self).__init__()
        self.loss_config = loss_config

        self._neighborhoods_per_batch = self.loss_config.neighborhoods_per_batch
        self._layer_weights = self.loss_config.layer_weights

        # initialize the loss modules
        self._temperature = self.loss_config.temperature
        self._buffer_params = self.loss_config.buffer_params
        self.info_criterion = SimclrInfoNCECriterion(
            self._buffer_params, self._temperature
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

        normalized_output = nn.functional.normalize(flat_outputs, dim=1, p=2)
        loss = self.info_criterion(normalized_output)

        for layer, layer_output in spatial_outputs.items():
            features, pos = layer_output
            layer_loss = self.spatial_loss(
                features, pos.coordinates, pos.neighborhood_indices
            )
            loss += self._layer_weights[layer] * layer_loss

        return loss
