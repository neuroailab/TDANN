import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.utils.hydra_config import AttrDict


@register_model_head("identity")
class Identity(nn.Module):
    """
    Pass through head that does nothing to its inputs
    """

    def __init__(self, model_config: AttrDict):
        super().__init__()

    # the input to the model should be a torch Tensor or list of torch tensors.
    def forward(self, batch: torch.Tensor):
        return batch
