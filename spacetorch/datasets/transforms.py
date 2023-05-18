from typing import Union, Dict, Any

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image
import torch


@register_transform("ReplaceWithNoise")
class ReplaceWithNoise(ClassyTransform):
    """
    Simply replaces image with white noise
    """

    # the input image should either be Image.Image PIL instance or torch.Tensor
    def __call__(self, image: Union[Image.Image, torch.Tensor]):
        # scaling by 0.5 puts values in [-2, 2], which is about where the simCLR preproc
        # ends up
        return 0.5 * torch.randn_like(image)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ReplaceWithNoise":
        return cls()
