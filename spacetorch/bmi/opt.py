from dataclasses import dataclass, asdict, field
from typing import List, Dict

from lucent.optvis.render import hook_model
from lucent.optvis import param
from lucent.optvis.render import tensor_to_img_array
from lucent.optvis.objectives import wrap_objective, handle_batch
from lucent.optvis.param.color import to_valid_rgb
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from spacetorch.types import RN18Layer, DeviceStr
from .pattern import Pattern

DEFAULT_LAYERS: List[RN18Layer] = [
    "layer1.0",
    "layer1.1",
    "layer2.0",
    "layer2.1",
    "layer3.0",
    "layer3.1",
    "layer4.0",
    "layer4.1",
]


@dataclass
class ObjectiveParams:
    """ObjectiveParams track which layers should be used during optimization, and
    whether the original or the "smudged" versions of activity patterns should be used.
    """

    layers: List[RN18Layer] = field(default_factory=lambda: DEFAULT_LAYERS)
    smudged: bool = False


@dataclass
class OptParams:
    """Optimization params: mostly learning rate and number of steps to run for"""

    objective_params: ObjectiveParams
    lr: float = 5e-2
    num_steps: int = 3_000


@wrap_objective()
def reproduce_activity(
    patterns: Dict[RN18Layer, Pattern], device: DeviceStr, layers=None, smudged=False
):
    layers = layers or DEFAULT_LAYERS

    @handle_batch(None)
    def inner(hook_fn):
        total = 0
        for ls in layers:
            mls = f"base_model.{ls}"
            if smudged:
                targ = patterns[ls].smudged_feats
            else:
                targ = patterns[ls].feats

            targ = torch.from_numpy(targ).to(device)
            out = torch.flatten(hook_fn(mls.replace(".", "_")))
            mse = ((out - targ) ** 2).mean()
            total += mse
        return total

    return inner


@dataclass
class OptResults:
    image: np.ndarray
    mse: float
    params: OptParams


def optimize(model: nn.Module, patterns, opt_params: OptParams, device) -> OptResults:
    param_f = lambda: param.pixel_image((1, 3, 224, 224))  # noqa
    params, image_f = param_f()
    image_f = to_valid_rgb(image_f)
    optimizer = torch.optim.Adam(params, lr=opt_params.lr)
    hook = hook_model(model, image_f)
    objective_f = reproduce_activity(
        patterns, device, **asdict(opt_params.objective_params)
    )

    for idx in tqdm(range(1, opt_params.num_steps + 1)):

        def optimizer_step():
            optimizer.zero_grad()
            try:
                model(image_f())
            except RuntimeError as ex:
                if idx == 1:
                    print(ex)
            loss = objective_f(hook)
            loss.backward()
            return loss

        optimizer.step(optimizer_step)

    image = tensor_to_img_array(image_f())
    final = image.copy()
    raw = final.squeeze()
    normed = (raw - np.min(raw)) / np.ptp(raw)
    loss = objective_f(hook)
    loss = float(loss.detach().cpu().numpy())
    return OptResults(image=normed, mse=loss, params=opt_params)
