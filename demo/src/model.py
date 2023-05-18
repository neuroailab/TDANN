import torch
import torch.nn as nn
import torchvision

LAYERS = [
    "layer1.0",
    "layer1.1",
    "layer2.0",
    "layer2.1",
    "layer3.0",
    "layer3.1",
    "layer4.0",
    "layer4.1",
]


def load_model_from_checkpoint(checkpoint_path: str):
    model = torchvision.models.resnet18(pretrained=False)

    # drop the FC layer
    model.fc = nn.Identity()

    # load weights
    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("base_model") and "fc." not in k:
            remainder = k.split("base_model.")[-1]
            new_state_dict[remainder] = v

    model.load_state_dict(new_state_dict)

    # freeze all weights
    for param in model.parameters():
        param.requires_grad = False

    return model
