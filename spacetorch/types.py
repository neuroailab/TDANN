from typing import Tuple
from typing_extensions import Literal

AggMode = Literal["mode", "mean", "circmean", "circstd"]
PositionType = Literal[
    "simclr_swap",
    "supervised_swap",
    "resnet50",
    "chanx2",
]

RN18Layer = Literal[
    "layer1.0",
    "layer1.1",
    "layer2.0",
    "layer2.1",
    "layer3.0",
    "layer3.1",
    "layer4.0",
    "layer4.1",
]

VVSRegion = Literal[
    "retina",
    "V1",
    "V2",
    "V4",
    "IT"
]

DeviceStr = Literal["cpu", "cuda"]

# shallow aliases
LayerString = str
Dims = Tuple[int, int, int]