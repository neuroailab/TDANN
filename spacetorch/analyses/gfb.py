"""Helper to load a V1-like tissue map from the VOneNet Gabor filter bank
"""

from typing_extensions import Literal
from spacetorch.datasets import DatasetRegistry

import vonenet

from spacetorch.datasets import sine_gratings
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.maps.v1_map import V1Map
from spacetorch.models.positions import NetworkPositions
from spacetorch.paths import CACHE_DIR, POSITION_DIR
from spacetorch.utils import generic_utils, gpu_utils


def get_gfb_tissue(position_type: Literal["ImageNet", "SineGrating2019"]):
    cache_id = f"VOneNet_64_64_stride8_{position_type}"
    cache_loc = CACHE_DIR / "tuning_curve_fits" / cache_id / "responses.pkl"
    if cache_loc.exists():
        responses = generic_utils.load_pickle(cache_loc)
        print(f"Loaded {cache_id} from cache")
    else:
        model = vonenet.VOneNet(
            simple_channels=64,
            complex_channels=64,
            model_arch=None,  # pyright: ignore
            stride=8,
        )
        model.to(gpu_utils.DEVICE)
        sine_features, _, labels = get_features_from_layer(
            model,
            DatasetRegistry.get("SineGrating2019"),
            "output",
            batch_size=32,
            return_inputs_and_labels=True,
        )
        responses = sine_gratings.SineResponses(sine_features, labels)

    load_dir = POSITION_DIR / "VOneNet" / f"VOneNet_swappedon_{position_type}"
    positions = NetworkPositions.load_from_dir(load_dir).layer_positions["layer2.0"]

    return V1Map(positions.coordinates, responses, cache_id=cache_id)
