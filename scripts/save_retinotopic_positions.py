"""
Saves initial positions for each layer of a given model, initialized retinotopically
"""
import argparse
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple, Any

from spacetorch.models import BRAIN_MAPPING, OUTPUT_DIMS_FOR_224_INPUTS
from spacetorch.models.positions import (
    LayerPositions,
    TISSUE_SIZES,
    NEIGHBORHOOD_WIDTHS,
)
from spacetorch.types import Dims, LayerString, VVSRegion
from spacetorch.utils.spatial_utils import (
    collapse_and_trim_neighborhoods,
    jitter_positions,
    place_conv,
    precompute_neighborhoods,
)
from spacetorch.utils.generic_utils import load_config_from_yaml
from spacetorch.paths import POSITION_DIR

# script constants
INITIAL_RF_OVERLAP = 0.1  # how much do hypercolumns overlap in the first layer?
POS_VERSION = 2  # increment this every time the position scheme changes

# set up logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)-15s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser


@dataclass
class LayerPlacement:
    """A layer placement describes the high-level parameters for how units in a given
    layer should be arranged.

    Attributes:
        name: the name of the layer, e.g., "layer4.1"
        tissue_size: total extent of the layer tissue in mm, e.g., 10mm
        dims: expected dimensionality of the outputs in this layer, e.g., (128, 28, 28)
            means that the outputs for this layer come from 128 distinct kernels and
            (28 * 28 = 784) spatial positions
        rf_overlap: The units at each spatial position occupy a square subregion of the tissue.
            This parameter controls how much adjacent subregions overlap
        neighborhood_width: size in mm of a "neighborhood" of units. During training,
            only units in the same neighborhood participate in computation of the spatial
            cost
    """

    name: str
    tissue_size: float
    dims: Dims
    rf_overlap: float = INITIAL_RF_OVERLAP
    neighborhood_width: float = 0.5


def get_placement_configs(
    brain_mapping: Dict[LayerString, VVSRegion],
    output_dims_for_224_inputs: Dict[LayerString, Dims],
) -> List[LayerPlacement]:
    placement_configs: List[LayerPlacement] = [
        LayerPlacement(
            name=layer,
            tissue_size=TISSUE_SIZES[brain_area],
            dims=output_dims_for_224_inputs[layer],
            neighborhood_width=NEIGHBORHOOD_WIDTHS[brain_area],
        )
        for (layer, brain_area) in brain_mapping.items()
    ]

    # beginning with INITIAL_RF_OVERLAP, the amount of overlap for each layer is 
    # proportional to the amount that the spatial feature maps have shrunk. For example,
    # after the first block (layer1.0 and layer1.1), feature maps shrink by 2x from 64x64
    # to 28x28. Accordingly, we double RF overlap for the next set of layers
    initial_x = placement_configs[0].dims[-1]
    for cfg in placement_configs:
        downsampling_ratio = initial_x / cfg.dims[-1]
        cfg.rf_overlap = min(INITIAL_RF_OVERLAP * downsampling_ratio, 1.0)

    return placement_configs


def create_position_dict(cfg: LayerPlacement) -> Dict[str, Any]:
    positions, rf_radius = place_conv(
        dims=cfg.dims,
        pos_lims=(0, cfg.tissue_size),
        offset_pattern="random",
        rf_overlap=cfg.rf_overlap,
        return_rf_radius=True,
    )

    positions = jitter_positions(positions, jitter=0.3)

    neighborhood_list = precompute_neighborhoods(
        positions, radius=cfg.neighborhood_width / 2, n_neighborhoods=20_000
    )

    neighborhoods = collapse_and_trim_neighborhoods(
        neighborhood_list, keep_fraction=0.95, keep_limit=500, target_shape=None
    )

    return {
        "positions": positions,
        "neighborhoods": neighborhoods,
        "radius": cfg.neighborhood_width / 2,
    }


def main():
    args = get_parser().parse_args()
    cfg = load_config_from_yaml(args.config)

    save_dir = POSITION_DIR / cfg.initial_position_dir
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving to {save_dir}")

    placement_configs = get_placement_configs(
        BRAIN_MAPPING[cfg.model.base_model_name],
        OUTPUT_DIMS_FOR_224_INPUTS[cfg.model.base_model_name],
    )

    # save each placement config
    for cfg in placement_configs:
        position_dict = create_position_dict(cfg)
        layer_positions = LayerPositions(
            name=cfg.name,
            dims=cfg.dims,
            coordinates=position_dict["positions"],
            neighborhood_indices=position_dict["neighborhoods"],
            neighborhood_width=position_dict["radius"] * 2,
        )
        layer_positions.save(save_dir)

    version_file = save_dir / "version.txt"
    with open(version_file, "w") as stream:
        stream.write(POS_VERSION)


if __name__ == "__main__":
    main()
