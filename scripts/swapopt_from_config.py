import argparse
import logging
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from vissl.utils.hydra_config import AttrDict

from spacetorch.swapopt import Swapper
from spacetorch.utils.plot_utils import remove_spines
from spacetorch.utils.generic_utils import load_config_from_yaml

# set up logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)-15s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--feature_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    return parser


def log(to_print: str):
    logger.info(to_print)


def resolve_layer(layers: List[str], cmd_line_layer: str):
    """
    Prefers SLURM_ARRAY_TASK_ID if present, but falls back to command line arg
    """
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        return layers[int(task_id)]

    return cmd_line_layer


def plot_metrics(swapper: Swapper, save_path: Path):
    line_kwargs = {"lw": 1.5, "c": "k"}

    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
    axes[0].plot(swapper.metrics.num_swaps, **line_kwargs)
    axes[1].plot(swapper.metrics.losses, **line_kwargs)

    axes[0].set_ylabel("Number of Swaps")
    axes[1].set_ylabel("Neighborhood Loss")

    remove_spines(axes)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="white")


def main():
    args = get_parser().parse_args()
    cfg: AttrDict = load_config_from_yaml(Path(args.config))

    layer = resolve_layer(cfg.model.layers, args.layer)
    assert layer in cfg.model.layers, "Provided layer not in config model.layers"
    log(f"Constructing swapper for layer {layer}")
    swapper: Swapper = Swapper(
        cfg, feature_path=args.feature_path, layer=layer, dataset_name=args.dataset_name
    )

    # swapper can get blocked if running it would overwrite existing position files
    if not swapper.blocked:
        log("Swapping")
        swapper.swap()

        log("Saving positions back")
        swapper.save_positions()

        log("Saving metrics plot")
        save_path = Path(f"{cfg.name}_{layer}_metrics.png")
        plot_metrics(swapper, save_path)

        log("All done!")


if __name__ == "__main__":
    main()
