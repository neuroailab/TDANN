# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features
"""

import argparse
from typing import Any, List

from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict

from spacetorch.utils import use_gpus

# spacetorch x VISSL
import spacetorch.utils.vissl.registration  # noqa
from spacetorch.utils.vissl.hooks import spatial_hook_generator


def hydra_main(overrides: List[Any]):
    print(f"####### overrides: {overrides}")
    cfg = compose_hydra_configuration(overrides)
    args, config = convert_to_attrdict(cfg)

    launch_distributed(
        cfg=config,
        node_id=args.node_id,
        engine_name=args.engine_name,
        hook_generator=spatial_hook_generator,
    )


if __name__ == "__main__":
    """
    Example usage:

    `python train.py config=test/integration_test/quick_simclr`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        help="comma sep list of GPUs to use, to pass to CUDA_VISIBLE_DEVICES",
    )
    args, overrides = parser.parse_known_args()

    # limit GPU visibility
    use_gpus(args.gpus)
    hydra_main(overrides=overrides)
