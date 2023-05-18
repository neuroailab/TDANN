from typing import Dict

from numpy.random import default_rng
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

import spacetorch.analyses.core as core
from spacetorch.constants import RNG_SEED
from spacetorch.models.positions import LayerPositions
from spacetorch.paths import RESULTS_DIR
from spacetorch.wiring_length import WireLengthExperiment, Shifts

# constants
NUM_PATTERNS = 50

# for V1-like layers
V1__NB_WIDTH = 3.5
V1__KMEANS_THRESH = 0.9
V1__NUM_NBS = 20
V1__PCTILE = 95
V1_SRC = "layer2.0"
V1_TRG = "layer2.1"

# for VTC-like layers
VTC__KMEANS_THRESH = 10.0
VTC__PCTILE = 99
VTC_SRC = "layer4.0"
VTC_TRG = "layer4.1"


def process_model_v1(
    model: nn.Module, positions: Dict[str, LayerPositions]
) -> pd.DataFrame:
    results = {"Wiring Length": [], "Pattern": [], "Window": [], "Shift Direction": []}

    rng = default_rng(seed=RNG_SEED)
    wle = WireLengthExperiment(
        model=model,
        layer_positions=positions,
        source_layer=V1_SRC,
        target_layer=V1_TRG,
        num_patterns=NUM_PATTERNS,
    )

    # run for a number of randomly-selected neighborhoods
    extent = 24.5 * 1.5
    for window in tqdm(range(V1__NUM_NBS), desc="V1 | Neighborhood"):
        start_x = rng.uniform(low=0, high=extent - V1__NB_WIDTH)
        start_y = rng.uniform(low=0, high=extent - V1__NB_WIDTH)
        lims_x = [start_x, start_x + V1__NB_WIDTH]
        lims_y = [start_y, start_y + V1__NB_WIDTH]

        for pattern in range(NUM_PATTERNS):
            for direction in Shifts:
                wl = wle.compute_wl(
                    pattern,
                    kmeans_dist_thresh=V1__KMEANS_THRESH,
                    active_pctile=V1__PCTILE,
                    lims=[lims_x, lims_y],
                    direction=direction,
                )

                results["Wiring Length"].append(wl)
                results["Pattern"].append(pattern)
                results["Window"].append(window)
                results["Shift Direction"].append(str(direction.value))

    return pd.DataFrame(results)


def process_model_vtc(
    model: nn.Module, positions: Dict[str, LayerPositions]
) -> pd.DataFrame:
    results = {"Wiring Length": [], "Pattern": [], "Window": [], "Shift Direction": []}

    wle = WireLengthExperiment(
        model=model,
        layer_positions=positions,
        source_layer=VTC_SRC,
        target_layer=VTC_TRG,
        num_patterns=NUM_PATTERNS,
    )

    for pattern in tqdm(range(NUM_PATTERNS), desc="VTC | Pattern"):
        for direction in Shifts:
            wl = wle.compute_wl(
                pattern,
                kmeans_dist_thresh=VTC__KMEANS_THRESH,
                active_pctile=VTC__PCTILE,
                lims=None,
                direction=direction,
            )

            results["Wiring Length"].append(wl)
            results["Pattern"].append(pattern)
            results["Window"].append(0)
            results["Shift Direction"].append(str(direction.value))

    return pd.DataFrame(results)


def process_model(model_name: str, positions: Dict[str, LayerPositions]):
    # figure out what the save path will be
    save_path = RESULTS_DIR / "wiring_length" / f"{model_name}.csv"
    if save_path.is_file():
        print(f"{model_name} already run; skipping")
        return

    save_path.parent.mkdir(exist_ok=True, parents=True)

    # load model
    model = core.load_model_from_analysis_config(model_name)

    v1_df = process_model_v1(model, positions)
    vtc_df = process_model_vtc(model, positions)

    # concatenate into single df
    v1_df["Layer"] = ["V1"] * len(v1_df)
    vtc_df["Layer"] = ["VTC"] * len(vtc_df)
    df = pd.concat([v1_df, vtc_df], ignore_index=True)

    df.to_csv(save_path)


def main():
    prefix = "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
    abs_prefix = "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl"
    sup_prefix = (
        "supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019"
    )

    base_names = [
        # all alpha values for main model
        f"{prefix}_lw0",
        f"{prefix}_lw01",
        f"{prefix}",
        f"{prefix}_lwx2",
        f"{prefix}_lwx5",
        f"{prefix}_lwx10",
        f"{prefix}_lwx100",
        # alpha = 0.25 for abs and sup models
        f"{abs_prefix}",
        f"{sup_prefix}",
    ]

    seed_mods = []
    for seed in range(1, 5):
        for base_name in base_names:
            seed_mods.append(f"{base_name}_seed_{seed}")

    all_names = seed_mods + base_names

    for name in all_names:
        positions = (
            core.get_positions("simclr_swap")
            if "simclr" in name
            else core.get_positions("supervised_swap")
        )
        try:
            process_model(name, positions)
        except AssertionError:
            print(f"{name} missing analysis config (not trained yet?)")


if __name__ == "__main__":
    main()
