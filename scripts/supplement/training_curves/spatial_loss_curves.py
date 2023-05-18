from pathlib import Path
import re

import pandas as pd
import torch
from spacetorch.paths import CHECKPOINT_DIR, RESULTS_DIR

from spacetorch.utils import seed_str
import spacetorch.utils.vissl.registration  # noqa
from spacetorch.utils.vissl.spatial_loss_eval import SCLEval

LAYERS = [
    "layer1_0",
    "layer1_1",
    "layer2_0",
    "layer2_1",
    "layer3_0",
    "layer3_1",
    "layer4_0",
    "layer4_1",
]


def step(path):
    return int(re.findall(r"\d+", path.stem)[0])


def main():
    input_batch = None
    log_dir_base = str(
        CHECKPOINT_DIR / "simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
    )
    config_base = (
        "simswap_simclr"
        "/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
    )

    # prepare results dataframe
    results = {"Seed": [], "Iteration": [], "Loss": [], "Layer": []}

    for seed in range(5):
        print(f"\n\n====={seed}=========\n\n")
        log_dir = Path(f"{log_dir_base}{seed_str(seed)}_checkpoints")
        config_pth = f"{config_base}{seed_str(seed)}"

        # restrict to every other for speed
        checkpoints = sorted(
            log_dir.glob("*phase*.torch"),
            key=lambda p: int(p.stem.split("phase")[-1]),
        )[::2]

        with torch.no_grad():
            for checkpoint in checkpoints:
                it = step(checkpoint)
                ev = SCLEval(checkpoint, config_pth, input_batch=input_batch)
                input_batch = ev.input_batch

                for layer in LAYERS:
                    loss = ev(layer)

                    results["Seed"].append(seed)
                    results["Iteration"].append(it)
                    results["Loss"].append(loss)
                    results["Layer"].append(layer)

    df = pd.DataFrame(results)
    save_path = RESULTS_DIR / "spatial_loss_results.pkl"
    df.to_pickle(save_path)


if __name__ == "__main__":
    main()
