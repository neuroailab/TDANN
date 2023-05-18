from pathlib import Path
import re

import pandas as pd
import torch
from spacetorch.paths import CHECKPOINT_DIR, RESULTS_DIR

from spacetorch.utils import seed_str
import spacetorch.utils.vissl.registration  # noqa
from spacetorch.utils.vissl.spatial_loss_eval import SCLEval


def step(path: Path) -> int:
    """Training steps from Path"""
    return int(re.findall(r"\d+", path.stem)[0])


def main():
    # initialize input batch as None, so that it gets computed for the first time
    input_batch = None

    models = {
        "Absolute SL": {
            "log_dir_base": (
                f"{CHECKPOINT_DIR}"
                "/simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl"
            ),
            "config_base": (
                (
                    "old_scl/simclr_spatial_resnet18_"
                    "swappedon_SineGrating2019_old_scl"
                )
            ),
        }
    }

    # prepare results dataframe
    results = {"Model": [], "Seed": [], "Iteration": [], "Loss": []}

    for name, spec in models.items():
        log_dir_base = spec["log_dir_base"]
        config_base = spec["config_base"]

        for seed in range(5):
            print("\n\n==============")
            print(f"{name} | {seed}")
            print("\n\n==============")
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
                    ev = SCLEval(
                        checkpoint,
                        config_pth,
                        input_batch=input_batch,
                        use_old_version=True,
                    )

                    # set the input batch variable so that it's the same for future
                    # checkpoints
                    input_batch = ev.input_batch

                    results["Model"].append(name)
                    results["Seed"].append(seed)
                    results["Iteration"].append(it)
                    results["Loss"].append(ev("layer4_1"))

    df = pd.DataFrame(results)
    save_path = RESULTS_DIR / "old_scl_loss_results.pkl"
    df.to_pickle(save_path)


if __name__ == "__main__":
    main()
