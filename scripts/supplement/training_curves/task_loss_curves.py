from pathlib import Path
import re

import pandas as pd
import torch
from spacetorch.paths import RESULTS_DIR

from spacetorch.utils import seed_str
import spacetorch.utils.vissl.registration  # noqa
from spacetorch.utils.vissl.task_loss_eval import TaskLossEval
from spacetorch.paths import CHECKPOINT_DIR


def step(path):
    return int(re.findall(r"\d+", path.stem)[0])


def main():
    input_batch = None
    log_dir_base = (
        f"{CHECKPOINT_DIR}/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
    )
    config_base = (
        "simswap_simclr"
        "/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
    )

    # prepare results dataframe
    results = {"Seed": [], "Iteration": [], "Loss": []}

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
                ev = TaskLossEval(checkpoint, config_pth, input_batch=input_batch)
                loss = ev()
                input_batch = ev.input_batch

                results["Seed"].append(seed)
                results["Iteration"].append(it)
                results["Loss"].append(loss)

                print(it, loss)

    df = pd.DataFrame(results)
    save_path = RESULTS_DIR / "task_loss_results.pkl"
    df.to_pickle(save_path)


if __name__ == "__main__":
    main()
