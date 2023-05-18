from dataclasses import dataclass
from typing import List, Union, Any
import pandas as pd

from spacetorch.paths import RESULTS_DIR
from spacetorch.utils import generic_utils


@dataclass
class LoadSpec:
    full_name: str
    df_name_key: str
    df_name_val: Any
    seed: int

    @property
    def load_path(self):
        return (
            RESULTS_DIR
            / "wiring_length"
            / f"{self.full_name}{generic_utils.seed_str(self.seed)}.csv"
        )


def load_wiring_length_results(
    model_specs: Union[LoadSpec, List[LoadSpec]]
) -> pd.DataFrame:
    """
    Loads wiring length results for multiple models
    Inputs:
        model_names: (str, str) tuple with the full model name and the short model
        name
    """
    model_specs = generic_utils.make_iterable(model_specs)

    dfs = []
    for spec in model_specs:
        if not spec.load_path.is_file():
            print(f"Requested {spec.full_name}, which does not exist")
            continue

        df = pd.read_csv(spec.load_path)

        # add the name as a new column
        df[spec.df_name_key] = [spec.df_name_val] * len(df)

        # add seed
        df["Seed"] = [spec.seed] * len(df)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
