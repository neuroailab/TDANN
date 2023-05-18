import random
import os
from pathlib import Path
import pickle
from typing import List, Tuple, Union, Any
import yaml

import numpy as np
import torch
from vissl.utils.hydra_config import AttrDict

Iterable = Union[List, Tuple, np.ndarray]


def seed_str(seed: int) -> str:
    return "" if seed == 0 else f"_seed_{seed}"


def disable_result_caching() -> None:
    """
    The result-caching library can mess up processing of results
    if function calls look the same internally. Best to disable it
    completely when using brainscore functionality and eat the performance
    cost
    """
    # os.environ["RESULTCACHING_DISABLE"] = "1"
    os.environ["RESULTCACHING_DISABLE"] = "brainscore.score_model"


def make_deterministic(seed: int = 1):
    """
    Set a bunch of seeds to make the following computation determinstic
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def make_iterable(x) -> Iterable:
    """
    If x is not already array-like, turn it into a list or np.array

    Inputs
        x: either array_like (in which case nothing happens) or non-iterable,
            in which case it gets wrapped by a list
    """

    if not isinstance(x, (list, tuple, np.ndarray)):
        return [x]
    return x


def load_config_from_yaml(path: Union[str, Path]) -> AttrDict:
    """
    Load a yaml config as an AttrDict
    """
    # force path to a Path object
    path = Path(path)

    with path.open("r") as stream:
        raw_dict = yaml.safe_load(stream)

    config = AttrDict(raw_dict)

    return config


def string_to_list(s, delimiter=",", casting_fn=str):
    """Casts (with no checks) each part of a delimited string a member of a list"""
    return [casting_fn(part) for part in s.split(delimiter)]


def castable_to_type(element: Any, typ: type) -> bool:
    """
    Returns True if the element can be safely cast to the specified type
    """
    try:
        typ(element)
    except ValueError:
        return False

    return True


def load_pickle(path):
    path = Path(path)
    with path.open("rb") as stream:
        return pickle.load(stream)


def write_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("wb") as stream:
        return pickle.dump(data, stream)
