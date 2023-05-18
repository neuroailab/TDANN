import os
import torch
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def use_gpus(gpu_num: Optional[str]) -> None:
    """
    Quick wrapper to set GPU visibility for CUDA

    Args
        gpu_num (str): which gpu number to restrict CUDA visibility to. If None, default
        to -1 (use CPU). Comma separate to make multiple GPUs visibile, e.g., '0,1,4'
    """

    if gpu_num is None:
        gpu_num = "-1"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
