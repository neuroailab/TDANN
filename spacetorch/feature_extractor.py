"""
Feature Extractor, inspired by Model Tools PytorchWrapper (DiCarlo lab)
"""

import math
from typing import Optional, List, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.utils.generic_utils import make_deterministic


# ReturnedFeatures can be either
# 1. a single np array (single layer only)
# 2. a dictionary of string to np arrays (multiple layers)
# 3. a tuple of ({1, 2}, inputs, labels)
Features = Union[np.ndarray, Dict[str, np.ndarray]]
ReturnedFeatures = Union[Features, Tuple[Features, np.ndarray, np.ndarray]]


class FeatureExtractor:
    """
    Extracts activations from a layer of a model.

    Arguments:
        dataloader : (torch.utils.data.DataLoader) dataloader. assumes images
                     have been transformed correctly (i.e. ToTensor(),
                     Normalize(), Resize(), etc.)
        n_batches  : (int) number of batches to obtain image features
        vectorize  : (boolean) whether to convert layer features into vector
    """

    def __init__(
        self,
        dataloader: DataLoader,
        n_batches: Optional[int] = None,
        vectorize: bool = False,
        verbose: bool = True,
        spatial_mp: bool = False,
    ):
        self.dataloader = dataloader
        self.n_batches = n_batches or len(self.dataloader)
        self.vectorize = vectorize
        self.verbose = verbose
        self.spatial_mp = spatial_mp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_features(
        self,
        model: nn.Module,
        model_layer_strings: Union[str, List[str]],
        return_inputs_and_labels: bool = False,
    ):

        if not isinstance(model_layer_strings, list):
            model_layer_strings = [model_layer_strings]

        assert len(model_layer_strings), "no layer strings provided"

        layer_results: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {
            k: [] for k in model_layer_strings
        }
        hooks = []

        # add forward hooks to each model layer
        for layer_name in model_layer_strings:
            layer = resolve_sequential_module_from_str(model, layer_name)
            hook = self.get_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        # switch model to eval mode
        model.eval()

        self.inputs = list()
        self.labels = list()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.dataloader),
                total=self.n_batches,
                desc="batch",
                disable=not self.verbose,
            ):

                if batch_idx == self.n_batches:
                    break

                x, label_x = batch
                x = x.to(self.device)

                model(x)
                label_x = label_x.cpu().numpy()
                self.inputs.append(x.cpu().numpy())
                self.labels.append(label_x)

        self.labels = np.concatenate(self.labels)
        self.inputs = np.concatenate(self.inputs)

        # Reset forward hook so next time function runs, previous hooks
        # are removed
        for hook in hooks:
            hook.remove()

        self.layer_feats = {k: np.concatenate(v) for k, v in layer_results.items()}

        # corner case: for a single layer, just return features
        if len(self.layer_feats) == 1:
            self.layer_feats = self.layer_feats[model_layer_strings[0]]

        if return_inputs_and_labels:
            return self.layer_feats, self.inputs, self.labels

        return self.layer_feats

    def get_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if self.spatial_mp:
                output = torch.amax(output, dim=(2, 3))
            out = output.cpu().numpy()

            if self.vectorize:
                out = np.reshape(out, (len(out), -1))
            target_dict[name].append(out)

        hook = layer.register_forward_hook(hook_function)
        return hook


def get_features_from_layer(
    model: nn.Module,
    dataset: Dataset,
    model_layer_strings: Union[str, List[str]],
    batch_size: int = 32,
    max_batches: Optional[int] = None,
    return_inputs_and_labels: bool = False,
    shuffle: bool = True,
    verbose: bool = True,
    spatial_mp: bool = False,
) -> ReturnedFeatures:
    """Workhorse feature extractor.

    Inputs:
        model: the pytorch module to run images through
        dataset: dataset to make a dataloader out of, to get image samples
        model_layer_strings: either a single string, or a list of strings indicating
            which layers to get features from.
        batch_size: self-explanatory
        max_batches: either None (iterate the whole dataset), or the maximum number
            of batches to use
        return_inputs_and_labels: if True return a tuple with (Features, inputs, labels)
        shuffle: if True, shuffle the dataset before use
        verbose: whether to print progress with tqdm

    Returns:
        ReturnedFeatures: either a tuple including features, inputs, and labels, or just
            features. Features can be either a single numpy array, or a dictionary
            mapping from layer names to numpy arrays.
    """

    n_images: int = len(dataset)  # type: ignore
    n_batches: int = max_batches or math.ceil(n_images / batch_size)

    make_deterministic(seed=1)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True
    )

    extractor = FeatureExtractor(
        dataloader, n_batches, verbose=verbose, spatial_mp=spatial_mp
    )
    return extractor.extract_features(
        model, model_layer_strings, return_inputs_and_labels=return_inputs_and_labels
    )
