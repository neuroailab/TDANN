# Using existing models
The code in this subdirectory is a concise copy of the most important pieces of the TDANN framework. We demonstrate the simplest use case: getting responses and positions for units in a pretrained model to a single RGB image. 

Start by downloading a model checkpoint and the unit positions [here](https://osf.io/64qv3/).

You can then follow along with [this notebook](demo.ipynb) to extract spatially-registered
responses to an example image.
For example, you can use the checkpoint `tdann_data/tdann/checkpoints/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_checkpoints/model_final_checkpoint_phase199.torch` as `weights_path`, and the positions directory `tdann_data/tdann/positions/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0/resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019_NBVER2` as `positions_dir` to test an existing model.

If using a checkpoint that starts with `simclr_spatial`, use the positions directory above (ending in NBVER2).
If using a checkpoint that starts with `supervised_spatial`, use `tdann_data/tdann/positions/supervised_resnet18_lw0/resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019`.

# Training your own models
To train your own models, you'll need three things:
1. A model that is registered with VISSL
2. An instance of the `NetworkPositions` class, saved to disk in a standardized format
3. A YAML config to tell VISSL how to train the network

You'll also need images to train on (e.g. raw ImageNet images), but I assume you have these already.

The easiest way to explain model training is with a simple example. Let's say you want to train a TDANN with the SimCLR objective, but with unit positions distributed according to a random uniform distribution. For simplicity, let's further assume that each layer is assigned to a 10 x 10mm square of simulated cortical tissue. Note this is _not_ how the models in the paper were trained!

## The model
First, you'll need a model function. For this example, you can actually use an existing `spacetorch` model (see `spacetorch/models/trunks/resnet.py`), but it's simple to define your own. I'll base this model off of ResNet-18.

```python
from typing import Dict

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

from vissl.utils.hydra_config import AttrDict
from vissl.models.trunks import register_model_trunk
from vissl.models.model_helpers import Identity

from spacetorch.models.positions import NetworkPositions, LayerPositions

@register_model_trunk("eshednet")
class EshedNet(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        """Create a new EshedNet

        Inputs:
            model_config: an AttrDict (like a dictionary, but with dot syntax support)
                that specifies the parameters for the model trunk. Specifically, we will
                expect that "model_config.TRUNK.TRUNK_PARAMS.position_dir" exists
            
            model_name: VISSL will pass the model name as the second arg, but we don't
                use it for anything in this case
        """
        super(EshedNet, self).__init__()

        self.positions = self._load_positions(model_config.TRUNK.TRUNK_PARAMS.position_dir)
        self.base_model = resnet18(pretrained=False)

        # remove the FC layer, we're not going to need it
        self.base_model.fc = Identity()

    def _load_positions(self, position_dir: Path) -> Dict[str, LayerPositions]:
        """Load the positions from disk, and cast them to `torch.Tensor`s

        Inputs:
            position_dir: the path to the directory holding each of the layer position
                objects

        Returns:
            layer_positions: a dictionary mapping layer names to LayerPositions objects
        """
        assert position_dir.is_dir()
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    # VISSL requires this signature for forward passes
    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[Union[torch.Tensor, Dict[str, Any]]]:
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        maxpool = self.base_model.maxpool(x)

        x_1_0 = self.base_model.layer1[0](maxpool)
        x_1_1 = self.base_model.layer1[1](x_1_0)
        x_2_0 = self.base_model.layer2[0](x_1_1)
        x_2_1 = self.base_model.layer2[1](x_2_0)
        x_3_0 = self.base_model.layer3[0](x_2_1)
        x_3_1 = self.base_model.layer3[1](x_3_0)
        x_4_0 = self.base_model.layer4[0](x_3_1)
        x_4_1 = self.base_model.layer4[1](x_4_0)

        x = self.base_model.avgpool(x_4_1)
        flat_outputs = torch.flatten(x, 1)

        # build a mapping of layer names to (features, positions) pairs
        spatial_outputs = {
            "layer1_0": (x_1_0, self.positions["layer1.0"]),
            "layer1_1": (x_1_1, self.positions["layer1.1"]),
            "layer2_0": (x_2_0, self.positions["layer2.0"]),
            "layer2_1": (x_2_1, self.positions["layer2.1"]),
            "layer3_0": (x_3_0, self.positions["layer3.0"]),
            "layer3_1": (x_3_1, self.positions["layer3.1"]),
            "layer4_0": (x_4_0, self.positions["layer4.0"]),
            "layer4_1": (x_4_1, self.positions["layer4.1"]),
        }

        return [flat_outputs, spatial_outputs]
```

A couple of things worth pointing out:
1. The output of this model function is a list of two elements. The first element is a Tensor containing the model's regular output. The second element is a dictionary mapping layer names to 2-tuples. Each of those tuples has the features for a given layer and the `LayerPositions` object for that layer.
2. The decorator `@register_model_trunk("eshednet")` enables us to refer to this model as "eshednet" later, when we write our training config in YAML. 

## The positions
For each layer, we will produce a `LayerPositions` object, then save one such object to disk for every layer in the model. During training, we just need to point to the directory in which those objects are saved. Here's a simple example:

```python
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.random import default_rng

from spacetorch.models.positions import LayerPositions

# types
DIMS = Tuple[int, int, int]

# constants
SAVE_DIR = Path("eshednet_positions")
SEED = 424
LAYER_SIZE = 10 #mm
LAYER_DIMS: Dict[str, DIMS] = {
    "layer1.0": (64, 56, 56),
    "layer1.1": (64, 56, 56),
    "layer2.0": (128, 28, 28),
    "layer2.1": (128, 28, 28),
    "layer3.0": (256, 14, 14),
    "layer3.1": (256, 14, 14),
    "layer4.0": (512, 7, 7),
    "layer4.1": (512, 7, 7)
}

# generate random coordinates and neighborhoods for each layer
rng = default_rng(seed=SEED)

for name, dims in LAYER_DIMS.items():
    num_units = np.prod(dims)
    coordinates = rng.uniform(low=0, high=LAYER_SIZE, size=(num_units, 2))

    neighborhoods = np.stack([
        rng.choice(
            np.arange(num_units),
            size=(500,),
            replace=False
        )
        for _ in range(20_000)
    ])

    layer_pos = LayerPositions(
        dims=dims,
        name=name,
        coordinates=coordinates,
        neighborhood_indices=neighborhood_indices,
        neighborhood_width=IT_SIZE
    )

    # to save as pickle
    layer_pos.save(SAVE_DIR)

    # to save as npz instead, comment preceding line and uncomment following line
    # layer_pos.save_np(SAVE_DIR)
```

What's going on with `neighborhoods`? For efficiency, the TDANN training loop will select a number of precomputed subsets of units on every batch, and compute the spatial cost for only that subset. In a layer with N units, it is not feasible to compute the full N x N correlation matrix on each batch, so we resort to computing an M x M matrix, where M << N.
In practice, a neighborhood size of around 500 captures a good number of units without increasing training time by too much.
The full repository contains code for producing unit positions and neighborhoods in more sophisiticated ways, e.g., with retinotopic initialization.

## The YAML config
The following file is copied from a real TDANN config, but with even more comments to explain different choices you may want to make. Note that we refer to the model we created and the position save directory we just used.

Let's say this file is saved to `configs/config/eshednet_config.yaml`

```yaml
config:
  CHECKPOINT:
    AUTO_RESUME: true
    # save checkpoints every 5 epochs
    CHECKPOINT_FREQUENCY: 5
    # specify where to save checkpoints to
    DIR: eshednet_checkpoint_dir
  DATA:
    # number of workers to load data; decrease if CPU/RAM are slammed, increase for speed
    NUM_DATALOADER_WORKERS: 4
    TRAIN:
      # batch size for each GPU. We typically train with 4, so effective batch size is 4 * 128 = 512
      BATCHSIZE_PER_REPLICA: 128
      # collate samples using the VISSL simclr collator
      COLLATE_FUNCTION: simclr_collator
      COPY_TO_LOCAL_DISK: false
      DATASET_NAMES:
      - imagenet1k_folder
      DATA_LIMIT: -1
      # specify where to load imagenet images from
      DATA_PATHS:
      - /dataset_dir/imagenet_raw/train
      DATA_SOURCES:
      - disk_folder
      DROP_LAST: true
      LABEL_TYPE: sample_index
      MMAP_MODE: true
      # specify the simclr transforms to use
      TRANSFORMS:
      - name: ImgReplicatePil
        num_times: 2
      - name: RandomResizedCrop
        size: 224
      - name: RandomHorizontalFlip
        p: 0.5
      - name: ImgPilColorDistortion
        strength: 1.0
      - name: ImgPilGaussianBlur
        p: 0.5
        radius_max: 2.0
        radius_min: 0.1
      - name: ToTensor
      - mean:
        - 0.485
        - 0.456
        - 0.406
        name: Normalize
        std:
        - 0.229
        - 0.224
        - 0.225
  DISTRIBUTED:
    BACKEND: nccl
    NUM_NODES: 1
    # specify how many GPUs we'll use
    NUM_PROC_PER_NODE: 4
    RUN_ID: auto
  LOG_FREQUENCY: 500
  # specify that we're using a custom spatial SimCLR loss (registered in the spacetorch code)
  LOSS:
    name: spatial_correlation_simclr_info_nce_loss
    spatial_correlation_simclr_info_nce_loss:
      buffer_params:
        embedding_dim: 128
      # set spatial weight magnitude (alpha) in each layer
      layer_weights:
        layer1_0: 0.25
        layer1_1: 0.25
        layer2_0: 0.25
        layer2_1: 0.25
        layer3_0: 0.25
        layer3_1: 0.25
        layer4_0: 0.25
        layer4_1: 0.25
      # set the number of neighborhoods used in computing the spatial loss on each batch. Anecdotally, this doesn't matter much, and 1 is the fastest we can go
      neighborhoods_per_batch: 1
      temperature: 0.1
      # toggle to "True" to use the "Absolute SL". Defaults to newer "Relative SL"
      use_old_version: false
  MACHINE:
    DEVICE: gpu
  METERS:
    name: ''
  MODEL:
    HEAD:
      PARAMS:
      - - - mlp
          - dims:
            - 512
            - 512
            use_relu: true
            skip_last_layer_relu_bn: False
        - - mlp
          - dims:
            - 512
            - 128
      - - identity
        - {}
    SYNC_BN_CONFIG:
      CONVERT_BN_TO_SYNC_BN: true
      SYNC_BN_TYPE: pytorch
    TRUNK:
      # use the model we registered
      NAME: eshednet
      TRUNK_PARAMS:
        # use the position_dir we created
        position_dir: eshednet_positions
  MULTI_PROCESSING_METHOD: forkserver
  OPTIMIZER:
    momentum: 0.9
    name: sgd
    nesterov: false
    # decide how long to train for
    num_epochs: 200
    param_schedulers:
      lr:
        auto_lr_scaling:
          auto_scale: true
          base_lr_batch_size: 256
          base_value: 0.3
        end_value: 0.0
        name: cosine
        start_value: 0.15
        update_interval: step
    regularize_bias: true
    regularize_bn: false
    use_larc: false
    weight_decay: 1.0e-06
  SEED_VALUE: 0
  TENSORBOARD_SETUP:
    USE_TENSORBOARD: true
  TEST_MODEL: false
  TEST_ONLY: false
  TRAINER:
    # instruct train loop to use custom tdann code
    TRAIN_STEP_NAME: custom_train_step
  VERBOSE: true
```

You are now (finally!) ready to train the model. Let's say you want to run this on GPUs 1, 3, 5, and 7 on your system. Just run
`python train.py config=eshednet_config --gpus 1,3,5,7`