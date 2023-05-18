# A Unifying Principle for the Functional Organization of Visual Cortex
Code accompanying our paper on topographic deep neural networks.

## Authors:
* Eshed Margalit :email:
* Hyodong Lee
* Dawn Finzi
* James J. DiCarlo
* Kalanit Grill-Spector*
* Daniel L. K. Yamins*

*co-senior authors

## Overview of the repository
This repository has the following components:
* `configs/`: 
    * `analysis_configs/`: YAML files that establish the link between checkpoint paths and model names, used during analyses and figure generation
    * `config/`: Training configuration files for use with the VISSL training framework
* `notebooks/`: Jupyter notebooks (saved as Markdown files) that reproduce all figures and supplementary figures in the paper
* `scripts/`: standalone scripts to be run from the command line. Contains a dedicated README explaining each script's purpose.
* `spacetorch/`: installable Python package providing classes and methods for model training, evaluation, plotting, and figure creation
* `train.py`: core model training script, modeled after [this example in the VISSL repository](https://github.com/facebookresearch/vissl/blob/main/tools/run_distributed_engines.py).

## Data and model weights
### Using or training models
You can download model weights [here](https://osf.io/64qv3/), under `tdann_data/tdann/checkpoints`.

Take a look at [the instructions](demo/README.md) for help using the existing models or training your own.

### Datasets
There are a number of datasets used in this work:
| Dataset | Description | Where to get it |
|---------|-------------|---------------|
| ImageNet| Our models were trained on the initial release of ImageNet, i.e., before faces were detected and blurred. We haven't tested our models on the more recent releases. | [https://www.image-net.org/](https://www.image-net.org/) |
|fLoc| The functional localizer (fLoc) images consist of 144 images each of 10 categories, plus a scrambled image category that we ignore in this work. | [http://vpnl.stanford.edu/fLoc/](http://vpnl.stanford.edu/fLoc/) |
|Sine Gratings| A set of sine gratings images at 8 orientations, 8 spatial frequencies, 5 spatial phases, and two types of colors: black/white and red/cyan. | [OSF Download](https://osf.io/64qv3/) under `tdann_data/datasets/sine_grating_images_20190507` | 
|Ecoset| A natural image dataset introduced in Mehrer et al., 2021 | [https://www.kietzmannlab.org/ecoset/](https://www.kietzmannlab.org/ecoset/)|

### Other
The easiest way to get the code to run is to clone the folder structure on the machine the code was tested on.
Download [the `tdann_data` folder](https://osf.io/64qv3/) and place it in a location of your filesystem with plenty of free storage.

Next, set the environment variable `ST_BASE_FS` to point to that directory, either temporarily:
```
export ST_BASE_FS=/path/to/downloaded/dir
```
or permanently, by adding those lines to your shell config file (e.g., `.bashrc`/`.zshrc`).

The `spacetorch` package will look in that folder first to find checkpoints, unit positions, etc.
If you have large datasets (like ImageNet) elsewhere on your filesystem, either edit `paths.py` as needed, or pass different arguments to the Dataset constructors.

## Installation
To install dependencies, follow the instructions in [INSTALL.md](INSTALL.md).

## Citation
Check back soon!