# Installation
Installing the code needed to recreate and reproduce all results in the paper has three components: the basics, the simple, and the complex.

## The Basics
The code provided in this repository was tested with Python 3.8.9, on an Ubuntu 16.04 workstation with an NVIDIA Titan Xp GPU. 

I recommend creating a virtual environment (or conda environment) to install all of the dependencies if possible:
```shell
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## The Complex
Many of the dependencies of this codebase are fickle, in that they depend on your specific operating system and available libraries.
As such, I'm going to list them here, and trust you to install the version that make the most sense for you. Here are the versions I have installed:
* [PyTorch](https://pytorch.org/get-started/locally/) (1.8.0)
* [torchvision](https://pytorch.org/get-started/locally/) (0.9.0)
* [VISSL](https://vissl.ai/) (0.1.6)
* [VOneNet](https://github.com/dicarlolab/vonenet) (0.1.0)
* [(Optional) Brain-Score](https://github.com/brain-score) (1.3)
* [(Optional) Brain-Score Model Tools](https://github.com/brain-score/model-tools) 

> **Note**
> The Brain-Score dependencies are only needed if you plan to run brain-score benchmarks or load pickled `Score` objects from BrainScore. Installing the model-tools package is particularly difficult, since it requires an antiquated version of tensorflow. I recommend cloning it and removing that requirement from their `setup.py`. Sorry.

## The Simple
If installing the Complex stuff went okay, the rest is a breeze.
Just install the requirements of this repo:
```shell
pip install -r requirements.txt
```

and install the `spacetorch` package itself:
```shell
cd tdann/
pip install .
```

## A complete walkthrough
For completeness, here's an end-to-end walkthrough of the installation that led to a version
of the code that could generate all of the figures:

1. Basic setup
```shell
git clone <path to repo>
cd <path to repo>
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install VISSL
```shell
cd ../ && git clone --recursive https://github.com/facebookresearch/vissl.git && cd vissl/
git checkout v0.1.6
git checkout -b v0.1.6
pip install --progress-bar off -r requirements.txt
pip install opencv-python
pip uninstall -y classy_vision
pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d
pip uninstall -y fairscale
pip install fairscale@https://github.com/facebookresearch/fairscale/tarball/df7db85cef7f9c30a5b821007754b96eb1f977b6
pip install -e ".[dev]"
```

3. Reinstall compatible versions of torch:
```shell
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

4. Install VOneNet
```shell
cd ../ && git clone https://github.com/dicarlolab/vonenet.git && cd vonenet
pip install .
```

5. Finish installing dependencies
```shell
cd ../<repo name>
pip install -r requirements.txt
pip install -e .
```