# Scripts
1. `spatial_loss_curves/`: scripts to compute the magnitude of the spatial loss throughout training for different classes of models
    1. `by_task_loss.py`: computes spatial loss for both categorization-trained and self-supervised (SL_Rel)
        model variants
    2. `by_spatial_loss.py`: computes spatial loss for the SL_Abs model variant
2. `supplement/`:  scripts to generate the outputs of any figures that only appear in the supplementary information
    1. `training_curves/`: scripts that compute the value of the spatial and task loss components during training, as evidence that they do in fact get minimized
    2. `layer_selection.py`: computes circular variance (CV) and category selectivity for each layer
    3. `make_orientation_biased_sine_grating_dir.py`: one-off data wrangling script that generates a directory of sine grating images with an over-representation of the cardinal orientations
    4. `save_retinal_wave_features.py`: computes and saves (in HDF5 format) the average response of each model unit to each simulated retinal wave
    5. `save_vonenet_features.py`: computes and saves (in HDF5 format) the responses of the VOneNet block to a specified dataset
3. `run_brainscore_benchmarks.py`: runs the specified brainscore benchmark for the specified model (must be a ResNet variant for the layer names to make sense)
4. `save_features_from_config.py`: saves responses (in HDF5 format) of a given model to images in a specified dataset. Can be restricted to a maximum number of total images, e.g., only the first 5000 ImageNet validation set images.
5. `save_retinotopic_positions.py`: initial placement of unit positions in each layer in a retinotopic arrangement. Saves positions for each layer as a separate file, but within a common directory. Each positions file contains coordinates, the neighborhood size, and precomputed neighborhood indices for easy use during training.
6. `swapopt_from_config.py`: given a set of features and a model analysis config (which specifies the initial positions to be "swap optimized"), runs swapopt and saves the resulting set of positions
7. `precompute_wiring.py`: computes the wiring length between a pair of early ("V1-like") model layers, and a pair of later layers ("VTC-like"). Saves results as a pandas-compatible CSV.
8. `precompute_eigvals.py`: computes the eigenspectrum of responses to natural images for a variety of models. Results for each model are saved as a dictionary in the NPZ format. 