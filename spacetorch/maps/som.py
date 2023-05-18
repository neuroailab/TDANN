from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from minisom import MiniSom
import numpy as np
from numpy.random import default_rng
from sklearn.decomposition import PCA

from torchvision.models import alexnet
from torch.utils.data import DataLoader

from spacetorch.datasets import DatasetRegistry
from spacetorch.datasets.sine_gratings import SineResponses
from spacetorch.datasets.floc import fLocResponses
from spacetorch.feature_extractor import FeatureExtractor
from spacetorch.maps.it_map import ITMap
from spacetorch.maps.v1_map import V1Map
from spacetorch.analyses.core import get_features_from_model
from spacetorch.paths import CACHE_DIR, RESULTS_DIR
from spacetorch.utils import array_utils, generic_utils


class SOM(ITMap):
    @classmethod
    def from_pickle(cls, load_path: Union[str, Path]):
        som: MiniSom = generic_utils.load_pickle(load_path)
        positions = (
            np.stack(som.get_euclidean_coordinates(), axis=-1).reshape((-1, 2))
            / 128.0
            * 10.0
        )

        # get floc responses. Assume all models were generated with the nonspatial lw0
        cache_path = CACHE_DIR / "nonspatial_simclr_lw0_floc.npz"
        if cache_path.exists():
            floc_features = np.load(cache_path)["floc_features"]
            labels = np.load(cache_path)["labels"]
            print("Loaded SOM features from cache")
        else:
            NONSPATIAL_MODEL_NAME = (
                "nonspatial/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0"
            )
            floc_features, _, labels = get_features_from_model(
                NONSPATIAL_MODEL_NAME,
                layers=["avgpool"],
                dataset_name="fLoc",
                max_batches=None,
                batch_size=144,
                return_inputs_and_labels=True,
            )
            np.savez(cache_path, floc_features=floc_features, labels=labels)
        flat_floc_features = array_utils.flatten(floc_features)
        floc_transformed_features = som.pca.transform(
            flat_floc_features
        )  # type: ignore
        som_responses = np.stack(
            [som.activate(sample) for sample in floc_transformed_features]
        )
        floc_responses = fLocResponses(som_responses, labels)

        return cls(positions, floc_responses)


@dataclass
class SOMParams:

    # effective learning rate
    epsilon: float = 0.02

    # maximum value of theta and phi
    R_theta: float = 1
    R_phi: float = 1

    # size of the cortical sheet, only matters in relation to sigmas
    X: float = 15
    Y: float = 15

    # sigma parameter of guassians used to generate orientations
    sig_a: float = 0.1
    sig_b: float = 0.1

    # sigma parameter of guassians used to generate positions
    sig_x: float = 0.5
    sig_y: float = 0.5

    # global sigma to determine neighborhood function
    sigma: float = 2.5

    # number of dimensions of the feature vector (a, b, sf, color, x, y)
    feature_dim: int = 6

    # number of model neurons on a side
    edge_size: int = 128

    # random seed
    seed: int = 424


class V1SOM:
    def __init__(
        self,
        params: Optional[SOMParams] = None,
        n_training_samples: int = 10_000,
        n_training_iterations: int = 1_000_000,
    ):
        self.params = params or SOMParams()
        self.name = f"{self.__class__.__name__}_seed_{self.params.seed}"

        # figure out what this SOM name should be, check if it already exists
        self.save_path = RESULTS_DIR / "soms" / f"{self.name}.pkl"

        # create the SOM if needed
        self.rng = default_rng(seed=self.params.seed)
        self.som = MiniSom(
            self.params.edge_size,
            self.params.edge_size,
            self.params.feature_dim,
            sigma=self.params.sigma,
            learning_rate=self.params.epsilon,
            random_seed=self.params.seed,
        )
        self._initialize_som_weights()
        self.samples = self._create_samples(n_samples=n_training_samples)
        self._train(n_training_iterations)

    @classmethod
    def build(cls, params: Optional[SOMParams], *args, **kwargs):
        params = params or SOMParams()
        name = f"{cls.__name__}_seed_{params.seed}"
        save_path = RESULTS_DIR / "soms" / f"{name}.pkl"
        if save_path.exists():
            print(f"Loading from disk: {save_path}")
            return generic_utils.load_pickle(save_path)

        return cls(params, *args, **kwargs)

    def save(self, path):
        generic_utils.write_pickle(path, self)

    def make_tissue(self, total_size: float = 4.56, cache_id: Optional[str] = None):
        self.sine_responses = self._get_sine_responses()
        self.positions = (
            np.stack(self.som.get_euclidean_coordinates(), axis=-1).reshape((-1, 2))
            / self.params.edge_size
            * total_size
        )
        self.tissue = V1Map(
            positions=self.positions,
            sine_responses=self.sine_responses,
            cache_id=cache_id,
        )

    def _train(self, num_iterations: int = 1_000_000):
        self.som.train(self.samples, num_iterations, random_order=True, verbose=True)
        self.save(self.save_path)

    # these will need to be defined in child class
    def _initialize_som_weights(self):
        raise NotImplementedError

    def _create_samples(self, n_samples: int):
        raise NotImplementedError

    def _get_sine_responses(self):
        raise NotImplementedError


class FeaturePoorV1SOM(V1SOM):
    def _create_samples(self, n_samples: int):
        samples = []
        for _ in range(n_samples):
            x = self.rng.uniform(high=self.params.X)
            y = self.rng.uniform(high=self.params.Y)

            theta = self.rng.uniform(low=0, high=np.pi)  # ori

            a = self.params.R_theta * np.cos(2 * theta)
            b = self.params.R_theta * np.sin(2 * theta)

            color = 0 if self.rng.random() > 0.5 else 1
            sf = self.rng.uniform(low=0, high=1)

            samples.append([a, b, sf, color, x, y])

        return samples

    def _initialize_som_weights(self):
        # manual initialization of weights
        weights = np.zeros(
            (self.params.edge_size, self.params.edge_size, self.params.feature_dim)
        )
        for k in range(self.params.edge_size):
            for L in range(self.params.edge_size):
                zeta_x = self.rng.normal(loc=0, scale=self.params.sig_x)
                zeta_y = self.rng.normal(loc=0, scale=self.params.sig_y)
                x = k * self.params.X / self.params.edge_size + zeta_x
                y = L * self.params.Y / self.params.edge_size + zeta_y

                theta = self.rng.uniform(low=0, high=np.pi)  # ori
                sf = self.rng.normal(loc=0.5, scale=self.params.sig_a)
                color = self.rng.normal(loc=0.5, scale=self.params.sig_a)

                a = self.rng.normal(loc=0, scale=self.params.sig_a) * np.cos(2 * theta)
                b = self.rng.normal(loc=0, scale=self.params.sig_b) * np.sin(2 * theta)

                weights[k, L] = a, b, sf, color, x, y

        self.som._weights = weights

    def _get_sine_responses(self):
        non_ret_weights = np.reshape(self.som.get_weights()[..., :4], (-1, 4))
        non_ret_weights[:, 2] = non_ret_weights[:, 2] - np.mean(non_ret_weights[:, 2])
        non_ret_weights[:, 3] = non_ret_weights[:, 3] - np.mean(non_ret_weights[:, 3])

        sine_dataset = DatasetRegistry.get("SineGrating2019")
        dataset_iterator = iter(sine_dataset)

        all_labels = []
        responses = []
        for sample in dataset_iterator:
            _, label = sample
            angle, sf, _, color = label
            theta = np.deg2rad(angle)

            a = np.cos(2 * theta)
            b = np.sin(2 * theta)
            normed_sf = sf / 96.9 - 0.5
            normed_color = color - 0.5
            stim_vec = np.array([a, b, normed_sf, normed_color])
            resp = np.dot(non_ret_weights, stim_vec)

            responses.append(resp)
            all_labels.append(label)

        features = np.stack(responses)
        labels = np.stack(all_labels)

        return SineResponses(features, labels)


class FeatureRichV1SOM(V1SOM):
    def _create_samples(self, n_samples: int):
        self.model = alexnet(pretrained=True, progress=True).to("cuda")
        num_batches = 5
        dataloader = DataLoader(
            DatasetRegistry.get("ImageNet"),
            batch_size=100,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        extractor = FeatureExtractor(dataloader, num_batches, verbose=True)
        features = extractor.extract_features(
            self.model,
            "features.1",  # ReLU-1
        )

        flat_features = array_utils.flatten(features)
        print(flat_features.shape)
        self.pca = PCA(n_components=self.params.feature_dim)
        self.pca.fit(flat_features)
        xformed_samples = self.pca.transform(flat_features)
        return xformed_samples

    def _initialize_som_weights(self):
        """Dont do any fancy init"""
        pass

    def _get_sine_responses(self):
        num_batches = 20
        dataloader = DataLoader(
            DatasetRegistry.get("SineGrating2019"),
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        extractor = FeatureExtractor(dataloader, num_batches, verbose=True)
        sine_features, _, labels = extractor.extract_features(
            self.model, "features.1", return_inputs_and_labels=True  # ReLU-1
        )
        flat_features = array_utils.flatten(sine_features)
        xformed_samples = self.pca.transform(flat_features)
        som_responses = np.stack(
            [self.som.activate(sample) for sample in xformed_samples]
        )
        return SineResponses(som_responses, labels)
