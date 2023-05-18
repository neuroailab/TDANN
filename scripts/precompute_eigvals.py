import argparse
import math
import numpy as np

from spacetorch.analyses import alpha, core, dimensionality
from spacetorch.models.trunks.resnet import LAYER_ORDER
from spacetorch.paths import RESULTS_DIR
from spacetorch.utils.generic_utils import seed_str
from spacetorch.utils import array_utils

DATASET = "NSD"

bases = {
    "tdann": ("simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"),
    "categorization": (
        "supswap_supervised/supervised_spatial_resnet18_swappedon_SineGrating2019"
    ),
    "absolute_sl": ("simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl"),
}


def build_name(base_name, suffix, seed):
    # manual override for supervised lw0
    if "supervised" in base_name and suffix == "_lw0":
        return (
            "nonspatial/supervised_lw0/supervised_spatial_resnet18_swappedon_"
            f"SineGrating2019_lw0{seed_str(seed)}"
        )

    # manual override for abs sl (it's the same as rel sl)
    if "old_scl" in base_name and suffix == "_lw0":
        return (
            "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_"
            f"lw0{seed_str(seed)}"
        )

    return f"{base_name}{suffix}{seed_str(seed)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tdann")
    parser.add_argument("--num_images", type=int, default=3000)

    parser.add_argument("--spatial_mp", action="store_true")
    parser.add_argument("--no-spatial_mp", dest="spatial_mp", action="store_false")
    parser.set_defaults(spatial_mp=False)
    return parser.parse_args()


def main():
    args = parse_args()
    spatial_mp_str = "spatial_mp" if args.spatial_mp else "no_pool"
    BASE_NAME = bases.get(args.model)
    assert BASE_NAME is not None

    SAVE_DIR = (
        RESULTS_DIR
        / f"eigvals__{args.num_images}_images__{spatial_mp_str}"
        / DATASET
        / BASE_NAME
    )
    print(f"Save dir is {SAVE_DIR}")

    BATCH_SIZE = 128
    max_batches = math.ceil(args.num_images / BATCH_SIZE)

    for spatial_weight, suffix in alpha.suffix_lookup.items():
        for seed in range(5):
            print(f"Processing alpha = {spatial_weight}, seed {seed}")
            model_name = build_name(BASE_NAME, suffix, seed)
            descriptive_name = f"{BASE_NAME}{suffix}{seed_str(seed)}"
            feature_dict = core.get_features_from_model(
                model_name,
                DATASET,
                verbose=True,
                max_batches=max_batches,
                batch_size=BATCH_SIZE,
                layers=LAYER_ORDER,
                spatial_mp=args.spatial_mp,
            )

            for full_layer_name, layer_features in feature_dict.items():
                short_layer_name = full_layer_name.split("base_model.")[-1]

                save_path = SAVE_DIR / (
                    f"{alpha.name_from_sw[spatial_weight]}_"
                    f"seed_{seed}_"
                    f"{short_layer_name.replace('.', '_')}.npz"
                )
                save_path.parent.mkdir(exist_ok=True, parents=True)

                # features will be images x channels. Transpose so that channels are
                # treated as features, and images are treated as samples
                if len(layer_features.shape) > 2:
                    layer_features = array_utils.flatten(layer_features)

                eigvals = dimensionality.compute_eigvals(layer_features)

                # save the eigvals
                np.savez(
                    save_path,
                    eigvals=eigvals,
                    layer=short_layer_name,
                    spatial_weight=spatial_weight,
                    model_name=descriptive_name,
                    seed=seed,
                )


if __name__ == "__main__":
    main()
