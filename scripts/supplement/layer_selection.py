import numpy as np
import pandas as pd

from spacetorch.datasets import sine_gratings, floc
from spacetorch.maps.v1_map import V1Map
from spacetorch.maps.it_map import ITMap
from spacetorch.analyses.core import get_features_from_model, get_positions
from spacetorch.utils import seed_str, array_utils
from spacetorch.paths import RESULTS_DIR

layers = [
    "layer1.0",
    "layer1.1",
    "layer2.0",
    "layer2.1",
    "layer3.0",
    "layer3.1",
    "layer4.0",
    "layer4.1",
]
contrasts = floc.DOMAIN_CONTRASTS


def get_tissues(
    model_name,
    dataset_name,
    response_constructor,
    tissue_constructor,
    positions,
    tiss_kwargs=None,
):
    tiss_kwargs = tiss_kwargs or {}
    feature_dict, _, labels = get_features_from_model(
        model_name,
        dataset_name,
        layers=layers,
        verbose=True,
        return_inputs_and_labels=True,
        step="latest",
    )

    response_dict = {
        layer.split("base_model.")[-1]: response_constructor(features, labels)
        for layer, features in feature_dict.items()
    }

    tissues = {
        layer: tissue_constructor(
            positions[layer].coordinates, responses, cache_id=None, **tiss_kwargs
        )
        for layer, responses in response_dict.items()
    }

    return tissues


def main():
    positions = get_positions("simclr_swap")
    cv_results = {"Seed": [], "Layer": [], "% Selective": []}
    cat_results = {"Seed": [], "Layer": [], "% Selective": [], "Contrast": []}

    for seed in range(5):
        print(seed)
        print("\tLoading v1 tissue")
        model_name = (
            "simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3"
            f"{seed_str(seed)}"
        )
        v1_tissues = get_tissues(
            model_name,
            "SineGrating2019",
            sine_gratings.SineResponses,
            V1Map,
            positions,
            tiss_kwargs={"smooth_orientation_tuning_curves": False},
        )

        print("\tLoading it tissue")
        it_tissues = get_tissues(
            model_name,
            "fLoc",
            floc.fLocResponses,
            ITMap,
            positions,
        )

        # circular variance analysis
        print("\tCV Analysis")
        for layer, tissue in v1_tissues.items():
            cv = tissue.responses.circular_variance

            mean_responses = tissue.responses._data.mean("image_idx").values
            cv = cv[~np.isnan(cv) & (mean_responses > 1)]
            frac_sel = np.mean(cv < 0.6) * 100

            cv_results["Seed"].append(seed)
            cv_results["Layer"].append(layer)
            cv_results["% Selective"].append(frac_sel)

        # cat sel analysis
        print("\tCat Analysis")
        for contrast in contrasts:
            for layer, tissue in it_tissues.items():
                cat_sel = tissue.responses.selectivity(
                    on_categories=contrast.on_categories,
                    selectivity_fn=array_utils.tstat,
                )
                frac_sel = np.mean(cat_sel >= 10) * 100

                cat_results["Seed"].append(seed)
                cat_results["Layer"].append(layer)
                cat_results["% Selective"].append(frac_sel)
                cat_results["Contrast"].append(contrast.name)

    cv_df = pd.DataFrame(cv_results)
    cat_df = pd.DataFrame(cat_results)

    cv_df.to_pickle(RESULTS_DIR / "supp_layer_selection_cv_df.pkl")
    cat_df.to_pickle(RESULTS_DIR / "supp_layer_selection_cat_df.pkl")


if __name__ == "__main__":
    main()
