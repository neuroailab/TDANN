"""
PyTorch implementations of spatial loss functions
"""
import torch
import torch.nn as nn

from spacetorch.utils.torch_utils import corrcoef, pdist, pearsonr, lower_tri


def spatial_loss_batch(
    base_loss_fn,
    features: torch.Tensor,
    positions: torch.Tensor,
    neighborhoods: torch.Tensor,
    n_runs: int = 1,
    agg_func=torch.mean,
):
    """
    Wrapper around the base_loss_fn that iterates many times, so an external caller just
    has to call the loss "once" and get back some statistic of multiple runs

    Args are identical to `spatial_correlation_loss`, with two additions:
        n_runs: how many independent runs to compute the loss over
        agg_func: function to aggregate results of each run. If None, return the list.
    """
    scl_list = torch.stack(
        [base_loss_fn(features, positions, neighborhoods) for _ in range(n_runs)]
    )

    if agg_func is None:
        return scl_list

    return agg_func(scl_list)


def spatial_correlation_loss(
    features: torch.Tensor, positions: torch.Tensor, neighborhoods: torch.Tensor
):
    """
    Spatial correlation loss, defined as:

    L = 0.5 * 1 - corr(response_similarity, spatial_similarity)

    for a single, randomly chosen neighborhood
    """
    # pick a random neighborhood
    indices = neighborhoods[torch.randint(len(neighborhoods), (1,))].squeeze()

    # flatten CHW dims and select neighborhood indices
    neighborhood_features = torch.flatten(features, start_dim=1)[:, indices]
    neighborhood_positions = positions[indices]

    # compute spatial and repsonse similarity
    distance_similarity = 1 / (pdist(neighborhood_positions) + 1)
    response_similarity = corrcoef(neighborhood_features.t())

    # we want to maximize the alignment between spatial and response similarity
    similarity_alignment = pearsonr(
        lower_tri(response_similarity), lower_tri(distance_similarity)
    )

    # similarity alignment is in [-1, 1], so we convert to a distance by subtracting
    # from 1.0 The value will be in [0, 2], so we divide by 2.0 to guarantee values in
    # [0, 1]
    loss = (1 - similarity_alignment) / 2.0
    return loss


def old_spatial_correlation_loss(
    features: torch.Tensor, positions: torch.Tensor, neighborhoods: torch.Tensor
):
    """
    Spatial correlation loss used 2020 and earlier -- not corr of corrs, but direct
    subtraction
    """
    # pick a random neighborhood
    indices = neighborhoods[torch.randint(len(neighborhoods), (1,))].squeeze()

    # flatten CHW dims and select neighborhood indices
    neighborhood_features = torch.flatten(features, start_dim=1)[:, indices]
    neighborhood_positions = positions[indices]

    # compute spatial and repsonse similarity
    distance_similarity = 1 / (pdist(neighborhood_positions) + 1)
    response_similarity = corrcoef(neighborhood_features.t())

    loss = torch.abs(lower_tri(response_similarity) - lower_tri(distance_similarity))

    return torch.mean(loss)


class SpatialCorrelationLossModule(nn.Module):
    def __init__(self, neighborhoods_per_batch: int, use_old_version: bool = False):
        super(SpatialCorrelationLossModule, self).__init__()
        self.base_loss = (
            old_spatial_correlation_loss
            if use_old_version
            else spatial_correlation_loss
        )
        self.neighborhoods_per_batch = neighborhoods_per_batch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, features, positions, neighborhoods):
        return spatial_loss_batch(
            self.base_loss,
            features.to(self.device),
            positions.to(self.device),
            neighborhoods.to(self.device),
            n_runs=self.neighborhoods_per_batch,
        )
