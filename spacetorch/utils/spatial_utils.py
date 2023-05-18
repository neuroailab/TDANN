"""
Utilities for getting and initializing positions
"""
from dataclasses import dataclass
import logging
from typing import Callable, Optional, Tuple, List
from typing_extensions import Literal

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
import scipy.stats
from shapely.ops import unary_union, polygonize
from shapely import geometry
from sklearn.cluster import KMeans
from spacetorch.types import Dims

from spacetorch.utils.array_utils import sem

# set up logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

# types
PosLims = Tuple[float, float]


@dataclass
class Window:
    indices: np.ndarray
    lims: List[List[float]]
    num_units: int


@dataclass
class WindowParams:
    width: float = 1.5
    window_number_limit: Optional[int] = None
    edge_buffer: int = 0
    unit_number_limit: Optional[int] = 5000


def concave_hull(points, alpha):
    """
    From: https://gist.github.com/dwyerk/10561690

    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = (
        (triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2
        + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2
    ) ** 0.5
    b = (
        (triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2
        + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2
    ) ** 0.5
    c = (
        (triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2
        + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2
    ) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles)


def agg_by_distance(
    distances: np.ndarray,
    values: np.ndarray,
    bin_edges: Optional[np.ndarray] = None,
    num_bins: int = 10,
    agg_fn: Callable[[np.ndarray], float] = np.nanmean,
    spread_fn: Callable[[np.ndarray], float] = sem,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the mean and spread of `values` by binning `distances`, using the
    specified agg_fn and spread_fn

    Inputs
        distances (N,): a flat vector indicating pairwise distance between points.
            Typically this is the flattened lower triangle of a pairwise square matrix,
            excluding the diagonal
        values (N,): a flat vector indicating some pairwise value between units, e.g.,
            the pairwise correlation, or difference in tuning curve peaks
        bin_edges: if provided, `num_bins` is ignored, and bin_edges is not recomputed
        num_bins: how many evenly-spaced bins the distances should be clumped into
        agg_fn: function for aggregating `values` in each bin, defaults to mean
        spread_fn: function for computing spread of `values` in each bin, defaults to
            SEM
    """

    # sort distances and values in order of increasing distance
    dist_sort_ind = np.argsort(distances)
    sorted_distances = distances[dist_sort_ind]
    sorted_values = values[dist_sort_ind]

    # compute evenly-spaced bins for distances
    if bin_edges is None:
        bin_edges = np.histogram_bin_edges(sorted_distances, bins=num_bins)
    else:
        num_bins = len(bin_edges) - 1

    # iterate over bins, computing "means" (can be any aggregated value too) and spreads
    means = np.zeros((num_bins,))
    spreads = np.zeros((num_bins,))

    for i in range(1, num_bins + 1):
        bin_start = bin_edges[i - 1]
        bin_end = bin_edges[i]
        valid_values = sorted_values[
            (sorted_distances >= bin_start) & (sorted_distances < bin_end)
        ]
        if valid_values.shape[0] == 0:
            means[i - 1] = np.nan
            spreads[i - 1] = np.nan
        else:
            means[i - 1] = agg_fn(valid_values)
            spreads[i - 1] = spread_fn(valid_values)

    # return values for each bin along with the computed bin_edges
    return means, spreads, bin_edges


def collapse_and_trim_neighborhoods(
    nb_list,
    keep_fraction=0.95,
    min_nb_size: int = 25,
    keep_limit=None,
    target_shape=None,
):
    """
    The neighborhood generation process leads to a different number
    of units in each neighborhood. This function trims all neighborhoods
    to a common value so that the tensor of indices can have static shape.

    Inputs:
        nb_list(list): a list of vectors of neighborhood indices
        keep_fraction (float: [0, 1]): the fraction of neighborhoods to keep
        keep_limit (int) maximum number of units to keep
        target_shape (n_neighborhoods, n_units): the final shape of the neighborhood
            array
    """

    assert keep_fraction > 0.0 and keep_fraction <= 1.0

    neighborhood_sizes = np.array([len(n) for n in nb_list])
    if target_shape is not None:
        # sort neighborhoods in order from smallest to largest
        sort_ind = np.argsort(neighborhood_sizes)
        target_nbs = target_shape[0]
        target_units = target_shape[1]

        # take biggest neighborhoods to maximize chances of having target_units inside
        neighborhoods_to_keep = sort_ind[-target_nbs:]
        nb_list = np.array(nb_list)[neighborhoods_to_keep]

        n_list = list()
        for indices in nb_list:
            if len(indices) < target_units:
                continue
            indices_to_keep = np.random.choice(
                indices, size=(target_units,), replace=False
            )
            n_list.append(indices_to_keep)
    else:
        logger.info(
            "Neighborhood sizes range from %d to %d, with a median of %d"
            % (
                np.min(neighborhood_sizes),
                np.max(neighborhood_sizes),
                np.median(neighborhood_sizes),
            )
        )

        # convert keep fraction to percentile cutoff
        percentile = (1 - keep_fraction) * 100
        n_size = max(
            int(scipy.stats.scoreatpercentile(neighborhood_sizes, percentile)),
            min_nb_size,
        )
        failing = np.less(neighborhood_sizes, n_size)
        logger.info(
            "Minimum of %d excludes %.1f%% of neighborhoods"
            % (n_size, np.mean(failing) * 100.0)
        )

        # take final unit limit as minimum of n_size and keep_limit
        if keep_limit is None:
            final_limit = n_size
        else:
            final_limit = np.min([n_size, keep_limit])

        # do the trimming
        n_list = list()
        for indices in nb_list:
            if len(indices) < final_limit:
                continue
            indices_to_keep = np.random.choice(
                indices, size=(final_limit,), replace=False
            )
            n_list.append(indices_to_keep)

    logger.info("Kept %d neighborhoods." % len(n_list))
    neighborhoods = np.stack(n_list)
    return neighborhoods


def precompute_neighborhoods(
    positions: np.ndarray, radius: float = 0.5, n_neighborhoods: int = 10
):
    """
    Inputs:
        positions: N x 2 position matrix
        radius: radius for a neighborhood (width will be 2 * radius)
        n_neighborhoods: how many neighborhoods to generate
    """
    start_xs = np.random.uniform(
        low=np.min(positions[:, 0]) + radius,
        high=np.max(positions[:, 0]) - radius,
        size=(n_neighborhoods,),
    )

    start_ys = np.random.uniform(
        low=np.min(positions[:, 1]) + radius,
        high=np.max(positions[:, 1]) - radius,
        size=(n_neighborhoods,),
    )

    neighborhoods = [
        indices_within_limits(
            positions,
            [[xstart - radius, xstart + radius], [ystart - radius, ystart + radius]],
        )
        for (xstart, ystart) in zip(start_xs, start_ys)
    ]
    return neighborhoods


def place_conv(
    dims: Dims,
    pos_lims: Tuple[float, ...],
    offset_pattern: Literal["random", "grid"],
    rf_overlap: float = 0.0,
    flatten: bool = True,
    return_rf_radius: bool = False,
):
    """
    Places units in a conv layer, depends primarly on channel_order

    Inputs
        dims (3,): dimensions of features, (num_channels, x, y)
        pos_lims (2,) or (4,): min and max of cortical sheet, in mm. If two elements,
            limits are symmetrical for rows and columns. If len == 4, they are
            interpreted as [x0, x1, y0, y1]
        offset_pattern: one of 'random' or 'grid'
        rf_overlap: fraction between 0 and 1 for hypercolumns to physically overlap.
            0 means no overlap (default), 1 means perfect overlap
    """

    num_chan, num_x, num_y = dims
    if len(pos_lims) == 2:
        rf_centers, rf_radius = compute_rf_centers(dims, rf_overlap, pos_lims)
        xx, yy = np.meshgrid(rf_centers, rf_centers)
    elif len(pos_lims) == 4:
        x_rf_centers, x_rf_radius = compute_rf_centers(dims, rf_overlap, pos_lims[:2])
        y_rf_centers, y_rf_radius = compute_rf_centers(dims, rf_overlap, pos_lims[2:])
        xx, yy = np.meshgrid(x_rf_centers, y_rf_centers)

        # not sure about this choice, but seems ok for now as long as aspect ratio is
        # defined in the interval [0, 1]
        rf_radius = max(x_rf_radius, y_rf_radius)

    # N x 2 array; x in first col, y in second
    anchors = np.stack((xx.ravel(), yy.ravel()), axis=1)

    # create a set of position offsets dictating how units corresponding to different
    # channels should be moved relative to the position of the anchor. There will be
    # as many offsets as there are anchors, and the size of each offset vector is
    # the number of channels in the layer
    if offset_pattern == "random":
        offsets = [
            np.random.uniform(size=(num_chan, 2), low=-rf_radius, high=rf_radius)
            for _ in range(len(anchors))
        ]
    elif offset_pattern == "grid":
        offsets = [
            grid_pattern(num_chan, extent=rf_radius) for _ in range(len(anchors))
        ]
    else:
        raise Exception("Offset pattern not recognized")

    position_list = []
    for offset, anchor in zip(offsets, anchors):
        position_list.append(anchor + offset)

    # stack all spatial anchors
    positions = np.stack(position_list)

    # move channel to first dimension, so now we have (channel, x * y, 2)
    positions = np.swapaxes(positions, 0, 1)

    # reshape to either (channel * x * y, 2) or (channel, x, y, 2)
    if flatten:
        positions = positions.reshape((-1, 2))
    else:
        positions = positions.reshape((*dims, 2))

    if return_rf_radius:
        return positions, rf_radius

    return positions


def jitter_positions(pos: np.ndarray, jitter: float = 0):
    # add noise to each unit according to jitter
    rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=jitter, size=pos.shape)
    jittered = pos + noise

    # squish: divide the positions by the ratio of new range to old range
    jittered_squished = jittered / (np.ptp(jittered, axis=0) / np.ptp(pos, axis=0))

    # slide: start all positions at 0, then add back the original minimum (probably 0)
    return jittered_squished - np.min(jittered_squished, axis=0) + np.min(pos, axis=0)


def compute_rf_centers(layer_dims, rf_overlap, pos_lims):
    """
    For a conv layer, returns rf_centers given layer dimensions
        and receptive field overlap

    Some algebra for figuring out rf_radius:
    Variables:
        width: how wide each rf is (mm)
        xmax: maximum x (space to play with) (mm)
        overlap: how much each rf should overlap (fraction)
        N: how many rfs we need to fit

        1. Total space is occupied by N receptive fields, but after subtracting overlap
        xmax = (N)(width) - [(N-1)(overlap * width)]
             = (N)(width) - [(N*overlap*width) - (overlap*width)]
             = (N)(width) - (N*overlap*width) + (overlap*width)
             = (width)(N - N*overlap + overlap)
        width = xmax / (N - N*overlap + overlap)


    """
    assert len(layer_dims) > 1

    num_rfs = layer_dims[-1]
    map_width = np.ptp(pos_lims)

    rf_width = map_width / (num_rfs - (num_rfs * rf_overlap) + rf_overlap)
    rf_radius = rf_width / 2.0

    rf_centers = np.linspace(rf_radius, map_width - rf_radius, num_rfs)
    return rf_centers, rf_radius


def indices_within_limits(positions, limits: List[List[float]], unit_limit=None):
    """
    Inputs
        positions: N x 2 matrix of positions, where N is the number of units
        limits:  The ith list indexes the position limits for the ith column of the
            `positions` matrix
        unit_limit: maximum number of units to retain


    Returns
        indices: a 1-D array of indices where positions are within the limits
    """

    indices = np.where(
        (positions[:, 0] >= limits[0][0])
        & (positions[:, 0] <= limits[0][1])
        & (positions[:, 1] >= limits[1][0])
        & (positions[:, 1] <= limits[1][1])
    )[0]

    if isinstance(unit_limit, int) and len(indices) > unit_limit:
        indices = np.random.choice(indices, size=(unit_limit,), replace=False)

    return indices


def grid_pattern(n, extent: float = 1.0):
    """
    Returns grid that is close to square but missing
    some entries to match n
    """

    sq_ceil = int(np.ceil(np.sqrt(n)))

    nx = ny = sq_ceil

    xs = np.linspace(-extent, extent, nx)
    ys = np.linspace(-extent, extent, ny)

    xx, yy = np.meshgrid(xs, ys)
    xx = xx.ravel()[:n]
    yy = yy.ravel()[:n]

    perm = np.random.permutation(n)
    xx = xx[perm]
    yy = yy[perm]

    return np.stack((xx, yy), axis=1)


def get_adjacent_windows(
    positions,
    width=1.0,
    shift=None,
    window_number_limit=None,
    unit_number_limit=None,
    edge_buffer=None,
    spacing=1.0,
) -> List[Window]:
    """
    Returns a list of Window objects
        'window'

    Inputs:
        positions
        width: how wide, in units of x and y, a window should be
        shift: after determining the grid of windows, how much should it be incremented
            in x and y?
        window_number_limit: return at most this many windows (by random subsampling).
            If None, return all windows
        unit_number_limit: return at most this many units per window (by random
            subsampling). If None, return all units
        edge_buffer (int): remove the border of the grid up to this number of boxes,
            e.g., if 2, remove the left two columns, right two columns, top two columns,
            and bottom 2 columns
        spacing (float): if 1.0, has no effect. If less than 1, shrinks windows by this
            fraction such that a gap is present between them. If greater than 1,
            expands windows such that they overlap. Setting this to a value other than
            1.0 overrides the width argument (but uses it as a multiplying factor)
    """
    # num_bins is a 2-element array with the number of bins to create in each dimension
    # of positions
    num_bins = np.floor_divide(np.ptp(positions, axis=0), width).astype(int)

    if shift is None:
        shift = [0, 0]
    x_bin_starts = (
        np.linspace(
            np.min(positions[:, 0]), np.max(positions[:, 0]) - width, num_bins[0]
        )
        + shift[0]
    )
    y_bin_starts = (
        np.linspace(
            np.min(positions[:, 1]), np.max(positions[:, 1]) - width, num_bins[1]
        )
        + shift[1]
    )

    if edge_buffer:
        x_bin_starts = x_bin_starts[edge_buffer:-edge_buffer]
        y_bin_starts = y_bin_starts[edge_buffer:-edge_buffer]

    delta_per_side = width * (1.0 - spacing) / 2

    windows = []
    for x_start in x_bin_starts:
        xlims = [x_start, x_start + width]
        xlims[0] = xlims[0] + delta_per_side
        xlims[1] = xlims[1] - delta_per_side
        for y_start in y_bin_starts:
            ylims = [y_start, y_start + width]

            # adjust limits based on spacing
            ylims[0] = ylims[0] + delta_per_side
            ylims[1] = ylims[1] - delta_per_side

            indices = indices_within_limits(
                positions, [xlims, ylims], unit_limit=unit_number_limit
            )
            num_units = len(indices)
            if num_units < 2:
                continue

            windows.append(
                Window(indices=indices, lims=[xlims, ylims], num_units=num_units)
            )

    if window_number_limit is not None:
        if len(windows) > window_number_limit:
            windows = np.random.choice(
                windows, size=(window_number_limit,), replace=False
            ).tolist()

    return windows


def total_distance_to_nearest_centroid(positions, n_clusters=2) -> float:
    kmeans = KMeans(n_init=10, n_clusters=n_clusters)
    kmeans.fit(positions)

    total_distance = 0
    for cluster_idx in range(n_clusters):
        center = kmeans.cluster_centers_[cluster_idx]
        passing_pos = positions[kmeans.labels_ == cluster_idx]
        distances = cdist(passing_pos, [center]).squeeze()
        total_distance += np.sum(distances)

    return total_distance


def smoothness(curve: np.ndarray) -> float:
    """
    Summary statistic for the smoothness of a curve. Assumes the curve tracks change
    in some quantity as a function of cortical distance
    """
    nans = np.where(np.isnan(curve))[0]
    if len(nans) > 0:
        curve = curve[: nans[0]]

    if np.ptp(curve) == 0:
        return 1

    return (np.max(curve) - curve[0]) / np.max(curve)
