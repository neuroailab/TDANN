import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from spacetorch.utils.generic_utils import make_iterable
from spacetorch.utils.array_utils import norm


def add_horiz(ax: plt.Axes, mean, spread, x1, x2, color="k", skip_mean: bool = False):
    if spread is not None:
        ax.fill_between(
            np.linspace(x1, x2), spread[0], y2=spread[1], color=color, alpha=0.1
        )

    if not skip_mean:
        ax.axhline(mean, c=color, linestyle="dashed", alpha=0.7)


def add_single_point(ax: plt.Axes, x, mean, spread, **kwargs):
    ax.scatter(x, mean, marker="x", s=50, c="k")
    ax.errorbar(
        x, mean, yerr=[[mean - spread[0]], [spread[1] - mean]], c="k", alpha=0.7
    )


def remove_spines(axes, to_remove=None):
    """
    Removes spines from pyplot axis

    Inputs
        axes (list or np.ndarray): list of pyplot axes
        to_remove (str list): can be any combo of "top", "right", "left", "bottom"
    """

    # if axes is a 2d array, flatten
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()

    # enforce inputs as list
    axes = make_iterable(axes)

    if to_remove == "all":
        to_remove = ["top", "right", "left", "bottom"]
    elif to_remove is None:
        to_remove = ["top", "right"]

    for ax in axes:
        for spine in to_remove:
            ax.spines[spine].set_visible(False)


def add_scale_bar(
    ax, width, height=None, patch_kwargs=None, flipud=False, y_start=None
):
    """
    Adds a rectangular scale bar in the bottom right corner, just above x-axis
    Inputs
        width (float): width of bar
        height (float): height of bar, defaults to 0.01 * np.ptp(ylims)
        patch_kwargs (dict): other inputs
        flipud (bool): set to True for imshow
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if patch_kwargs is None:
        patch_kwargs = {"facecolor": "#222222"}

    if height is None:
        height = 0.025 * np.ptp(ylim)

    x_offset = np.ptp(xlim) * 0.04
    start_point = xlim[1] - (width + x_offset)
    y_start = y_start or ylim[0]
    if flipud:
        y_start += height
    xy = (start_point, y_start)
    rect = patches.Rectangle(xy, width, height, clip_on=False, **patch_kwargs)
    ax.add_patch(rect)


def plot_tuning_curve(ax, tuning_curve, mode="polygon", plot_params=None):
    """
    Plots a single tuning curve on the provided axis
    Inputs
        ax (pyplot axis): axis on which to plot the tuning curve
        tuning_curve (N,): the N-element tuning curve to use. Assumes that the
            values span 0 to 180, with the last value missing
        mode (str): if "polygon" plots as a filled radial polygon
        plot_params (dict or None): if provided, a dictionary to pass to either
            plt.plot() or patches.Polygon()
    """
    angles = np.linspace(0, 180, tuning_curve.shape[0] + 1)[:-1]

    if mode == "polygon":
        radians = (angles * np.pi) / 180.0
        if plot_params is None:
            plot_params = {
                "edgecolor": [0.7, 0, 0.3, 1.0],
                "facecolor": [0.7, 0, 0.3, 0.5],
                "linewidth": 3,
            }

        xs = []
        ys = []

        # add value for 0-180
        for angle, response in zip(radians, tuning_curve):
            xs.append(response * np.cos(angle))
            ys.append(response * np.sin(angle))

        # add value for 180-360
        for angle, response in zip(radians, tuning_curve):
            xs.append(response * np.cos(angle + np.pi))
            ys.append(response * np.sin(angle + np.pi))

        vertices = np.stack((xs, ys)).T
        polygon = patches.Polygon(vertices, closed=True, **plot_params)

        # plot invisible vertices so polygon renders
        ax.scatter(vertices[:, 0], vertices[:, 1], s=0)
        ax.add_patch(polygon)

        # add x and y axes
        ax.axvline(0, c="gray")
        ax.axhline(0, c="gray")

        # set axis limits to max of (7, max_response)
        axis_max = np.max((5, np.max(tuning_curve)))
        ax.set_xlim([-axis_max, axis_max])
        ax.set_ylim([-axis_max, axis_max])

        # remove outside border
        ax.axis("off")

        # plot circle contours
        mx = int(np.max(tuning_curve))
        for radius in np.arange(1, mx, step=mx // 3)[:-1]:
            ax.add_patch(
                patches.Circle(
                    (0, 0), radius=radius, fill=False, edgecolor="k", alpha=0.3
                )
            )
            # ax.text(np.sqrt(radius**2 / 2), np.sqrt(radius**2 / 2), f"{radius:d}")
    elif mode == "line":
        if plot_params is None:
            plot_params = {"c": "k", "linewidth": 3}
        ax.plot(np.arange(tuning_curve.shape[0]), tuning_curve, **plot_params)
        remove_spines(ax)
    else:
        raise ValueError(
            (
                f"Mode of {mode} not recognized. Please choose one of 'polygon' "
                "or 'line'"
            )
        )


def plot_torch_image(ax, tensor):
    data = tensor.detach().cpu().numpy()
    data = np.transpose(data, (1, 2, 0))
    data = norm(data)
    ax.imshow(data)


def blank_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
