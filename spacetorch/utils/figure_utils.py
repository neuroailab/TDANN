from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from spacetorch.models.trunks.resnet import LAYER_ORDER
from spacetorch.paths import FIGURE_DIR

SAVE_DIR = FIGURE_DIR / "manuscript_figures_v5"

# size of various figure components
SMALL_SIZE = 5
MEDIUM_SIZE = 6
BIGGER_SIZE = 7


def set_text_sizes():
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams["figure.dpi"] = 300


def move_legend(ax, **kwargs):
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", **kwargs)


def plot_neuron(ax, soma_location, size=1, fill_color="#222", alpha=1, lw=1.5):
    """Plot a cute little neuron with a soma, apical dendrite, and 2 basal dendrites"""
    x, y = soma_location
    top = [x, y + size]
    right = [x + size, y - size]
    left = [x - size, y - size]

    soma = Polygon(
        [top, right, left],
        fill=fill_color is not None,
        facecolor=fill_color,
        alpha=alpha,
        zorder=9,
        edgecolor="k",
        lw=lw,
    )

    top_line = Line2D([x, x], [y + size, y + 3 * size], color="k", alpha=alpha, lw=lw)
    right_line = Line2D(
        [x + size, x + size * 1.2],
        [y - size, y - 1.2 * size],
        color="k",
        alpha=alpha,
        lw=lw,
    )
    left_line = Line2D(
        [x - size, x - size * 1.2],
        [y - size, y - 1.2 * size],
        color="k",
        alpha=alpha,
        lw=lw,
    )

    ax.add_patch(soma)
    ax.add_artist(top_line)
    ax.add_artist(right_line)
    ax.add_artist(left_line)


def get_color(layer_or_spatial_weight):
    SPACENET_COLOR = "#7402E5"
    layer_mapping = cm.YlGnBu
    spacing = np.linspace(0.35, 0.9, 8)
    lookup = {
        layer: layer_mapping(spacing[idx]) for idx, layer in enumerate(LAYER_ORDER)
    }
    lookup["SW_0pt25"] = SPACENET_COLOR
    lookup["TDANN"] = SPACENET_COLOR
    lookup[r"$\alpha = 0.25$"] = SPACENET_COLOR
    lookup["SW_0"] = "#444"
    lookup["SimCLR Functional Only"] = "#444"
    lookup["Supervised Functional Only"] = "#666"
    lookup["Functional Only"] = "#444"
    lookup[r"$\alpha = 0$"] = "#444"
    lookup["Random"] = "#AAA"
    lookup["Unoptimized"] = "#AAA"
    lookup["Macaque V1"] = "#00AB33"
    lookup["Human"] = "#00AB33"
    lookup["ITN"] = "#0E8EDD"
    lookup["Swapopt"] = "#0A4A04"

    lookup["Hand-Crafted SOM"] = "#F14C0C"
    lookup["DNN-SOM"] = "#C41A0F"

    lookup["SimCLR"] = "#7402E5"
    lookup["Self-Supervision"] = SPACENET_COLOR
    lookup["Relative SL"] = SPACENET_COLOR
    lookup["Absolute SL"] = "#B40000"
    lookup["Supervised"] = "#E5A502"
    lookup["Categorization"] = "#E5A502"
    return lookup[layer_or_spatial_weight]


def get_sw(name) -> Optional[float]:
    lookup = {
        "SW_0": 0,
        "SW_0pt1": 0.1,
        "SW_0pt25": 0.25,
        "SW_0pt5": 0.5,
        "SW_1pt25": 1.25,
        "SW_2pt5": 2.5,
        "SW_25": 25,
    }
    return lookup.get(name, None)


def get_label(name):
    lookup = {
        "Random": "Unoptimized",
        "SW_0": "Functional Only",
        "SW_0pt1": r"$\alpha = 0.1$",
        "SW_0pt25": "TDANN",
        "SW_0pt5": r"$\alpha = 0.5$",
        "SW_1pt25": r"$\alpha = 1.25$",
        "SW_2pt5": r"$\alpha = 2.5$",
        "SW_25": r"$\alpha = 25.0$",
        "angles": "Orientation",
        "sfs": "Spatial Frequency",
        "colors": "Chromaticity",
        "top_1": "Top-1",
        "top_5": "Top-5",
    }
    return lookup.get(name, name)


def add_colorbar(fig, ax, mappable, **cbar_kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(
        mappable, cax=cax, extend="both", extendrect=True, **cbar_kwargs
    )


def save(fig, name: str, ext: str = "pdf", dpi: int = 300, close_after: bool = False):
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    path = SAVE_DIR / f"{name}.{ext}"
    path.parent.mkdir(exist_ok=True, parents=True)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    if close_after:
        plt.close(fig)
