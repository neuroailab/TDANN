from typing import Dict
from matplotlib.cm import GnBu
from spacetorch.utils import figure_utils

cmap = GnBu
palette = {
    "SW_0": "#000",
    "SW_0pt1": cmap(0.3),  # 0.1
    "SW_0pt25": cmap(0.43),  # 0.25
    "SW_0pt5": cmap(0.56),  # 0.5
    "SW_1pt25": cmap(0.69),  # 1.25
    "SW_2pt5": cmap(0.84),  # 2.5
    "SW_25": cmap(0.97),  # 25
    "Unoptimized": figure_utils.get_color("Unoptimized"),  # Unopt
}

palette_float = {
    0: "#000",
    0.1: cmap(0.3),  # 0.1
    0.25: cmap(0.43),  # 0.25
    0.5: cmap(0.56),  # 0.5
    1.25: cmap(0.69),  # 1.25
    2.5: cmap(0.84),  # 2.5
    25: cmap(0.97),  # 25
}

name_lookup = {
    "SW_0": r"$\alpha = 0$",
    "SW_0pt1": r"$\alpha = 0.1$",
    "SW_0pt25": r"$\alpha = 0.25$",
    "SW_0pt5": r"$\alpha = 0.5$",
    "SW_1pt25": r"$\alpha = 1.25$",
    "SW_2pt5": r"$\alpha = 2.5$",
    "SW_25": r"$\alpha = 25.0$",
    "Unoptimized": "Unoptimized",
}

suffix_lookup: Dict[float, str] = {
    0.0: "_lw0",
    0.1: "_lw01",
    0.25: "",
    0.5: "_lwx2",
    1.25: "_lwx5",
    2.5: "_lwx10",
    25.0: "_lwx100",
}

sw_from_name = {
    "SW_0": 0,
    "SW_0pt1": 0.1,
    "SW_0pt25": 0.25,
    "SW_0pt5": 0.5,
    "SW_1pt25": 1.25,
    "SW_2pt5": 2.5,
    "SW_25": 25,
}

name_from_sw = {
    0.0: "SW_0",
    0.1: "SW_0pt1",
    0.25: "SW_0pt25",
    0.5: "SW_0pt5",
    1.25: "SW_1pt25",
    2.5: "SW_2pt5",
    25.0: "SW_25",
}
