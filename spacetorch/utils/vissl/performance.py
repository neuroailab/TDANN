import json
from dataclasses import dataclass
from typing_extensions import Literal

import numpy as np

from spacetorch.utils import seed_str

Precision = Literal["top_1", "top_5"]


@dataclass
class Performance:
    spatial_weight: float
    path: str
    model: str
    seed: int = 0

    @property
    def name(self):
        return f"{self.model}_SW_{self.spatial_weight}{seed_str(self.seed)}"

    def __post_init__(self):
        with open(self.path, "r") as stream:
            self.data = [json.loads(line) for line in stream.readlines()]

    def build_curve(self, precision: Precision = "top_1"):
        iterations = []
        accuracies = []

        for point in self.data:
            iteration = point["train_phase_idx"]

            test = point.get("test_accuracy_list_meter")
            if test is None:
                continue

            iterations.append(iteration)
            accuracies.append(test[precision]["0"])

        return iterations, accuracies

    def best(self, precision: Precision):
        _, accs = self.build_curve(precision=precision)
        return np.max(accs)
