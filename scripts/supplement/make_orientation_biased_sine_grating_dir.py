"""
This script copies sine grating images from one directory to another, but
restricts the phases to be copied for each orientation as to enforce a
'cardinal orientation bias'.

We use the resulting images to train a variant of the topographic model whose
inputs are orientation-biased sine grating images.
"""

import argparse
from pathlib import Path
import shutil


class SinePath:
    def __init__(self, filename: Path):
        self.filename = filename
        parts = filename.name.split("_")

        self.angle = float(parts[1][:-3])
        self.sf = float(parts[2][:-2])
        self.phase = float(parts[3][:-5])
        self.color_string = parts[4].split(".jpg")[0]
        self.color = 0.0 if self.color_string == "bw" else 1.0

    def __repr__(self) -> str:
        return (
            f"Angle: {self.angle}\n"
            f"SF: {self.sf}\n"
            f"Phase: {self.phase}\n"
            f"Color: {self.color}"
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str)
    parser.add_argument("--dest_dir", type=str)
    return parser.parse_args()


def main():
    args = get_args()

    source_dir = Path(args.source_dir)
    assert source_dir.is_dir(), f"{args.source_dir} does not exist"
    paths = [SinePath(f) for f in source_dir.glob("*.jpg")]

    # set phases that are allowed for each orientation
    allowed_phases = {
        0.0: [0.0, 1.3, 2.5, 3.8, 5.0],
        22.5: [0.0, 1.3, 2.5, 3.8],
        45.0: [0.0, 1.3, 2.5],
        67.5: [0.0, 1.3, 2.5, 3.8],
        90.0: [0.0, 1.3, 2.5, 3.8, 5.0],
        112.5: [0.0, 1.3, 2.5, 3.8],
        135.0: [0.0, 1.3, 2.5],
        157.5: [0.0, 1.3, 2.5, 3.8],
    }

    valid_paths = []
    for path in paths:
        allowed = path.phase in allowed_phases[path.angle]
        if allowed:
            valid_paths.append(path)

    counts = {k: 0 for k in allowed_phases}

    for path in valid_paths:
        counts[path.angle] += 1

    print("final counts: ", counts)

    # prepare destination dir
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in valid_paths:
        shutil.copy(str(path.filename), str(dest_dir))


if __name__ == "__main__":
    main()
