# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Install the badminton task into an existing Isaac Lab environment."""

from pathlib import Path

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = Path(__file__).resolve().parent
METADATA = toml.load(EXTENSION_PATH / "config" / "extension.toml")["package"]

setup(
    name="badmintonlab_game_play",
    version=METADATA["version"],
    description=METADATA["description"],
    keywords=METADATA["keywords"],
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=["gymnasium", "torch"],
    extras_require={"dev": ["build<1.4", "ruff"], "video": ["imageio>=2.31,<3", "imageio-ffmpeg>=0.4"]},
    zip_safe=False,
)
