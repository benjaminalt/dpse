"""
This file is part of DPSE 

Copyright (C) 2025 ArtiMinds Robotics GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import math
import random
import numpy as np
import torch
from pyquaternion.quaternion import Quaternion

from spi.utils import transformations

dimensions = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def random_quaternion():
    u_1 = random.uniform(0, 1)
    u_2 = random.uniform(0, 1)
    u_3 = random.uniform(0, 1)
    return Quaternion(math.sqrt(1 - u_1) * math.sin(2 * math.pi * u_2),
                         math.sqrt(1 - u_1) * math.cos(2 * math.pi * u_2),
                         math.sqrt(u_1) * math.sin(2 * math.pi * u_3),
                         math.sqrt(u_1) * math.cos(2 * math.pi * u_3))


def random_direction_vector(theta):
    """
    Uniformly sample a direction vector from a cone of angle theta
    See https://math.stackexchange.com/questions/56784/generate-a-random-direction-within-a-cone
    :param theta: Max orientation angle
    :return: A direction vector (normalized)
    """
    z = random.uniform(math.cos(theta), 1)
    phi = random.uniform(0, 2 * math.pi)
    coeff = math.sqrt(1 - z ** 2)
    vec = np.array([coeff * math.cos(phi), coeff * math.sin(phi), z])
    return vec / np.linalg.norm(vec)


def uniform_in_range(range: torch.Tensor) -> torch.Tensor:
    if len(range.size()) > 1:
        return (range[0] + torch.rand(range.size(-1)) * (range[1] - range[0])).float()
    return (range[0] + torch.rand(1) * (range[1] - range[0])).float()