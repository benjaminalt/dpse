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

from typing import Union

import numpy as np
import torch
from utils.utils import differentiable_len
from spi.utils import transformations


def normalize(t: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    return transformations.scale(t, limits[0], limits[1], -1, 1)


def denormalize(t: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    return transformations.scale(t, -1, 1, limits[0], limits[1])


def normalize_traj(traj: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    meta_inf = traj[:, :, :2]
    unscaled_trajectories = traj[:, :, 2:]
    scaled_trajectories = transformations.scale(unscaled_trajectories, limits[0, 2:], limits[1, 2:], -1, 1)
    res = torch.cat((meta_inf, scaled_trajectories), dim=-1)
    return res


def denormalize_traj(traj: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    meta_inf = traj[:, :, :2]
    scaled_trajectories = traj[:, :, 2:]
    unscaled_trajectories = transformations.scale(scaled_trajectories, -1, 1, limits[0, 2:], limits[1, 2:])
    res = torch.cat((meta_inf, unscaled_trajectories), dim=-1)
    return res


def success_probability(trajectory: torch.Tensor) -> torch.Tensor:
    has_batch_dim = len(trajectory.size()) == 3
    if not has_batch_dim:
        trajectory = trajectory.unsqueeze(0)
    return torch.mean(trajectory[:, :, 1], dim=1)


def trajectory_length(trajectory: torch.Tensor) -> Union[int, torch.Tensor]:
    if type(trajectory) == np.ndarray:
        if len(trajectory.shape) > 2:
            raise NotImplementedError("trajectory_lengths expects non-batched numpy arrays")
        eos_index = np.nonzero(trajectory[:, 0])[0]
        if len(eos_index) > 0:
            return eos_index[0]
        return len(trajectory)
    elif type(trajectory) == torch.Tensor:
        has_batch_dim = len(trajectory.size()) == 3
        if not has_batch_dim:
            trajectory = trajectory.unsqueeze(0)
        lengths = differentiable_len(trajectory)
        return lengths if has_batch_dim else lengths.squeeze()
    else:
        raise NotImplementedError(
            "trajectory_length expects inputs of type ndarray or tensor, not {}".format(type(trajectory)))