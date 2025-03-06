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

from abc import ABC

import torch
from spi.neural_templates.models.trajectory_loss import QuaternionLoss

from spi.neural_programs.constraints.constraint import Constraint

eps = 1e-9


class PathLengthConstraintCartesian(Constraint, ABC):
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        pos_diff = trajectory[:,1:,2:5] - trajectory[:,:-1,2:5] + eps
        pos_euclidean_distances = torch.sqrt((pos_diff * pos_diff).sum(dim=-1))
        batch_of_lengths = torch.sum(pos_euclidean_distances, dim=1)
        pos_mean_length = torch.mean(batch_of_lengths)
        ori_mean_length = QuaternionLoss()(trajectory[:,1:,5:9], trajectory[:,:-1,5:9])
        return pos_mean_length + ori_mean_length

    @staticmethod
    def from_dict(dic):
        return PathLengthConstraintCartesian(dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "PathLengthConstraintCartesian",
            "weight": self.weight
        }


class PathLengthConstraintConfiguration(Constraint, ABC):
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def loss(self, joint_trajectory: torch.Tensor) -> torch.Tensor:
        joint_pos_diff = joint_trajectory[:, 1:] - joint_trajectory[:, :-1] + eps
        joint_pos_abs_distances = joint_pos_diff.abs().sum(dim=-1)
        batch_of_lengths = torch.sum(joint_pos_abs_distances, dim=1)
        return torch.mean(batch_of_lengths)

    @staticmethod
    def from_dict(dic):
        return PathLengthConstraintConfiguration(dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "PathLengthConstraintConfiguration",
            "weight": self.weight
        }
