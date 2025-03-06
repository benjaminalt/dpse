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

import torch

from spi.neural_programs.constraints.constraint import ConstraintType, Constraint


class ContactConstraint(Constraint):
    def __init__(self, constraint_type: ConstraintType, target_force: torch.Tensor,
                 dimension: int = 11, weight: float = 1.0):
        self.target_force = target_force
        self.dimension = dimension
        self.constraint_type = constraint_type
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        duration = 3 if self.constraint_type == ConstraintType.GOAL else trajectory.size(1)
        median_force = torch.median(trajectory[:, -duration:, self.dimension])
        return torch.nn.L1Loss()(median_force, self.target_force)

    @staticmethod
    def from_dict(dic):
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()
