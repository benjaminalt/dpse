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

from typing import List

import torch
from spi.neural_programs.constraints.constraint import ConstraintType, Constraint


class ConstraintCuboid(Constraint):
    def __init__(self, constraint_type: ConstraintType, domain_config: dict,
                 extent_min: torch.Tensor, extent_max: torch.Tensor, dims: List[int], weight: float = 1.0):
        self.constraint_type = constraint_type
        self.domain_config = domain_config
        self.extent_min = extent_min
        self.extent_max = extent_max
        self.dims = dims
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        if self.constraint_type == ConstraintType.GOAL:
            dx = torch.max(torch.stack([self.extent_min[0] - trajectory[:, -1, self.dims[0]],
                                        torch.zeros(trajectory.size(0)),
                                        trajectory[:, -1, self.dims[0]] - self.extent_max[0]], dim=-1), dim=-1)[0]
            dy = torch.max(torch.stack([self.extent_min[1] - trajectory[:, -1, self.dims[1]],
                                        torch.zeros(trajectory.size(0)),
                                        trajectory[:, -1, self.dims[1]] - self.extent_max[1]], dim=-1), dim=-1)[0]
            dz = torch.max(torch.stack([self.extent_min[2] - trajectory[:, -1, self.dims[2]],
                                        torch.zeros(trajectory.size(0)),
                                        trajectory[:, -1, self.dims[2]] - self.extent_max[2]], dim=-1), dim=-1)[0]
        else:
            dx = torch.max(torch.stack([self.extent_min[0] - trajectory[:, :, self.dims[0]],
                                        torch.zeros(trajectory.size()[:2]),
                                        trajectory[:, :, self.dims[0]] - self.extent_max[0]], dim=-1), dim=-1)[0]
            dy = torch.max(torch.stack([self.extent_min[1] - trajectory[:, :, self.dims[1]],
                                        torch.zeros(trajectory.size()[:2]),
                                        trajectory[:, :, self.dims[1]] - self.extent_max[1]], dim=-1), dim=-1)[0]
            dz = torch.max(torch.stack([self.extent_min[2] - trajectory[:, :, self.dims[2]],
                                        torch.zeros(trajectory.size()[:2]),
                                        trajectory[:, :, self.dims[2]] - self.extent_max[2]], dim=-1), dim=-1)[0]
        dst = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return torch.max(dst)

    @staticmethod
    def from_dict(dic):
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()


class PositionConstraintCuboid(ConstraintCuboid):
    def __init__(self, constraint_type: ConstraintType, domain_config: dict, extent_min: torch.Tensor,
                 extent_max: torch.Tensor, weight: float = 1.0):
        super(PositionConstraintCuboid, self).__init__(constraint_type, domain_config, extent_min, extent_max,
                                                       [2, 3, 4], weight)

    @staticmethod
    def from_dict(dic):
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()


class ForceConstraintCuboid(ConstraintCuboid):
    def __init__(self, constraint_type: ConstraintType, domain_config: dict, extent_min: torch.Tensor,
                 extent_max: torch.Tensor, weight: float = 1.0):
        super(ForceConstraintCuboid, self).__init__(constraint_type, domain_config, extent_min, extent_max,
                                                    [9, 10, 11], weight)

    @staticmethod
    def from_dict(dic):
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()
