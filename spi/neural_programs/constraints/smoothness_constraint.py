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

from spi.neural_programs.constraints.constraint import Constraint

eps = 1e-7


class SmoothnessConstraint(Constraint):
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        pos_deltas = trajectory[:,1:, 2:5] - trajectory[:,:-1,2:5] + eps
        dot_product = torch.bmm(pos_deltas[:, 1:].view(pos_deltas.size(0) * (pos_deltas.size(1) - 1), 1, -1),
                                pos_deltas[:, :-1].view(pos_deltas.size(0) * (pos_deltas.size(1) - 1), -1, 1))
        dot_product = dot_product.view(pos_deltas.size(0), pos_deltas.size(1) - 1)
        norm_a = torch.norm(pos_deltas[:, :-1], dim=-1)
        norm_b = torch.norm(pos_deltas[:, 1:], dim=-1)
        loss_val = 1 - (dot_product / (norm_a * norm_b))
        return loss_val.max()

    @staticmethod
    def from_dict(dic):
        return SmoothnessConstraint(dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "SmoothnessConstraint",
            "weight": self.weight
        }
