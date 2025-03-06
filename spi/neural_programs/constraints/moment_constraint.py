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


class MomentConstraint(Constraint):
    def __init__(self, order: int, acc_fn=torch.mean, weight: float = 1.0):
        self.order = order
        self.acc_fn = acc_fn
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        moments = trajectory[:, :, 2:5]
        for i in range(self.order):
            moments = moments[:,1:] - moments[:,:-1]
        return self.acc_fn(moments)

    @staticmethod
    def from_dict(dic):
        if dic["acc_fn"] != "mean":
            raise NotImplementedError("Cannot deserialize MomentConstraint for acc_fn other than 'mean'")
        return MomentConstraint(dic["order"], torch.mean, dic["weight"])

    def to_dict(self) -> dict:
        if self.acc_fn != torch.mean:
            raise NotImplementedError("Cannot serialize MomentConstraint for acc_fn other than torch.mean")
        return {
            "type_id": "MomentConstraint",
            "order": self.order,
            "acc_fn": "mean",
            "weight": self.weight
        }
