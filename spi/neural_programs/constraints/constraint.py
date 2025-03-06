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

from abc import ABC, abstractmethod
from enum import Enum

import torch


class ConstraintType(Enum):
    GOAL = 0
    TRAJECTORY = 1

    @staticmethod
    def from_string(string: str):
        return ConstraintType[string]


class Constraint(ABC):
    weight = 1.0

    @abstractmethod
    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def from_dict(dic: dict):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass
