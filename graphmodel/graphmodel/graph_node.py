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


class GraphNodeType(Enum):
    MOVE_LINEAR = 1
    MOVE_LINEAR_RELATIVE = 2
    MOVE_LINEAR_RELATIVE_CONTACT = 3
    GRASP = 4


class GraphNode(ABC):
    @abstractmethod
    def simulate(self, inputs: torch.Tensor, point_from: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def type(self) -> GraphNodeType:
        pass
