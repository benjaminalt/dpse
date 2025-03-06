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

from spi.utils import transformations
    
class ForceTorque:
    def __init__(self, fx: float, fy: float, fz: float, mx: float, my: float, mz: float):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

    @staticmethod
    def from_parameters(parameters: list):
        return ForceTorque(*parameters)

    def parameters(self) -> list:
        return [self.fx, self.fy, self.fz, self.mx, self.my, self.mz]

    @staticmethod
    def parameter_names() -> list:
        return ["fx", "fy", "fz", "mx", "my", "mz"]

    def scale(self, from_min: list, from_max: list, to_min: list, to_max: list):
        self.fx = transformations.scale(self.fx, from_min[0], from_max[0], to_min[0], to_max[0])
        self.fy = transformations.scale(self.fy, from_min[1], from_max[1], to_min[1], to_max[1])
        self.fz = transformations.scale(self.fz, from_min[2], from_max[2], to_min[2], to_max[2])
        self.mx = transformations.scale(self.mx, from_min[3], from_max[3], to_min[3], to_max[3])
        self.my = transformations.scale(self.my, from_min[4], from_max[4], to_min[4], to_max[4])
        self.mz = transformations.scale(self.mz, from_min[5], from_max[5], to_min[5], to_max[5])
