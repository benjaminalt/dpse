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

import random

from spi.utils import transformations


class Position(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_relative(self, reference_position):
        return Position(
            self.x - reference_position.x,
            self.y - reference_position.y,
            self.z - reference_position.z
        )
    
    def to_absolute(self, reference_position):
        return Position(
            reference_position.x + self.x,
            reference_position.y + self.y,
            reference_position.z + self.z
        )

    def to_xyz(self):
        return [self.x, self.y, self.z]
    
    def parameters(self):
        return [self.x, self.y, self.z]

    @staticmethod
    def from_parameters(parameters):
        return Position(*parameters)

    @staticmethod
    def parameter_names():
        return ["x", "y", "z"]

    def scale(self, from_min, from_max, to_min, to_max):
        self.x = transformations.scale(self.x, from_min[0], from_max[0], to_min[0], to_max[0])
        self.y = transformations.scale(self.y, from_min[1], from_max[1], to_min[1], to_max[1])
        self.z = transformations.scale(self.z, from_min[2], from_max[2], to_min[2], to_max[2])

    @staticmethod
    def random(x_min, x_max, y_min, y_max, z_min, z_max):
        return Position(random.uniform(x_min, x_max),
                        random.uniform(y_min, y_max),
                        random.uniform(z_min, z_max))

    def __eq__(self, other):
        if type(other) != Position:
            return False
        if other is self:
            return True
        return other.x == self.x and other.y == self.y and other.z == self.z
