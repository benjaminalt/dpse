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

from pyquaternion.quaternion import Quaternion
import numpy as np

from spi.utils import transformations
from spi.utils.sampling import random_quaternion


class Orientation(object):

    def __init__(self, q: Quaternion):
        self.q = q

    @staticmethod
    def from_parameters(parameters):
        return Orientation(Quaternion(w=parameters[0], x=parameters[1], y=parameters[2], z=parameters[3]))

    def parameters(self):
        return [self.q.w, self.q.x, self.q.y, self.q.z]

    @staticmethod
    def from_euler_zyx(rx, ry, rz, degrees=False):
        return Orientation(transformations.quaternion_from_euler_zyx(rx, ry, rz, degrees))

    @staticmethod
    def from_qxyzw(qx, qy, qz, qw):
        return Orientation(Quaternion(w=qw, x=qx, y=qy, z=qz))

    def to_euler_zyx(self):
        return transformations.euler_zyx_from_quaternion(self.q)

    def to_qxyzw(self):
        return [self.q.x, self.q.y, self.q.z, self.q.w]

    def to_relative(self, reference_orientation):
        return Orientation(self.q * reference_orientation.q.inverse)

    def to_absolute(self, reference_orientation):
        return Orientation(self.q * reference_orientation.q)

    @staticmethod
    def parameter_names():
        return ["qw", "qx", "qy", "qz"]

    def normalize(self):
        self.q = self.q.normalised

    def scale(self, from_min, from_max, to_min, to_max):
        w = transformations.scale(self.q.w, from_min[0], from_max[0], to_min[0], to_max[0])
        x = transformations.scale(self.q.x, from_min[1], from_max[1], to_min[1], to_max[1])
        y = transformations.scale(self.q.y, from_min[2], from_max[2], to_min[2], to_max[2])
        z = transformations.scale(self.q.z, from_min[3], from_max[3], to_min[3], to_max[3])
        self.q = Quaternion(w=w, x=x, y=y, z=z)

    def smoothen(self, other):
        if np.linalg.norm((other.q - self.q).elements) > np.linalg.norm((other.q + self.q).elements):
            self.q = -self.q

    @staticmethod
    def random():
        return Orientation(random_quaternion())

    def __eq__(self, other):
        if type(other) != Orientation:
            return False
        if other is self:
            return True
        return other.q == self.q
