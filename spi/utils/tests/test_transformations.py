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

import unittest
import numpy as np

from pyquaternion.quaternion import Quaternion
from spi.utils import transformations


class TestTransformations(unittest.TestCase):
    def test_rotation_matrix_from_quaternion(self):
        q = Quaternion(w=1, x=0, y=0, z=0)
        self.assertTrue(np.array_equal(transformations.rotation_matrix_from_quaternion(q),
                                       np.eye(3, 3)))

        q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        self.assertTrue(np.array_equal(transformations.rotation_matrix_from_quaternion(q),
                                        np.array([[0,0,1], [1,0,0], [0,1,0]])))

        q = Quaternion(w=0.1, x=0.1, y=0.5, z=0.5)
        self.assertTrue(np.allclose(transformations.rotation_matrix_from_quaternion(q),
                                    np.array([[-0.9230769,  0.0000000,  0.3846154],
                                              [0.3846154,  0.0000000,  0.9230769],
                                              [0.0000000,  1.0000000,  0.0000000]])))

    def test_quaternion_from_rotation_matrix(self):
        m = np.eye(3, 3)
        self.assertEqual(transformations.quaternion_from_rotation_matrix(m), Quaternion(w=1, x=0, y=0, z=0))

        m = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.assertEqual(transformations.quaternion_from_rotation_matrix(m), Quaternion(w=0.5, x=0.5, y=0.5, z=0.5))

        m = np.array([[-0.9230769,  0.0000000,  0.3846154],
                      [0.3846154,  0.0000000,  0.9230769],
                      [0.0000000,  1.0000000,  0.0000000]])
        q = transformations.quaternion_from_rotation_matrix(m)
        q_ref = Quaternion(w=0.1, x=0.1, y=0.5, z=0.5).normalised
        self.assertTrue(np.allclose(list(q),
                                    list(q_ref)))

    def test_euler_zyx_from_quaternion(self):
        q = Quaternion(0.446, 0.739, 0, 0.505)
        euler_zyx = transformations.euler_zyx_from_quaternion(q)
        self.assertTrue(np.allclose(euler_zyx, np.array([1.7091421, -0.8431537, 0.7443184]), atol=1e-3))

    def test_quaternion_from_euler_zyx(self):
        euler_zyx = [1.7091421, -0.8431537, 0.7443184]
        q = transformations.quaternion_from_euler_zyx(*euler_zyx)
        self.assertTrue(np.allclose(q.elements, np.array([0.446, 0.739, 0, 0.505]), atol=1e-3))

    def test_quaternion_euler_roundtrip(self):
        # q = Quaternion(0.446, 0.739, 0, 0.505)
        q = Quaternion(*[4.32978028e-17, -7.07106781e-01, -4.32978028e-17,  7.07106781e-01])
        euler_zyx = transformations.euler_zyx_from_quaternion(q)
        q_new = transformations.quaternion_from_euler_zyx(*euler_zyx)
        self.assertTrue(np.allclose(q.elements, q_new.elements, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
