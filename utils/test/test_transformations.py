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

import math
import unittest

import torch
import numpy as np
import pyquaternion

from spi.utils import transformations


class TestTransformations(unittest.TestCase):
    def setUp(self) -> None:
        torch.random.manual_seed(42)

    def test_quaternion_to_rotation_matrix_correct(self):
        for _ in range(10):
            q_torch = torch.rand(32, 4)
            rotation_mat_torch = transformations.quaternion_to_rotation_matrix(q_torch)
            q = [pyquaternion.Quaternion(*torch_quat) for torch_quat in q_torch]
            rotation_mat = np.array([quat.rotation_matrix for quat in q])
            self.assertTrue(np.allclose(rotation_mat, rotation_mat_torch.numpy(), atol=1e-7))

    def test_quaternion_to_rotation_matrix_grad(self):
        q = torch.rand(32, 4)
        q.requires_grad = True
        rotation_mat = transformations.quaternion_to_rotation_matrix(q)
        torch.nn.MSELoss()(rotation_mat, torch.zeros_like(rotation_mat)).backward()

    def test_quaternion_from_rotation_matrix_correct(self):
        for _ in range(10):
            orig_q = torch.rand(32, 4)
            norm = torch.sqrt(torch.sum(orig_q ** 2, dim=-1, keepdim=True))
            q_n = orig_q / norm
            rotation_mat_torch = transformations.quaternion_to_rotation_matrix(q_n)
            q_torch = transformations.rotation_matrix_to_quaternion(rotation_mat_torch)
            self.assertTrue(torch.allclose(q_n, q_torch, atol=1e-7))

    def test_quaternion_from_rotation_matrix_grad(self):
        rotation_mat_torch = transformations.quaternion_to_rotation_matrix(torch.rand(32, 4))
        rotation_mat_torch.requires_grad = True
        q = transformations.rotation_matrix_to_quaternion(rotation_mat_torch)
        torch.nn.MSELoss()(q, torch.tensor([[1,0,0,0]], dtype=torch.float32).repeat(32, 1)).backward()

    def test_pose_euler_zyx_to_affine(self):
        poses_euler_zyx = torch.tensor([[1.3, 1.2, 1.1, 0.5, 0.3, 0.2],
                                        [-2.4, 1.5, 3.3, 1.5, -0.3, 0.7]])
        truth = torch.tensor([[[0.9362934, -0.0354930,  0.3494209, 1.3],
                               [0.1897961,  0.8882368, -0.4183454, 1.2],
                               [-0.2955202,  0.4580127,  0.8383867, 1.1],
                               [0.0, 0.0, 0.0, 1.0]],
                              [[0.7306817, -0.2710303,  0.6266155, -2.4],
                               [0.6154447, -0.1357996, -0.7763932, 1.5],
                               [0.2955202,  0.9529434,  0.0675778, 3.3],
                               [0.0, 0.0, 0.0, 1.0]]])
        pred = transformations.pose_euler_zyx_to_affine(poses_euler_zyx)
        self.assertTrue(torch.allclose(pred, truth, atol=1e-7))

    def test_affine_to_pose_euler_zyx(self):
        poses_affine = torch.tensor([[[0.9362934, -0.0354930, 0.3494209, 1.3],
                               [0.1897961, 0.8882368, -0.4183454, 1.2],
                               [-0.2955202, 0.4580127, 0.8383867, 1.1],
                               [0.0, 0.0, 0.0, 1.0]],
                              [[0.7306817, -0.2710303, 0.6266155, -2.4],
                               [0.6154447, -0.1357996, -0.7763932, 1.5],
                               [0.2955202, 0.9529434, 0.0675778, 3.3],
                               [0.0, 0.0, 0.0, 1.0]]])
        poses_euler_zyx = torch.tensor([[1.3, 1.2, 1.1, 0.5, 0.3, 0.2],
                                        [-2.4, 1.5, 3.3, 1.5, -0.3, 0.7]])
        pred = transformations.affine_to_pose_euler_zyx(poses_affine)
        self.assertTrue(torch.allclose(pred, poses_euler_zyx, atol=1e-7))

    def test_absolute_to_relative(self):
        reference = torch.tensor([0.07929, -0.49247, 0.22816, 0.72999, -0.13826, -0.13841, -0.65486]).unsqueeze(0)
        pose = torch.tensor([ 4.08442e-01, -1.93581e-01,  2.20000e-01, -9.87328e-01,  7.01736e-17, 5.07392e-17,  1.58694e-01]).unsqueeze(0)
        pose_relative = transformations.absolute_to_relative(pose, reference)
        self.assertTrue(torch.allclose(transformations.pose_to_affine(reference).matmul(transformations.pose_to_affine(pose_relative)),
                                       transformations.pose_to_affine(pose), atol=1e-7))

    def test_relative_to_absolute(self):
        reference = torch.tensor([0.07929, -0.49247, 0.22816, 0.72999, -0.13826, -0.13841, -0.65486]).unsqueeze(0)
        pose_relative = torch.tensor([-0.24321702, 0.35857195, 0.10006929, 0.8246617, 0.11454311, 0.15859707, 0.5307164]).unsqueeze(0)
        pose = transformations.relative_to_absolute(pose_relative, reference)
        self.assertTrue(torch.allclose(pose_relative, transformations.absolute_to_relative(pose, reference), atol=1e-7))

    def test_quaternion_distance(self):
        q1 = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        q2 = torch.tensor([0.7071068, 0, 0, 0.7071068], dtype=torch.float32)
        self.assertTrue(torch.allclose(transformations.quaternion_distance(q1, q2),
                                       torch.tensor(math.pi/2)))
        q3 = torch.tensor([0.9238795, 0, 0.3826834, 0])
        q4 = torch.tensor([0.9914449, 0, -0.1305262, 0])
        self.assertTrue(torch.allclose(transformations.quaternion_distance(q3, q4),
                                       torch.tensor(60 * math.pi / 180)))
        batch_1 = torch.stack((q1, q3))
        batch_2 = torch.stack((q2, q4))
        self.assertTrue(torch.allclose(transformations.quaternion_distance(batch_1, batch_2),
                                       torch.tensor([math.pi/2, 60 * math.pi / 180])))
