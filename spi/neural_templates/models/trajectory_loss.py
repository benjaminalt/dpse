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

import torch.nn
from spi.utils import transformations

eps = 1e-7


class QuaternionLoss(torch.nn.Module):
    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, prediction, label):
        """
        For each sequence in the batch, compute the mean pairwise quaternion distance between prediction and label
        https://math.stackexchange.com/a/90098
        :param prediction: shape (batch_size, seq_len, 4) --> Batches of sequences of quaternions
        :param label: shape (batch_size, seq_len, 4) --> Batches of sequences of quaternions
        :return: shape (batch_size)
        """
        thetas = transformations.quaternion_distance(prediction, label)
        # Want thetas to be all zeros, so use mean squared error for each trajectory
        return torch.nn.L1Loss()(thetas, torch.zeros(thetas.size(), device=thetas.device))


class TrajectoryLoss(torch.nn.Module):
    def __init__(self):
        super(TrajectoryLoss, self).__init__()
        self.component_weights = {
            "position": 1.0,
            "orientation": 1.0,
            "force": 1.0
        }

    def forward(self, x, y):
        """
        :param x: shape (batch_size, seq_len, 14)
        :param y: shape (batch_size, seq_len, 14)
        :return: shape (batch_size), the mean of the separate position, orientation, force and traj. length losses
        """
        pred_positions = x[:,:,:3]
        pred_orientations = x[:,:,3:7]
        pred_forces = x[:,:,7:13]
        label_positions = y[:,:,:3]
        label_orientations = y[:,:,3:7]
        label_forces = y[:,:,7:13]
        position_loss = torch.nn.L1Loss()(pred_positions, label_positions)
        orientation_loss = torch.nn.L1Loss()(pred_orientations, label_orientations)
        force_loss = torch.nn.L1Loss()(pred_forces, label_forces)

        return self.component_weights["position"] * position_loss + self.component_weights["orientation"] * orientation_loss + self.component_weights["force"] * force_loss
