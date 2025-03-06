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

T = 100


def differentiable_len(trajectory_tensor: torch.Tensor) -> torch.Tensor:
    has_batch_dim = len(trajectory_tensor.size()) == 3
    if not has_batch_dim:
        trajectory_tensor = trajectory_tensor.unsqueeze(0)
    not_eos_binary = differentiable_where(trajectory_tensor[:, :, 0], torch.tensor(0.5), torch.tensor(0.0), torch.tensor(1.0))
    length = torch.sum(not_eos_binary, dim=1)
    if not has_batch_dim:
        return length.squeeze()
    return length


def differentiable_where(t: torch.Tensor, threshold: torch.Tensor, value_if: torch.Tensor, value_else: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid((t - threshold) * T) * (value_if - value_else) + value_else