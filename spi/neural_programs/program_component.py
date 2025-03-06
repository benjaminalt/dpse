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


from typing import Tuple

import torch


class ProgramComponent(torch.nn.Module):
    def __init__(self, input_size: int, limits_x: torch.Tensor, limits_s: torch.Tensor, limits_Y: torch.Tensor,
                 device=None):
        super().__init__()
        self.input_size = input_size
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.limits_x = limits_x.to(device)
        self.limits_s = limits_s.to(device)
        self.limits_Y = limits_Y.to(device)

    def forward(self, x: torch.Tensor, s_in: torch.Tensor, Y_sim: torch.Tensor = None, denormalize_out=True,
                debug=False) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def set_device(self, device):
        self.device = device
        self.limits_x = self.limits_x.to(device)
        self.limits_s = self.limits_s.to(device)
        self.limits_Y = self.limits_Y.to(device)