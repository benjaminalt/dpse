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

import time

import torch
import numpy as np

from spi.neural_programs.program_component import ProgramComponent


def optimize(p: ProgramComponent, num_iterations: int, loss_fn, x: torch.Tensor, s_in: torch.Tensor, learning_rate: float,
             patience: int = 10, callback=None):
    print("optimize_inputs_spi::optimize: Starting optimization...")
    start_time = time.time()

    x.requires_grad = True
    p.eval()
    optimizer = torch.optim.Adam([x], lr=learning_rate)

    # schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt) for opt in optimizers]
    # schedulers = [torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda epoch: 0.95) for opt in optimizers]

    min_loss = np.inf
    patience_count = 0

    for i in range(num_iterations):
        iter_start_time = time.time()

        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        Y, s_out = p(x, s_in)
        loss = loss_fn(x, Y)
        loss.backward()

        # Apply gradient masks (depending on the neural template, some input parameters must remain fixed)
        if hasattr(p, "learnable_parameter_gradient_mask"):
            x.grad *= p.learnable_parameter_gradient_mask

        # print(x.grad)

        optimizer.step()

        # Clip
        # x = torch.where(x < p.limits_x[0], p.limits_x[0], x)
        # x = torch.where(x > p.limits_x[1], p.limits_x[1], x)

        # Detach x from the current computation graph. Required to prevent Torch from backpropagating through parts of
        # the graph twice
        # x = x.detach()

        print("[{: >3}] Loss:  {:.4f}, Time: {:.2f}".format(i, loss.item(), time.time() - iter_start_time))

        if callback is not None:
            callback(x.detach().cpu().clone(), Y.detach().cpu().clone(), loss.item())

        if patience_count == patience:
            print("Stopping optimization: Converged")
            break
        if loss < min_loss:  # Improved: Reset patience
            patience_count = 0
            min_loss = loss.item()
        else:  # Did not improve: Consume patience
            patience_count += 1

    print("optimize_inputs_spi::optimize: Optimization finished, took {0:.2f}s".format(time.time() - start_time))
