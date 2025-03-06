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
from copy import deepcopy

import torch
from graphmodel.graphmodel.graph_node_move_linear import GraphNodeMoveLinear
from spi.neural_programs.program_composite import ProgramComposite
from spi.neural_templates.program_primitive import ProgramPrimitive
from spi.simulation.static_simulator import StaticSimulator

from meta_learning_experiments.experiments.common_utils import torch_uniform_in_range

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SpikeSearch(ProgramComposite):
    def __init__(self, num_spikes, mlrc: ProgramPrimitive, experiment_config):
        ml_limits_x = torch.tensor(experiment_config["ml"]["limits_x"], dtype=torch.float32)
        ml_limits_s = torch.tensor(experiment_config["ml"]["limits_s"], dtype=torch.float32)
        ml_limits_Y = torch.tensor(experiment_config["ml"]["limits_Y"], dtype=torch.float32)
        components = []
        for i in range(num_spikes):
            # Move Linear
            move_linear = ProgramPrimitive("Move Linear", 9, ml_limits_x, ml_limits_s, ml_limits_Y,
                                           StaticSimulator(GraphNodeMoveLinear(), experiment_config["sampling_interval"],
                                                           multiproc=False),
                                           mlrc.model_config)
            move_linear.model_config["seq_len"] = 200
            move_linear.learnable_parameter_gradient_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                                         dtype=torch.float32, device=DEVICE)
            move_linear.set_device(DEVICE)
            components.append(move_linear)
            # MLRC
            contact_motion = deepcopy(mlrc)
            contact_motion.simulator.pool = None    # Disable multiprocessing for optimization
            contact_motion.learnable_parameter_gradient_mask = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                            dtype=torch.float32, device=DEVICE)
            contact_motion.set_device(DEVICE)
            components.append(contact_motion)
        super().__init__(components)
        self.experiment_config = experiment_config
        self.num_spikes = num_spikes
        self.learnable_parameter_gradient_mask = torch.cat([c.learnable_parameter_gradient_mask for c in self.components])

    @staticmethod
    def split_inputs(num_spikes, batched_input_tensor):
        """
        Takes a tensor of shape (batch_size, spike_search_input_dim) à la (ML inputs | MLRC inputs | ML inputs | MLRC inputs | ...)
        Returns 2 * num_spikes lists of shape (batch_size, input_dim)
        """
        split_inputs = []
        current_idx = 0
        for i in range(2 * num_spikes):
            if i % 2 == 0:
                # MLRC
                split_inputs.append(batched_input_tensor[:, current_idx:current_idx+9])
                current_idx += 9
            else:
                split_inputs.append(batched_input_tensor[:, current_idx:current_idx + 11])
                current_idx += 11
        return split_inputs

    @staticmethod
    def make_inputs(experiment_config, num_spikes, random=False):
        """
        Make one long input tensor, consisting of ML inputs | MLRC inputs | ML inputs | MLRC inputs | ...
        """
        template_inputs = []
        if random:
            grid_x = experiment_config["start_pose_limits"][0][0] + torch.rand(int(math.sqrt(num_spikes))) * (experiment_config["start_pose_limits"][1][0] - experiment_config["start_pose_limits"][0][0])
            grid_y = experiment_config["start_pose_limits"][0][1] + torch.rand(int(math.sqrt(num_spikes))) * (experiment_config["start_pose_limits"][1][1] - experiment_config["start_pose_limits"][0][1])
        else:
            grid_x = torch.linspace(experiment_config["start_pose_limits"][0][0], experiment_config["start_pose_limits"][1][0], int(math.sqrt(num_spikes)))
            grid_y = torch.linspace(experiment_config["start_pose_limits"][0][1], experiment_config["start_pose_limits"][1][1], int(math.sqrt(num_spikes)))
        grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y)
        grid = torch.stack((grid_xx, grid_yy), dim=-1).reshape(num_spikes, 2)
        grid = torch.cat((grid, torch.tensor(experiment_config["start_pose_limits"][0][2:]).unsqueeze(0).repeat(num_spikes, 1)), dim=-1)

        for i in range(num_spikes):
            # Move Linear
            move_linear_inputs = torch.cat((grid[i], torch.tensor([1.0, 1.0], dtype=torch.float32)), dim=-1)
            template_inputs.append(move_linear_inputs)
            # MLRC
            template_inputs.append(torch_uniform_in_range(experiment_config["mlrc"]["input_range"]))
        return torch.cat(template_inputs, dim=-1)
