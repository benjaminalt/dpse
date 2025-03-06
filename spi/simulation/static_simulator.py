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

from enum import Enum, auto

import torch
from graphmodel.graphmodel.graph_node_factory import GraphNodeFactory
from multiprocessing import cpu_count, Pool

from graphmodel.graphmodel.graph_node import GraphNode, GraphNodeType

from spi.simulation.differentiable_simulator import DifferentiableSimulator
from spi.utils.pytorch_utils import pad_padded_sequence


class TrajectoryProperties(object):
    def __init__(self, cartesian=True, force_torque=True, gripper=False):
        self.cartesian = cartesian
        self.force_torque = force_torque
        self.gripper = gripper

    def to_dict(self) -> dict:
        return {
            "cartesian": self.cartesian,
            "force_torque": self.force_torque,
            "gripper": self.gripper
        }

    @staticmethod
    def from_dict(dic: dict):
        return TrajectoryProperties(dic["cartesian"], dic["force_torque"], dic["gripper"])


class Group(Enum):
    MANIPULATOR = auto()
    GRIPPER = auto()
    BOTH = auto()


class StaticSimulator(DifferentiableSimulator):
    def __init__(self, graph_node: GraphNode, sampling_interval: float = 0.032, multiproc: bool = True,
                 group: Group = Group.MANIPULATOR, trajectory_properties: TrajectoryProperties = None):
        self.pool = Pool() if multiproc else None
        self.outputs_cached = {}
        self.graph_node = graph_node
        self.sampling_interval = sampling_interval
        self.group = group
        self.trajectory_properties = trajectory_properties if trajectory_properties is not None else TrajectoryProperties()

    @staticmethod
    def _simulate_single(graph_node: GraphNode, inputs_world: torch.Tensor, start_state_world: torch.Tensor,
                         group: Group, trajectory_properties: TrajectoryProperties):

        # For simulation, only a subset of the input parameters are required
        template_inputs = inputs_world
        path = graph_node.simulate(template_inputs, start_state_world)

        # Build trajectory according to group mask
        traj_len = path.size(0)
        eos_labels = torch.zeros(traj_len, 1, device=inputs_world.device)
        eos_labels[-1] = 1.0
        success_probs = torch.zeros(traj_len, 1, device=inputs_world.device)
        traj = torch.cat((eos_labels, success_probs), dim=-1)
        if group == Group.MANIPULATOR or group == Group.BOTH:
            traj = torch.cat((traj, path[:, :7]), dim=-1)
        elif trajectory_properties.cartesian:
            traj = torch.cat((traj, start_state_world[:7].unsqueeze(0).repeat(traj_len, 1)), dim=-1)
        if trajectory_properties.force_torque:
            traj = torch.cat((traj, torch.zeros(traj_len, 6, device=inputs_world.device)), dim=-1)
        if group == Group.GRIPPER:
            traj = torch.cat((traj, path), dim=-1)
        elif group == Group.BOTH:
            traj = torch.cat((traj, path[:, 7].view(-1, 1)), dim=-1)
        elif trajectory_properties.gripper:
            traj = torch.cat((traj, start_state_world[7].view(1, 1).repeat(traj_len, 1)), dim=-1)
        return traj

    def simulate(self, inputs_world: torch.Tensor, start_state_world: torch.Tensor, max_trajectory_len: int = 500,
                 cache: bool = True) -> torch.Tensor:
        """
        :param inputs_world:
        :param start_state_world: [cartesian_pose|gripper_state]
        :param max_trajectory_len: The fixed trajectory length for the batch
        :param cache: Whether or not to cache the results
        :return: Batch of trajectory tensors padded to max_trajectory_length
        """
        orig_device = inputs_world.device
        inputs_world = inputs_world.cpu()
        start_state_world = start_state_world.cpu()
        if len(inputs_world.size()) < 2:
            inputs_world = inputs_world.unsqueeze(0)
            start_state_world = start_state_world.unsqueeze(0)

        not_cached = []
        for i in range(len(inputs_world)):
            key = tuple(torch.cat((inputs_world[i], start_state_world[i]), dim=-1).reshape(-1).tolist())
            if key not in self.outputs_cached.keys():
                not_cached.append((inputs_world[i], start_state_world[i]))

        if self.pool is not None:
            args = [(self.graph_node, inputs_world, point_start_world, self.group, self.trajectory_properties) for inputs_world, point_start_world in not_cached]
            simulated = self.pool.starmap(self._simulate_single, args)
        else:
            simulated = [self._simulate_single(self.graph_node, inputs_world, point_start_world, self.group, self.trajectory_properties) for inputs_world, point_start_world in not_cached]

        # Downsample to 32 ms and cache
        for i in range(len(simulated)):
            assert simulated[i][-1,0] == 1.0
            simulated[i] = self.downsample(simulated[i])
            assert simulated[i][-1,0] == 1.0
            key = tuple(torch.cat(not_cached[i], dim=-1).reshape(-1).tolist())
            self.outputs_cached[key] = simulated[i]

        output_batch = []
        # Pad and transform to point_start
        for batch_idx, input_world in enumerate(inputs_world):
            key = tuple(torch.cat((input_world, start_state_world[batch_idx]), dim=-1).reshape(-1).tolist())
            trajectory = self.outputs_cached[key]
            assert trajectory[-1,0] == 1.0
            padded_trajectory = pad_padded_sequence(trajectory, max_trajectory_len)
            assert padded_trajectory[-1,0] == 1.0
            output_batch.append(padded_trajectory)

        if not cache:
            self.clear_cache()

        return torch.stack(output_batch).to(orig_device)

    def dump_cache(self, filepath: str):
        print("StaticSimulator::dump_cache: Dumping cache to {}".format(filepath))
        torch.save(self.outputs_cached, filepath)

    def load_cache(self, filepath: str):
        print("StaticSimulator::load_cache: Loading cache from {}".format(filepath))
        self.outputs_cached = torch.load(filepath)

    def clear_cache(self):
        self.outputs_cached.clear()

    def downsample(self, traj: torch.Tensor) -> torch.Tensor:
        internal_sampling_interval = 0.016
        if internal_sampling_interval < self.sampling_interval:
            downsampling_factor = int(self.sampling_interval / internal_sampling_interval)
            downsampled = traj[::downsampling_factor]
        elif internal_sampling_interval == self.sampling_interval:
            downsampled = traj
        else:
            raise NotImplementedError("Sampling interval smaller than 16ms, upsampling required")
        downsampled[-1, 0] = 1.0    # EOS marker might have been sampled away
        return downsampled

    @staticmethod
    def make_group_mask(cartesian: bool = False, force_torque: bool = False, gripper_state: bool = False) -> dict:
        return {
            "cartesian": cartesian,
            "force_torque": force_torque,
            "gripper_state": gripper_state
        }

    @staticmethod
    def from_dict(simulator_config: dict):
        graph_node_type = GraphNodeType[simulator_config["graph_node_type"]]
        return StaticSimulator(GraphNodeFactory.create(graph_node_type), simulator_config["sampling_interval"],
                               simulator_config["multiproc"])

    def to_dict(self):
        return {
            "type": "StaticSimulator",
            "graph_node_type": self.graph_node.type().name,
            "sampling_interval": self.sampling_interval,
            "multiproc": self.pool is not None
        }
