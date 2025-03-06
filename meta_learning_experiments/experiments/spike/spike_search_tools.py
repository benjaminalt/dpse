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
import os
import tempfile
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from graphmodel.graphmodel.graph_node_move_linear_relative_contact import GraphNodeMoveLinearRelativeContact
from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from spi.physics.apply_physics_move_linear_relative_contact import apply_physics_with_hole
from utils.transformations import euler_zyx_to_quaternion
from spi.simulation.static_simulator import StaticSimulator
from spi.utils.data_io import save_tensors_to_file

from meta_learning_experiments.experiments.common_utils import MetaDataset, success_rate_loss, cycle_time_loss, \
    set_offset_tag, torch_uniform_in_range


def make_grid_spike_pattern(num_points: int, x_left=-0.001, x_right=0.001, y_left=-0.001, y_right=0.001) -> torch.Tensor:
    """
    Make a grid of (relative) Poses in the XY plane, with num_points
    """
    grid_size = int(math.sqrt(num_points))
    grid_points = []
    for x in np.linspace(x_left, x_right, grid_size):
        for y in np.linspace(y_left, y_right, grid_size):
            grid_points.append(torch.tensor([x, y, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32))
    return torch.cat(grid_points, dim=-1)


def random_spike_search_start_point(n: int, experiment_config: dict) -> torch.Tensor:
    point_to_batch = []
    spike_search_config = experiment_config["spike_search_parameters"]
    ml_config = experiment_config["ml"]
    for _ in range(n):
        point_to_position = torch_uniform_in_range(torch.tensor(ml_config["limits_x"])[:, :7])
        theta = torch_uniform_in_range(torch.tensor(experiment_config["point_to_rotation_z_limits"]))
        point_to_orientation = euler_zyx_to_quaternion(torch.tensor([0.0, 0.0, theta]).unsqueeze(0)).squeeze()
        point_to = torch.cat((point_to_position, point_to_orientation), dim=-1)
        point_to_batch.append(point_to)
    return torch.stack(point_to_batch)


def random_spike_search_relative_inputs(n: int, experiment_config: dict) -> torch.Tensor:
    input_batch = []
    spike_search_config = experiment_config["spike_search_parameters"]
    grid_limits = experiment_config["grid_limits"]
    for _ in range(n):
        x_left, y_left = sorted(np.random.uniform(*grid_limits[0], 2))
        x_right, y_right = sorted(np.random.uniform(*grid_limits[1], 2))
        grid_points = make_grid_spike_pattern(spike_search_config["num_spikes"], x_left, x_right, y_left, y_right)
        num_spikes = torch.tensor([spike_search_config["num_spikes"]])
        min_depth = torch_uniform_in_range(torch.tensor(spike_search_config["min_depth_limits"]))
        max_depth = torch_uniform_in_range(torch.tensor(spike_search_config["max_depth_limits"]))
        min_force = torch_uniform_in_range(torch.tensor(spike_search_config["min_force_limits"]))
        max_force = torch_uniform_in_range(torch.tensor(spike_search_config["max_force_limits"]))
        vel = torch_uniform_in_range(torch.tensor(spike_search_config["vel_limits"]))
        acc = torch_uniform_in_range(torch.tensor(spike_search_config["acc_limits"]))
        input_batch.append(torch.cat((grid_points, num_spikes, min_depth, max_depth, min_force, max_force, vel, acc), dim=0))
    return torch.stack(input_batch)


def fixed_spike_search_relative_inputs(n, experiment_config: dict) -> torch.Tensor:
    input_batch = []
    spike_search_config = experiment_config["spike_search_parameters"]
    for _ in range(n):
        grid_points = make_grid_spike_pattern(spike_search_config["num_spikes"], x_left=-0.0015, x_right=0.0015,
                                              y_left=-0.0015, y_right=0.0015)
        num_spikes = torch.tensor([spike_search_config["num_spikes"]])
        min_depth = torch.tensor([-0.002])
        max_depth = torch.tensor([-0.003])
        min_force = torch.tensor([1.0])
        max_force = torch.tensor([3.0])
        vel = torch.tensor([0.02])
        acc = torch.tensor([0.4])
        input_batch.append(torch.cat((grid_points, num_spikes, min_depth, max_depth, min_force, max_force, vel, acc), dim=0))
    return torch.stack(input_batch)


def _augment_dataset(task_idx, task, experiment_config, target_size, tmpdir):
    simulator = StaticSimulator(GraphNodeMoveLinearRelativeContact(),
                                sampling_interval=experiment_config["sampling_interval"], multiproc=False)
    augmented_data = deepcopy(task.data)
    num_data = len(task.data["inputs"])
    hole_pose = task.data["hole_centers"][0]
    inputs = task.data["inputs"][0]
    new_inputs = []
    new_hole_centers = []
    new_start_states = []
    new_sim = []
    new_real = []
    while num_data < target_size * 3:
        new_start_state = torch_uniform_in_range(torch.tensor(experiment_config["start_pose_limits"], dtype=torch.float32))
        too_close_to_hole = False
        # If new start state too close to any of the holes: Pass
        for hole_center in task.data["hole_centers"]:
            if np.linalg.norm((new_start_state[:2].numpy() - hole_center[:2].numpy())) <= 0.0005:
                too_close_to_hole = True
                break
        if too_close_to_hole:
            continue
        new_inputs.append(inputs)
        new_hole_centers.append(hole_pose)
        new_start_states.append(new_start_state)
        simulated_trajectory_tensor = simulator.simulate(inputs, new_start_state,
                                                         max_trajectory_len=experiment_config["trajectory_length"],
                                                         cache=False)[0]
        min_force, max_force = inputs[7:9]
        # Hack: Be a bit lenient with the max force
        max_force = max_force + 2
        simulated_trajectory = Trajectory.from_tensor(simulated_trajectory_tensor)
        traj_physics, success_label = apply_physics_with_hole(experiment_config["physics"], simulated_trajectory,
                                                              Pose.from_parameters(hole_pose),
                                                              experiment_config["hole_parameters"]["radius"],
                                                              min_force, max_force,
                                                              sampling_interval=experiment_config[
                                                                  "sampling_interval"])
        # Assert that MLRC successful --> did NOT find the hole (which is what I want)
        if not success_label:
            fig, ax = plt.subplots(nrows=3, ncols=1)
            ax[0].scatter(task.data["hole_centers"][:, 0], task.data["hole_centers"][:, 1], color="black")
            for hole_center in task.data["hole_centers"]:
                circ = plt.Circle(hole_center[:2], experiment_config["hole_parameters"]["radius"], color="gray", alpha=0.1)
                ax[0].add_artist(circ)
            traj_physics_tensor = traj_physics.to_tensor()
            ax[0].plot(traj_physics_tensor[:, 2], traj_physics_tensor[:, 3], color="red")
            ax[1].plot(range(len(traj_physics_tensor)), traj_physics_tensor[:, 4], color="red", label="Z")
            ax[1].axhline(hole_pose[2])
            ax[2].plot(range(len(traj_physics_tensor)), traj_physics_tensor[:, 2+7+3], label="FZ")
            plt.show()
            raise RuntimeError()
        new_real.append(traj_physics.to_tensor(pad_to=experiment_config["trajectory_length"]))
        new_sim.append(simulated_trajectory_tensor)
        num_data += 1

    augmented_data["inputs"] = torch.cat((augmented_data["inputs"],
                                          torch.stack(new_inputs)), dim=0)
    augmented_data["hole_centers"] = torch.cat((augmented_data["hole_centers"],
                                                torch.stack(new_hole_centers)), dim=0)
    augmented_data["start_states"] = torch.cat((augmented_data["start_states"],
                                                torch.stack(new_start_states)), dim=0)
    augmented_data["sim"] = torch.cat((augmented_data["sim"],
                                       torch.stack(new_sim)), dim=0)
    augmented_data["real"] = torch.cat((augmented_data["real"],
                                        torch.stack(new_real)), dim=0)
    filepath = os.path.join(tmpdir, f"{task_idx}.h5")
    if os.path.exists(filepath):
        os.unlink(filepath)
    save_tensors_to_file(filepath, augmented_data)
    return filepath


def augment_dataset(ds: MetaDataset, experiment_config: dict, target_size=512):
    """
    Return a new dataset with target_size*3 (train/validate/test) samples per task, where the difference is
    filled with simulated data sampled in regions of the space with poor sampling density
    """
    # print(sum(ds[0].data["real"][:, -1, 1]) / len(ds[0].data["real"]))
    tmpdir = tempfile.mkdtemp()
    # pool = Pool(1)
    # with Pool(4) as pool:
    #     augmented_data_filepaths = pool.starmap(_augment_dataset, [(task_idx, task, experiment_config, target_size, tmpdir) for task_idx, task in enumerate(ds)])
    augmented_data_filepaths = [_augment_dataset(task_idx, task, experiment_config, target_size, tmpdir) for task_idx, task in enumerate(ds)]
    return MetaDataset(augmented_data_filepaths)


def regularizer_cdist(x: torch.Tensor, Y: torch.Tensor):
    spike_pos_x = x[:, ::9+11]
    spike_pos_y = x[:, 1::9+11]
    spike_pos = torch.stack((spike_pos_x, spike_pos_y), dim=-1)
    dist_mat = torch.cdist(spike_pos, spike_pos)
    dist_mat_tri = torch.triu(dist_mat, diagonal=1)
    min_distance_between_points = dist_mat_tri[dist_mat_tri.nonzero(as_tuple=True)].min()
    # return 1/(1000 * mean_distance_between_points) + torch.exp(1000 * (mean_distance_between_points - 0.0007))
    return 0.00005 / min_distance_between_points


def regularizer_grid(grid_xy:torch.Tensor, x: torch.Tensor, Y: torch.Tensor):
    spike_pos_x = x[:, ::9+11]
    spike_pos_y = x[:, 1::9+11]
    spike_pos = torch.stack((spike_pos_x, spike_pos_y), dim=-1)
    # dists = ((spike_pos - grid_xy)**2).sum(-1)   # Should now have batch of distances
    return 100000 * torch.nn.MSELoss()(spike_pos, grid_xy)


class SpikeSearchLoss(torch.nn.Module):
    def __init__(self, initial_grid_xy):
        super().__init__()
        self.initial_grid_xy = initial_grid_xy

    def forward(self, x: torch.Tensor, Y: torch.Tensor):
        sr = success_rate_loss(x, Y)
        # reg = regularizer_grid(self.initial_grid_xy, x, Y)
        reg = regularizer_cdist(x, Y)
        # print(reg)
        ct = cycle_time_loss(x, Y)
        return sr + reg


class SpikeSearchLossNSGA(object):
    def __init__(self, initial_grid_xy):
        self.initial_grid_xy = initial_grid_xy
        self.loss_fns = [success_rate_loss]#, functools.partial(regularizer_grid, self.initial_grid_xy)]
        self.loss_weights = [1.0]#, 1.0]


def spike_search_grid_xy_from_input_tensor(batched_input_tensor):
    initial_grid_x = batched_input_tensor[:, ::9 + 11].detach().clone()
    initial_grid_y = batched_input_tensor[:, 1::9 + 11].detach().clone()
    initial_grid_xy = torch.stack((initial_grid_x, initial_grid_y), dim=-1)
    return initial_grid_xy


def get_ml_and_mlrc_node_ids(rps) -> Tuple[int, int]:
    raise NotImplementedError()


def combine_partial_trajectories_to_spike_search_trajectory(ml_trajectories: List[Trajectory],
                                                            mlrc_trajectories: List[Trajectory]) -> Trajectory:
    combined_traj = []
    for i in range(len(ml_trajectories)):
        combined_traj.append(ml_trajectories[i].to_tensor())
        if i >= len(mlrc_trajectories):
            break
        combined_traj.append(mlrc_trajectories[i].to_tensor())
    combined_traj = Trajectory.from_tensor(torch.cat(combined_traj, dim=0))
    combined_traj.success_label = not mlrc_trajectories[-1].success_label   # Spike Search successful if last MLRC failed
    return combined_traj


def execute_spiral_on_real_robot_and_collect_trajectory_from_lar(optimal_param_tensor: torch.Tensor,
                                                                 rps, ws,
                                                                 offset_x: float, offset_y: float,
                                                                 node_id_offset_tag: int, node_id_approach: int,
                                                                 node_id_spike_search_relative: int,
                                                                 ml_node_id: int, mlrc_node_id: int,
                                                                 db_credentials: dict):
        raise NotImplementedError()
