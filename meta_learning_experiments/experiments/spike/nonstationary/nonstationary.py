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

import collections
import json
import os
import random
from copy import deepcopy

import numpy as np
from argparse import ArgumentParser

import scipy.stats
import torch
from graphmodel.graphmodel.graph_node_move_linear_relative_contact import GraphNodeMoveLinearRelativeContact
from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from matplotlib import pyplot as plt
from more_itertools import take
from spi.neural_programs.optimize_inputs_spi import optimize
from spi.neural_templates.program_primitive import ProgramPrimitive
from spi.physics.apply_physics_spike_search import apply_physics
from spi.simulation.static_simulator import StaticSimulator
from spi.utils.distributions import GaussianMixture
from spi.visualization.spike_search import NonstationarySpikeSearchAnimation

from meta_learning_experiments.experiments.common_utils import finetune_on_task, torch_uniform_in_range
from meta_learning_experiments.experiments.spike.spike_search import SpikeSearch
from meta_learning_experiments.experiments.spike.spike_search_tools import spike_search_grid_xy_from_input_tensor, \
    SpikeSearchLoss


class BrownianMotion(object):
    """
    Apply brownian motion to the modes of a GaussianMixture
    """
    def __init__(self, initial_distribution: GaussianMixture):
        self.hole_distribution = initial_distribution

    def __iter__(self):
        return self

    def __next__(self):
        offset = scipy.stats.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2) * 0.00000005).rvs()
        # print(offset)
        self.hole_distribution.means += offset  # Covariance 0.005 mm
        # print(self.hole_distribution.means)
        return deepcopy(self.hole_distribution)


class Drift(object):
    """
    Apply drift to the modes of a GaussianMixture
    """
    min_offset_per_iter = 0.00001
    max_offset_per_iter = 0.00005

    def __init__(self, initial_distribution: GaussianMixture):
        self.hole_distribution = initial_distribution
        self.offset = np.random.choice([-1, 1], (2,)) * np.random.uniform(self.min_offset_per_iter, self.max_offset_per_iter, (2,))

    def __iter__(self):
        return self

    def __next__(self):
        self.hole_distribution.means += self.offset
        return deepcopy(self.hole_distribution)


class Shift(object):
    """
    Apply a series of random shifts to the modes of a GaussianMixture
    """
    min_offset = 0.0005
    max_offset = 0.001

    def __init__(self, initial_distribution: GaussianMixture):
        self.hole_distribution = initial_distribution
        self.shift_prob = 0.05

    def __iter__(self):
        return self

    def __next__(self):
        if random.random() < self.shift_prob:
            offset = np.random.choice([-1, 1], (2,)) * np.random.uniform(self.min_offset, self.max_offset, (2,))
            self.hole_distribution.means += offset
        return deepcopy(self.hole_distribution)


def random_start_state(experiment_config: dict) -> torch.Tensor:
    start_state = torch_uniform_in_range(experiment_config["ml"]["limits_s"])
    start_state[2] = experiment_config["real_hole_pose"][2] + 0.0005 # Start 0.5 mm above hole plane
    return start_state


def main(args):
    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)
    with open(args.mlrc_model_config) as model_config_file:
        mlrc_model_config = json.load(model_config_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for experiment_idx in range(len(os.listdir(args.output_dir)), args.num_experiments):

        learned_mlrc = ProgramPrimitive.load(args.trained_mlrc_dir)

        limits_x = torch.tensor(experiment_config["mlrc"]["limits_x"], dtype=torch.float32)
        limits_s = torch.tensor(experiment_config["mlrc"]["limits_s"], dtype=torch.float32)
        limits_Y = torch.tensor(experiment_config["mlrc"]["limits_Y"], dtype=torch.float32)
        sim_mlrc = ProgramPrimitive("Move Linear Relative Contact", 7 + 2 + 2, limits_x, limits_s, limits_Y,
                                    StaticSimulator(GraphNodeMoveLinearRelativeContact(),
                                                    sampling_interval=experiment_config["sampling_interval"],
                                                    multiproc=False),
                                    mlrc_model_config, model=None)
        sim_spike = SpikeSearch(16, sim_mlrc, experiment_config)

        # Set up hole process
        xmin, xmax = experiment_config["hole_parameters"]["pos_x"]
        ymin, ymax = experiment_config["hole_parameters"]["pos_y"]
        initial_distribution = GaussianMixture.make_random(1, 6, xmin, xmax, ymin, ymax)
        if args.process_type == "brownian":
            hole_process = BrownianMotion(initial_distribution)
        elif args.process_type == "drift":
            hole_process = Drift(initial_distribution)
        elif args.process_type == "shift":
            hole_process = Shift(initial_distribution)
        else:
            raise ValueError("Invalid process type")

        training_data = collections.deque(maxlen=128 + 64)
        inputs = SpikeSearch.make_inputs(experiment_config, num_spikes=16, random=False).unsqueeze(
            0)  # Start with arbitrary reasonable inputs
        start_state = random_start_state(experiment_config).unsqueeze(0)
        hole_poses_over_time = []
        spike_points_over_time = []
        sim_over_time = []
        real_over_time = []
        actual_hole_over_time = []
        hole_distributions_over_time = []
        success_label_over_time = []

        orig_inputs = deepcopy(inputs)
        inference_enabled = False

        for i, hole_distribution in enumerate(take(args.num_steps, hole_process)):   # 100 steps of nonstationary process
            hole_distributions_over_time.append(hole_distribution)
            random_samples = hole_distribution.sample(100)
            hole_poses_over_time.append([Pose.from_parameters([*sample, experiment_config["real_hole_pose"][2], 1, 0, 0, 0]) for sample in random_samples])

            # Infer optimal inputs
            if not args.skip_optimization:
                initial_grid_xy = spike_search_grid_xy_from_input_tensor(inputs)    # TODO: or orig_inputs?
                loss_fn = SpikeSearchLoss(initial_grid_xy)
                learned_mlrc.set_device(torch.device("cpu"))
                learned_spike = SpikeSearch(16, learned_mlrc, experiment_config)
                if inference_enabled:
                    optimize(learned_spike, num_iterations=30, loss_fn=loss_fn, x=inputs, s_in=start_state, learning_rate=4e-5,
                             patience=50)

            # Simulate spike with these inputs
            Y_out_sim, s_out_sim = sim_spike(inputs, start_state)
            sim_over_time.append(Y_out_sim[0].tolist())
            split_inputs = SpikeSearch.split_inputs(sim_spike.num_spikes, inputs.detach().clone())
            move_linear_inputs = split_inputs[::2]
            mlrc_inputs = split_inputs[1::2]
            spike_points_over_time.append([Pose.from_parameters(move_linear_input[0, :7].detach().cpu()) for move_linear_input in move_linear_inputs])
            if args.debug:
                for spike_points in spike_points_over_time:
                    print(" ".join([f"{spike_point.position.x:.5f}" for spike_point in spike_points]))
            min_force, max_force = mlrc_inputs[0][0, -4:-2]
            move_linear_vel, move_linear_acc = move_linear_inputs[0][0, -2:]

            hole_x, hole_y = hole_distribution.sample(1)[0]
            actual_hole_over_time.append([hole_x, hole_y])
            traj_physics, success_label = apply_physics(experiment_config["physics"], Trajectory.from_tensor(Y_out_sim.squeeze()),
                                                        Pose.from_parameters(
                                                            [hole_x, hole_y, experiment_config["real_hole_pose"][2], 1, 0, 0, 0]),
                                                        experiment_config["hole_parameters"]["radius"],
                                                        min_force, move_linear_vel, move_linear_acc, 0.001, 0.001,
                                                        sampling_interval=experiment_config["sampling_interval"])
            real_over_time.append(traj_physics.to_tensor().tolist())
            success_label_over_time.append(success_label)

            # Split motion into MLs & MLRCs
            move_linears_real, mlrcs_real = split_spike_search_trajectory_into_submotions(traj_physics)
            move_linears_sim, mlrcs_sim = split_spike_search_trajectory_into_submotions(Trajectory.from_tensor(Y_out_sim.squeeze()))

            # Add inputs, start_states and trajectories of MLRCs to training data
            mlrc_trajs_real = [traj.to_tensor() for traj in mlrcs_real]
            mlrc_trajs_sim = [traj.to_tensor() for traj in mlrcs_sim]
            mlrc_start_states = [mlrc_traj[0, 2:9] for mlrc_traj in mlrc_trajs_real]

            for j in range(len(mlrcs_real)):
                training_data.append((mlrc_inputs[j].detach().squeeze(), mlrc_start_states[j].detach().squeeze(),
                                      mlrc_trajs_sim[j].detach(), mlrc_trajs_real[j].detach()))

            if args.debug and inference_enabled:  # Plot this run
                plt.scatter([traj[-1, 2] for traj in mlrc_trajs_real], [traj[-1, 3] for traj in mlrc_trajs_real],
                            color=["green" if traj[-1, 1] > 0.5 else "red" for traj in mlrc_trajs_real])  # Trajectory end points XY
                plt.scatter(initial_grid_xy[0, :, 0], initial_grid_xy[0, :, 1], marker="^", color="black")
                plt.scatter([sample[0] for sample in random_samples], [sample[1] for sample in random_samples],
                            color="gray", alpha=0.5)  # Hole poses XY
                plt.title(f"Run {i}/{args.num_steps}")
                plt.show()

            # Finetune MLRC on current training data once I have gathered enough
            if len(training_data) == training_data.maxlen:
                indices = list(range(training_data.maxlen))
                random.shuffle(indices)
                training_set = [training_data[j] for j in indices[:128]]
                train_inputs, train_start_states, train_sim, train_real = zip(*training_set)
                train_inputs = torch.stack(train_inputs)
                train_start_states = torch.stack(train_start_states)
                train_sim = torch.stack([Trajectory.from_tensor(traj_tensor).to_tensor(pad_to=30) for traj_tensor in train_sim])
                train_real = torch.stack([Trajectory.from_tensor(traj_tensor).to_tensor(pad_to=30) for traj_tensor in train_real])
                valid_set = [training_data[j] for j in indices[128:]]
                valid_inputs, valid_start_states, valid_sim, valid_real = zip(*valid_set)
                valid_inputs = torch.stack(valid_inputs)
                valid_start_states = torch.stack(valid_start_states)
                valid_sim = torch.stack([Trajectory.from_tensor(traj_tensor).to_tensor(pad_to=30) for traj_tensor in valid_sim])
                valid_real = torch.stack([Trajectory.from_tensor(traj_tensor).to_tensor(pad_to=30) for traj_tensor in valid_real])

                if args.debug:  # Plot training data
                    plt.scatter(train_real[:, -1, 2], train_real[:, -1, 3], color=["green" if traj[-1, 1] > 0.5 else "red" for traj in train_real]) # Trajectory end points XY
                    plt.scatter([sample[0] for sample in random_samples], [sample[1] for sample in random_samples], color="gray", alpha=0.5)    # Hole poses XY
                    plt.title(f"Training data: {id(training_data[0])} -> {id(training_data[-1])}")
                    plt.show()

                learned_mlrc = ProgramPrimitive.load(args.trained_mlrc_dir)     # Don't keep finetuning the same model: Reload to finetune from scratch
                learned_mlrc.set_device(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                finetune_on_task((train_inputs, None, train_start_states, train_sim, train_real),
                                 (valid_inputs, None, valid_start_states, valid_sim, valid_real),
                                 learned_mlrc, adapt_steps=512, lr=5e-4, invert_success_label=True)
                inference_enabled = True
        ani = NonstationarySpikeSearchAnimation(traj_physics, hole_poses_over_time, spike_points_over_time)
        output_dir = os.path.join(args.output_dir, str(experiment_idx))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ani.save(os.path.join(output_dir, "animation.gif"))
        with open(os.path.join(output_dir, "hole_distributions_over_time.json"), "w") as hole_distributions_over_time_file:
            json.dump([d.to_dict() for d in hole_distributions_over_time], hole_distributions_over_time_file)
        np.save(os.path.join(output_dir, "actual_hole_over_time.npy"), actual_hole_over_time)
        np.save(os.path.join(output_dir, "real_over_time.npy"), real_over_time)
        np.save(os.path.join(output_dir, "sim_over_time.npy"), sim_over_time)
        np.save(os.path.join(output_dir, "spike_points_over_time.npy"), [[[sp.position.x.item(), sp.position.y.item()] for sp in l] for l in spike_points_over_time])
        np.save(os.path.join(output_dir, "success_label_over_time.npy"), success_label_over_time)


def split_spike_search_trajectory_into_submotions(spike_search_traj: Trajectory):
    move_linear_trajs = []
    mlrc_trajs = []
    last_z = -np.inf
    current_traj = []
    going_down = False
    traj_physics_tensor = spike_search_traj.to_tensor()
    # plt.plot(range(len(traj_physics_tensor)), traj_physics_tensor[:, 4])
    # plt.show()
    for i in range(len(traj_physics_tensor)):
        if traj_physics_tensor[i, 4] < last_z - 0.00001:  # Going down
            if not going_down and len(current_traj) > 0:  # Have a Move Linear queued up
                move_linear_trajs.append(torch.stack(current_traj))
                current_traj = []
            going_down = True
        else:  # Going up
            if going_down and len(current_traj) > 0:  # Have a MLRC queued up
                mlrc_trajs.append(torch.stack(current_traj))
                current_traj = []
            going_down = False
        current_traj.append(traj_physics_tensor[i])
        last_z = traj_physics_tensor[i, 4]
    if not going_down and len(current_traj) > 0:
        move_linear_trajs.append(torch.stack(current_traj))
    elif going_down and len(current_traj) > 0:
        mlrc_trajs.append(torch.stack(current_traj))

    move_linear_trajs = [Trajectory.from_tensor(t) for t in move_linear_trajs]
    mlrc_trajs = [Trajectory.from_tensor(t) for t in mlrc_trajs]
    for mlrc_traj in mlrc_trajs:
        mlrc_traj.success_label = True       # All spikes made contact with the surface --> successful
    if spike_search_traj.success_label:
        mlrc_trajs[-1].success_label = False     # The last contact motion found the hole --> unsuccessful
    return move_linear_trajs, mlrc_trajs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("process_type", choices=["brownian", "drift", "shift"])
    parser.add_argument("num_experiments", type=int)
    parser.add_argument("num_steps", type=int)
    parser.add_argument("trained_mlrc_dir", type=str)
    parser.add_argument("mlrc_model_config", type=str)
    parser.add_argument("experiment_config", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_optimization", action="store_true")
    main(parser.parse_args())
