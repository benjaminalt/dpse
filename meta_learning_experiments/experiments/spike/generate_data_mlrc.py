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

import json
import os
import random
from copy import deepcopy

import torch
from graphmodel.graphmodel.graph_node_move_linear_relative_contact import GraphNodeMoveLinearRelativeContact
from spi.common.orientation import Orientation
from spi.common.pose import Pose
from spi.common.position import Position
from spi.common.trajectory import Trajectory
from spi.physics.apply_physics_move_linear_relative_contact import apply_physics_with_hole
from pyquaternion import Quaternion
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from spi.simulation.static_simulator import StaticSimulator
from spi.utils.data_io import save_tensors_to_file
from spi.utils.distributions import GaussianMixture

from meta_learning_experiments.experiments.common_utils import torch_uniform_in_range


def generate_data_for_task(num_examples: int, output_dir: str, hole_distributions_output_dir: str, task_idx: int,
                           experiment_config: dict, debug=False):
    """
    Generates a training data file for a single task, containing a given number of training examples
    """
    xmin, xmax = experiment_config["hole_parameters"]["pos_x"]
    ymin, ymax = experiment_config["hole_parameters"]["pos_y"]
    random_distribution = GaussianMixture.make_random(1, 6, xmin, xmax, ymin, ymax)
    if debug:
        random_distribution.plot(None, xmin, xmax, ymin, ymax)
        plt.show()
    real_hole_pose = Pose.from_parameters(experiment_config["real_hole_pose"])

    num_failed = 0
    inputs = []
    hole_centers = []
    start_states = []
    sim = []
    real = []
    while len(inputs) < num_examples:
        hole = random_distribution.sample(1)[0]
        hole_center = Pose(Position(*hole, real_hole_pose.position.z), Orientation(Quaternion()))
        mlrc_inputs = torch_uniform_in_range(experiment_config["mlrc"]["input_range"])
        start_state = deepcopy(hole_center) # Start close to hole center to increase % of failed MLRCs
        start_state.position.x += random.uniform(-0.00075, 0.00075)
        start_state.position.y += random.uniform(-0.00075, 0.00075)
        start_state.position.z += 0.0005     # Start 0.5 mm above hole plane
        start_state = torch.tensor(start_state.parameters(), dtype=torch.float32)
        simulator = StaticSimulator(GraphNodeMoveLinearRelativeContact(),
                                    sampling_interval=experiment_config["sampling_interval"], multiproc=False)
        simulated_trajectory_tensor = simulator.simulate(mlrc_inputs, start_state,
                                                         max_trajectory_len=experiment_config["trajectory_length"],
                                                         cache=False)[0]
        simulated_trajectory = Trajectory.from_tensor(simulated_trajectory_tensor)
        min_force, max_force = mlrc_inputs[7:9]
        traj_physics, success_label = apply_physics_with_hole(experiment_config["physics"], simulated_trajectory,
                                                              hole_center, experiment_config["hole_parameters"]["radius"],
                                                              min_force, max_force,
                                                              sampling_interval=experiment_config["sampling_interval"])
        if not success_label:
            num_failed += 1
        traj_physics_tensor = traj_physics.to_tensor(pad_to=experiment_config["trajectory_length"])
        traj_physics_tensor_unpadded = traj_physics.to_tensor()
        simulated_trajectory_tensor_unpadded = simulated_trajectory.to_tensor()
        if debug:
            fig, ax = plt.subplots(nrows=1, ncols=3)
            c = "green" if traj_physics_tensor[-1, 1] > 0.5 else "red"
            ax[0].plot(range(len(traj_physics_tensor_unpadded)), traj_physics_tensor_unpadded[:, 2], color=c, label="real")
            ax[0].plot(range(len(simulated_trajectory_tensor_unpadded)), simulated_trajectory_tensor_unpadded[:, 2], color="black", label="sim")
            ax[1].plot(range(len(traj_physics_tensor_unpadded)), traj_physics_tensor_unpadded[:, 3], color=c, label="real")
            ax[1].plot(range(len(simulated_trajectory_tensor_unpadded)), simulated_trajectory_tensor_unpadded[:, 3], color="black", label="sim")
            ax[2].plot(range(len(traj_physics_tensor_unpadded)), traj_physics_tensor_unpadded[:, 4], color=c, label="real")
            ax[2].plot(range(len(simulated_trajectory_tensor_unpadded)), simulated_trajectory_tensor_unpadded[:, 4], color="black", label="sim")
            ax[2].axhline(hole_center.position.z)
            plt.show()
        hole_centers.append(torch.tensor(hole_center.parameters(), dtype=torch.float32))
        inputs.append(mlrc_inputs)
        start_states.append(start_state)
        sim.append(simulated_trajectory_tensor)
        real.append(traj_physics_tensor)

    print(f"Num failed: {num_failed}")
    output_filepath = os.path.join(output_dir, f"{task_idx}.h5")
    save_tensors_to_file(output_filepath, {
        "inputs": torch.stack(inputs),
        "hole_centers": torch.stack(hole_centers),
        "start_states": torch.stack(start_states),
        "sim": torch.stack(sim),
        "real": torch.stack(real)
    })
    task_hole_distribution_output_dir = os.path.join(hole_distributions_output_dir, str(task_idx))
    os.makedirs(task_hole_distribution_output_dir)
    random_distribution.save(os.path.join(task_hole_distribution_output_dir, "hole_distribution.json"))


def generate_meta_train_data(num_tasks: int, num_examples_per_task: int, output_dir: str, hole_distributions_output_dir: str,
                             experiment_config: dict, debug=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    already_there = [int(os.path.splitext(thing)[0]) for thing in os.listdir(output_dir)]
    task_indices = [i for i in range(num_tasks) if i not in already_there]
    # task_indices = range(len(os.listdir(output_dir)), num_tasks)
    print(f"Generating {len(task_indices)} data...")
    if not debug:
        pool = Pool(cpu_count() - 1)
        pool.starmap(generate_data_for_task, [(num_examples_per_task, output_dir, hole_distributions_output_dir,
                                               task_idx, experiment_config, debug) for task_idx in task_indices])
    else:
        for task_idx in task_indices:
            generate_data_for_task(num_examples_per_task, output_dir, hole_distributions_output_dir,
                                   task_idx, experiment_config, debug)


def main(args):
    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)
    generate_meta_train_data(args.num_tasks, args.num_examples_per_task, args.output_dir,
                             args.hole_distributions_output_dir, experiment_config, args.debug)
