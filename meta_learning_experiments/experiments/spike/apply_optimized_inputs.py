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
from argparse import ArgumentParser

import torch
from am_control_plugin_python.rps_interface.http_interface import RPSInterface
from am_control_plugin_python.rps_interface.websocket_interface import RPSWebSocket
from graphmodel.graphmodel.graph_node_move_linear_relative_contact import GraphNodeMoveLinearRelativeContact
from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from meta_learning_experiments.experiments.spike.spike_search_tools import get_ml_and_mlrc_node_ids, \
    execute_spiral_on_real_robot_and_collect_trajectory_from_lar
from natsort import natsorted
from spi.neural_templates.program_primitive import ProgramPrimitive
from spi.utils.distributions import GaussianMixture

from meta_learning_experiments.experiments.spike.spike_search import SpikeSearch
from spi.physics.apply_physics_spike_search import apply_physics
from spi.simulation.static_simulator import StaticSimulator
from tqdm import tqdm

DB_CREDENTIALS = {"host": "nb067", "database": "lar_local", "user": "root", "password": "root"}

NODE_ID_APPROACH = 77
NODE_ID_SPIKE_SEARCH_RELATIVE = 615
NODE_ID_OFFSET_TAG = 1254


def execute_in_simulation(model_config_mlrc, hole_distributions, optimal_parameters, experiment_config, output_dir):
    limits_x = torch.tensor(experiment_config["mlrc"]["limits_x"], dtype=torch.float32)
    limits_s = torch.tensor(experiment_config["mlrc"]["limits_s"], dtype=torch.float32)
    limits_Y = torch.tensor(experiment_config["mlrc"]["limits_Y"], dtype=torch.float32)
    mlrc = ProgramPrimitive("Move Linear Relative Contact", 11, limits_x, limits_s, limits_Y,
                            StaticSimulator(GraphNodeMoveLinearRelativeContact(),
                                            experiment_config["sampling_interval"],
                                            multiproc=False), model_config_mlrc)
    spike_search = SpikeSearch(16, mlrc, experiment_config)
    true_hole_pose = torch.tensor(experiment_config["real_hole_pose"], dtype=torch.float32)
    start_state = true_hole_pose + torch.tensor([0, 0, 0.03, 0, 0, 0, 0])
    for task_idx in range(len(hole_distributions)):
        # spike_search.initialize_inputs(start_state)

        task_output_dir = os.path.join(output_dir, str(task_idx))
        os.makedirs(task_output_dir)
        optimal_params_tensor = torch.tensor(optimal_parameters[task_idx], dtype=torch.float32)
        hole_distribution = hole_distributions[task_idx]
        Y_pred, s_out_pred = spike_search(optimal_params_tensor.unsqueeze(0), start_state.unsqueeze(0))
        traj_prior = Trajectory.from_tensor(Y_pred.squeeze())
        inputs_split = SpikeSearch.split_inputs(spike_search.num_spikes, optimal_params_tensor.unsqueeze(0))
        min_force, max_force, vel, acc = inputs_split[1][0, -4:]
        # plot_trajectory_3d([traj_prior, traj_physics], ["black", "red"], ["sim", "real"])
        results = []
        for i in tqdm(range(128)):
            hole_x, hole_y = hole_distribution.sample(1)[0]
            traj_physics, success_label = apply_physics(experiment_config["physics"], traj_prior,
                                                        Pose.from_parameters(
                                                            [hole_x, hole_y, true_hole_pose[2], 1, 0, 0, 0]),
                                                        experiment_config["hole_parameters"]["radius"],
                                                        min_force, vel, acc, 0.001, 0.001,
                                                        sampling_interval=experiment_config["sampling_interval"])
            # fig, ax = plt.subplots()
            # ax.scatter([hole_x], [hole_y], color="purple", marker="^")
            # plot_2d_trajectory_xy(ax, [traj_prior.to_tensor(), traj_physics.to_tensor()], ["black", "red"], ["sim", "real"])
            # hole_circle = plt.Circle((hole_x, hole_y), experiment_config["hole_parameters"]["radius"], color="gray", alpha=0.5)
            # ax.add_patch(hole_circle)
            # plt.show()
            # print(f"Success: {traj_physics.success_label}")
            results.append({"success": traj_physics.success_label,
                            "cycle_time": len(traj_physics),
                            "path_length": traj_physics.path_length()})
        with open(os.path.join(task_output_dir, "results.json"), "w") as results_file:
            json.dump(results, results_file)


def execute_on_real_robot(hole_distributions, optimal_parameters, experiment_config, output_dir):
    # Assume correct program already loaded in RPS
    rps = RPSInterface("192.168.180.81")
    ws = RPSWebSocket("192.168.180.81")
    ws.connect()
    ml_node_id, mlrc_node_id = get_ml_and_mlrc_node_ids(rps)

    true_hole_pose = Pose.from_parameters(experiment_config["real_hole_pose"])
    for task_idx in range(len(hole_distributions)):
        task_output_dir = os.path.join(output_dir, str(task_idx))
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)
        if "results.json" in os.listdir(task_output_dir):
            print(f"Already have results for task {task_idx}, skipping...")
            continue
        hole_distribution = hole_distributions[task_idx]
        intermediate_results = []
        if "intermediate_results.json" in os.listdir(task_output_dir):
            print(f"Already have intermediate results for task {task_idx}, continuing...")
            with open(os.path.join(task_output_dir, "intermediate_results.json")) as intermediate_results_file:
                intermediate_results = json.load(intermediate_results_file)
        optimal_param_tensor = torch.tensor(optimal_parameters[task_idx])
        for i in range(128 - len(intermediate_results)):
            hole_x, hole_y = hole_distribution.sample(1)[0]
            offset_x, offset_y = hole_x - true_hole_pose.position.x, hole_y - true_hole_pose.position.y

            trajectory, revision_hash = execute_spiral_on_real_robot_and_collect_trajectory_from_lar(
                optimal_param_tensor, rps, ws, offset_x, offset_y, NODE_ID_OFFSET_TAG, NODE_ID_APPROACH,
                NODE_ID_SPIKE_SEARCH_RELATIVE, ml_node_id, mlrc_node_id, DB_CREDENTIALS)

            intermediate_results.append({"success": trajectory.success_label,
                                         "cycle_time": len(trajectory),
                                         "path_length": trajectory.path_length(),
                                         "revision_hash": revision_hash})
            with open(os.path.join(task_output_dir, "intermediate_results.json"), "w") as intermediate_results_file:
                json.dump(intermediate_results, intermediate_results_file)
        with open(os.path.join(task_output_dir, "results.json"), "w") as results_file:
            json.dump([{"success": intermediate_res["success"],
                        "cycle_time": intermediate_res["cycle_time"],
                        "path_length": intermediate_res["path_length"]} for intermediate_res in intermediate_results],
                      results_file)
    ws.close()


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    hole_distributions = []
    for task_dir in natsorted(os.listdir(args.task_description_dir)):
        hole_distributions.append(
            GaussianMixture.load(os.path.join(args.task_description_dir, task_dir, "hole_distribution.json")))
    optimal_parameters = []
    for task_dir in natsorted(os.listdir(args.optimized_inputs_dir)):
        task_dirpath = os.path.join(args.optimized_inputs_dir, task_dir)
        with open(os.path.join(task_dirpath, list(filter(lambda fn: fn.endswith(".json"), os.listdir(task_dirpath)))[
            0])) as optimal_param_file:
            optimal_parameters.append(json.load(optimal_param_file))
    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)
    with open(args.model_config_mlrc) as model_config_file:
        model_config_mlrc = json.load(model_config_file)
    if args.type == "sim":  # Execute in simulation
        execute_in_simulation(model_config_mlrc, hole_distributions, optimal_parameters, experiment_config,
                              args.output_dir)
    else:  # Execute on real robot
        execute_on_real_robot(hole_distributions, optimal_parameters, experiment_config, args.output_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("type", type=str, choices=["sim", "real"])
    parser.add_argument("task_description_dir", type=str,
                        help="Path to dir containing one folder for each task, which contains a hole_distribution.json file")
    parser.add_argument("optimized_inputs_dir", type=str,
                        help="Path to dir containing one folder for each task, which contains at least one JSON file with optimized inputs")
    parser.add_argument("experiment_config", type=str)
    parser.add_argument("model_config_mlrc", type=str)
    parser.add_argument("output_dir", type=str,
                        help="Path to dir, in which one folder for each task will be created, which will contain a JSON file with the results")
    main(parser.parse_args())
