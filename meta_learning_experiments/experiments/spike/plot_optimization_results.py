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
from typing import List

import matplotlib.pyplot as plt
import torch
from graphmodel.graphmodel.graph_node_move_linear_relative_contact import GraphNodeMoveLinearRelativeContact
from matplotlib import rcParams
from meta_learning_experiments.experiments.common_utils import MetaDataset, plot_hole_distribution_2d
from meta_learning_experiments.experiments.spike.spike_search import SpikeSearch
from natsort import natsorted
from spi.neural_templates.program_primitive import ProgramPrimitive
from spi.simulation.static_simulator import StaticSimulator

from spi.utils.distributions import GaussianMixture
from spi.utils import matplotlib_defaults


def plot_spike_search(task, hole_distribution: GaussianMixture, optimal_parameters_spi: List[float],
                      spi_label: str, experiment_config: dict, model_config: dict, output_filepath: str):
    fig, ax = plt.subplots()

    contour, mins_maxes = plot_hole_distribution_2d(ax, fig, hole_distribution)

    limits_x = torch.tensor(experiment_config["mlrc"]["limits_x"], dtype=torch.float32)
    limits_s = torch.tensor(experiment_config["mlrc"]["limits_s"], dtype=torch.float32)
    limits_Y = torch.tensor(experiment_config["mlrc"]["limits_Y"], dtype=torch.float32)
    mlrc_sim = StaticSimulator(GraphNodeMoveLinearRelativeContact(),
                               sampling_interval=experiment_config["sampling_interval"], multiproc=False)
    mlrc = ProgramPrimitive("Move Linear Relative Contact", 11, limits_x, limits_s, limits_Y, mlrc_sim, model_config)
    spike_search = SpikeSearch(16, mlrc, experiment_config)
    true_hole_pose = torch.tensor(experiment_config["real_hole_pose"], dtype=torch.float32)

    # Plot SPI results
    param_tensor = torch.tensor(optimal_parameters_spi).unsqueeze(0)
    inputs_split = SpikeSearch.split_inputs(16, param_tensor)
    spike_points_xy = torch.cat(inputs_split[::2], dim=0)[:, :2] * 1000
    ax.scatter(spike_points_xy[:, 0], spike_points_xy[:, 1], color="blue", marker="s", label=spi_label)

    start_state = inputs_split[0][0, :7] + torch.tensor([0, 0, 0.003, 0, 0, 0, 0])
    Y, s_out = spike_search(param_tensor, s_in=start_state.unsqueeze(0), denormalize_out=True)
    ax.plot(Y[0, :, 2] * 1000, Y[0, :, 3] * 1000, color="blue")

    # Plot initial grid
    naive_inputs = SpikeSearch.make_inputs(experiment_config, 16, random=False)
    naive_inputs_split = SpikeSearch.split_inputs(16, naive_inputs.unsqueeze(0))
    naive_spike_points_xy = torch.cat(naive_inputs_split[::2], dim=0)[:, :2] * 1000
    ax.scatter(naive_spike_points_xy[:, 0], naive_spike_points_xy[:, 1], color="orange", marker="^", label="Human",
               s=0.5 * rcParams['lines.markersize'] ** 2)

    # ax.set_xlim(*mins_maxes[:, 0])
    # ax.set_ylim(*mins_maxes[:, 1])
    ax.set_xlabel("Pos. (X) [mm]")
    ax.set_ylabel("Pos. (Y) [mm]")
    ax.legend(frameon=True, facecolor="white", framealpha=0.5, loc="lower right")
    # plt.show()
    plt.savefig(output_filepath)


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    meta_test_tasks = [os.path.join(args.test_data_dir, filename) for filename in
                       natsorted(os.listdir(args.test_data_dir))]
    meta_test_set = MetaDataset(meta_test_tasks)
    hole_distributions = []

    for task_dir in natsorted(os.listdir(args.hole_distribution_dir)):
        hole_distributions.append(
            GaussianMixture.load(os.path.join(args.hole_distribution_dir, task_dir, "hole_distribution.json")))

    optimal_parameters_spi = []
    for task_dir in natsorted(os.listdir(args.optimized_inputs_dir_spi)):
        task_dirpath = os.path.join(args.optimized_inputs_dir_spi, task_dir)
        with open(os.path.join(task_dirpath, list(filter(lambda fn: fn.endswith(".json"), os.listdir(task_dirpath)))[
            0])) as optimal_param_file:
            optimal_parameters_spi.append(json.load(optimal_param_file))
    spi_label = r"tSPI ($\mathcal{L}_{cycle}$)"
    if "cdist" in args.optimized_inputs_dir_spi:
        spi_label = r"tSPI ($\mathcal{L}_{cycle} + \mathcal{L}_{cdist}$)"
    elif "grid" in args.optimized_inputs_dir_spi:
        spi_label = r"tSPI ($\mathcal{L}_{cycle} + \mathcal{L}_{grid}$)"

    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)
    with open(args.model_config) as model_config_file:
        model_config = json.load(model_config_file)

    for task_idx in range(len(hole_distributions)):
        plot_spike_search(meta_test_set[task_idx], hole_distributions[task_idx], optimal_parameters_spi[task_idx],
                          spi_label, experiment_config, model_config,
                          os.path.join(args.output_dir, f"spike_search_2d_{task_idx}.pdf"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("hole_distribution_dir", type=str,
                        help="Path to dir containing one folder for each task, which contains a hole_distribution.json file")
    parser.add_argument("test_data_dir", type=str)
    parser.add_argument("optimized_inputs_dir_spi", type=str,
                        help="Path to dir containing one folder for each task, which contains at least one JSON file with optimized inputs")
    parser.add_argument("experiment_config", type=str)
    parser.add_argument("model_config", type=str)
    parser.add_argument("output_dir", type=str,
                        help="Path to dir, in which one folder for each task will be created, which will contain a JSON file with the results")
    main(parser.parse_args())
