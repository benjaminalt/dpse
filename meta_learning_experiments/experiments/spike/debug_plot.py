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
from copy import deepcopy
from typing import Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt

from am_control_plugin_python.data.template_data import ProgramData
from am_control_plugin_python.rps_interface.http_interface import RPSInterface
from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from meta_learning_experiments.experiments.spike.spike_search import SpikeSearch
from natsort import natsorted
from utils.transformations import pose_euler_zyx_to_affine
from spi.utils.distributions import GaussianMixture
from spi.utils.rps_utils import assert_ok
from spi.utils.lar_utils import last_n_run_ids, revision_hashes_for_run_ids, trajectories_for_revision_hash, \
    template_instance_id_for_node_id, run_ids_for_revision, tags_for_run

DB_CREDENTIALS = {"host": "nb067", "database": "lar_local", "user": "root", "password": "root"}


def _get_ml_and_mlrc_node_ids(rps) -> Tuple[int, int]:
    program_data = ProgramData.from_dict(assert_ok(rps.get_program_structure()))
    spike_search_template = list(filter(lambda t: t.type == "Spike Search Relative",
                                        program_data.topLevelTemplates))[0]
    spike_search_controller_ids = [dp.controllerNodeId for dp in spike_search_template.dynamicProperties]
    ml_node_id = spike_search_controller_ids[0]
    mlrc_node_id = spike_search_controller_ids[1]
    return ml_node_id, mlrc_node_id


def _combine_partial_trajectories_to_spike_search_trajectory(ml_trajectories: List[Trajectory],
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


def main(args):
    rps = RPSInterface()
    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)
    true_hole_pose = Pose.from_parameters(experiment_config["real_hole_pose"])
    for task_idx in map(int, natsorted(os.listdir(args.metrics_dir))):
        task_dirpath = os.path.join(args.optimized_param_dir, str(task_idx))
        with open(os.path.join(task_dirpath, list(filter(lambda fn: fn.endswith(".json"), os.listdir(task_dirpath)))[
            0])) as optimal_param_file:
            optimal_parameters = json.load(optimal_param_file)
        hole_distribution = GaussianMixture.load(os.path.join(args.hole_distributions_dir, str(task_idx), "hole_distribution.json"))
        with open(os.path.join(args.metrics_dir, str(task_idx), "intermediate_results.json")) as intermediate_results_file:
            intermediate_results = json.load(intermediate_results_file)
        for intermediate_result in intermediate_results:
            revision_hash = intermediate_result["revision_hash"]
            ml_node_id, mlrc_node_id = _get_ml_and_mlrc_node_ids(rps)
            ml_template_instance_id = template_instance_id_for_node_id(DB_CREDENTIALS, revision_hash, ml_node_id)
            mlrc_template_instance_id = template_instance_id_for_node_id(DB_CREDENTIALS, revision_hash, mlrc_node_id)
            mls = trajectories_for_revision_hash(DB_CREDENTIALS, revision_hash, [ml_template_instance_id])[0]
            mlrcs = trajectories_for_revision_hash(DB_CREDENTIALS, revision_hash, [mlrc_template_instance_id])[0]
            # Combine partial trajectories to full trajectory
            trajectory = _combine_partial_trajectories_to_spike_search_trajectory(mls, mlrcs)

            run_id = run_ids_for_revision(DB_CREDENTIALS, revision_hash)[0]
            tags = tags_for_run(DB_CREDENTIALS, run_id)
            offset_tag = next(filter(lambda tag: tag[0] == "offset", tags))
            offset_x, offset_y = list(map(float, offset_tag[1].split(" ")))
            fanuc_correction_offset_z = 0.33  # Fanuc LAR data always has this offset in Z, see https://track.int.artiminds.com/browse/INROP-4482

            # Apply offsets to trajectory
            homogeneous_traj_offset = pose_euler_zyx_to_affine(
                torch.tensor([[offset_x, offset_y, fanuc_correction_offset_z, 0.0, 0.0, 0.0]])).squeeze().float()
            trajectory.transform(homogeneous_traj_offset)
            trajectory_transformed = trajectory.to_tensor()

            # # Additional correction for Fanuc: Fanuc has "jump-back" behavior at the end of MLRC trajectories
            # # --> Must end the MLRC trajectory at its lowest pose along Z
            # min_z_idx = torch.argmin(trajectory_transformed[:, 4])
            # trajectory_transformed = trajectory_transformed[:min_z_idx + 1]

            hole_pose = torch.tensor(true_hole_pose.parameters())
            hole_pose[:2] += torch.tensor([offset_x, offset_y])    # Apply hole offset

            random_holes = np.array(hole_distribution.sample(128))

            fig, ax = plt.subplots(3, 1)
            ax[0].scatter(random_holes[:, 0], random_holes[:, 1], color="gray", alpha=0.5)
            ax[0].scatter(hole_pose[0], hole_pose[1], color="purple", marker="^")
            circle = plt.Circle((hole_pose[0], hole_pose[1]), 0.0005, color='purple', alpha=0.5)
            ax[0].add_patch(circle)
            ax[0].plot(trajectory_transformed[:, 2], trajectory_transformed[:, 3],
                     color="green" if trajectory.success_label else "red")
            inputs_split = SpikeSearch.split_inputs(16, torch.tensor(optimal_parameters).unsqueeze(0))
            ml_inputs = inputs_split[::2]
            ax[0].scatter([inp[0,0] for inp in ml_inputs], [inp[0,1] for inp in ml_inputs], color="black")
            ax[1].plot(range(len(trajectory_transformed[:, 2])), trajectory_transformed[:, 2])
            ax[2].plot(range(len(trajectory_transformed[:, 3])), trajectory_transformed[:, 3])
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("experiment_config", type=str)
    parser.add_argument("optimized_param_dir", type=str)
    parser.add_argument("metrics_dir", type=str)
    parser.add_argument("hole_distributions_dir", type=str)
    main(parser.parse_args())
