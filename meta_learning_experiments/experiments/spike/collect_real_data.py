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
import subprocess
from argparse import ArgumentParser
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from am_control_plugin_python.data.template_data import UpdateTemplateInputsRequest, TemplatePorts, TemplatePort, \
    ProgramData
from am_control_plugin_python.rps_interface.http_interface import RPSInterface
from am_control_plugin_python.rps_interface.websocket_interface import RPSWebSocket
from graphmodel.graphmodel.graph_node_move_linear_relative_contact import GraphNodeMoveLinearRelativeContact
from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from utils.transformations import pose_euler_zyx_to_affine, affine_transform
from spi.simulation.static_simulator import StaticSimulator
from spi.utils.data_io import save_tensors_to_file
from spi.utils.distributions import GaussianMixture
from spi.utils.lar_utils import trajectories_for_revision_hash, run_ids_for_revision, template_instance_id_for_node_id
from spi.utils.rps_utils import assert_ok, convert_pose_quaternion, execute_synchronously, set_spike_search_inputs

from meta_learning_experiments.experiments.common_utils import set_offset_tag
from meta_learning_experiments.experiments.spike.spike_search_tools import random_spike_search_relative_inputs, \
    fixed_spike_search_relative_inputs

NODE_ID_MOVE_LINEAR = 77
NODE_ID_SPIKE_SEARCH_RELATIVE = 615
NODE_ID_OFFSET_TAG = 1254

DB_CREDENTIALS = {"host": "nb067", "database": "lar_local", "user": "root", "password": "root"}


def execute_with_rps(rps: RPSInterface, websocket: RPSWebSocket, approach_point_from, move_linear_params,
                     spike_search_params):
    assert_ok(rps.update_template_inputs(UpdateTemplateInputsRequest(nodeId=NODE_ID_MOVE_LINEAR, inputs=TemplatePorts(
        poses=[TemplatePort(nameInner="PointTo", value=convert_pose_quaternion(move_linear_params[:7].tolist()))]))))
    set_spike_search_inputs(rps, NODE_ID_SPIKE_SEARCH_RELATIVE, spike_search_params)
    assert_ok(rps.simulate())
    assert_ok(rps.compile())
    ok = False
    while not ok:
        ok = execute_synchronously(rps, websocket, timeout_seconds=45)
        if not ok:
            subprocess.call(["C:\\Program Files (x86)\\Git\\git-bash.exe", "-c", "telegram_notify from_python"])
            print("Execution failed. Retrying...")
    revision_hash = assert_ok(rps.get_revision_hash())["revisionHash"]
    parameterization = assert_ok(rps.get_template(NODE_ID_SPIKE_SEARCH_RELATIVE))
    return revision_hash, parameterization


def random_inputs(spike_point_from, experiment_config: dict):
    """
    :param spike_point_from: Point above the hole surface from which the first MLRC starts
    """
    spike_search_params = random_spike_search_relative_inputs(1, experiment_config).squeeze()
    approach_motion_point_from = spike_point_from + torch.tensor([0, 0, 0.001, 0, 0, 0, 0], dtype=spike_point_from.dtype)
    move_linear_params = torch.cat((spike_point_from, torch.ones(2, dtype=spike_point_from.dtype)))
    return approach_motion_point_from, move_linear_params, spike_search_params

def fixed_inputs(spike_point_from, experiment_config: dict):
    spike_search_params = fixed_spike_search_relative_inputs(1, experiment_config).squeeze()
    approach_motion_point_from = spike_point_from + torch.tensor([0, 0, 0.001, 0, 0, 0, 0], dtype=spike_point_from.dtype)
    move_linear_params = torch.cat((spike_point_from, torch.ones(2, dtype=spike_point_from.dtype)))
    return approach_motion_point_from, move_linear_params, spike_search_params

def execute_spike_searches_for_task(num_examples: int, experiment_config: dict, rps_ip: str, task_output_dir: str,
                                    fixed_spike_inputs=False, debug=False):
    """
    Creates a new hole distribution and executes num_examples for sampled hole poses. Inputs can be random or fixed.
    The hole offsets are saved via a tag in the LAR.
    The inputs are saved to output_filepath.
    """
    true_hole_pose = Pose.from_parameters(experiment_config["real_hole_pose"])
    if "hole_distribution.json" in os.listdir(task_output_dir):
        random_distribution = GaussianMixture.load(os.path.join(task_output_dir, "hole_distribution.json"))
    else:
        xmin, xmax = experiment_config["hole_parameters"]["pos_x"]
        ymin, ymax = experiment_config["hole_parameters"]["pos_y"]
        random_distribution = GaussianMixture.make_random(1, 6, xmin, xmax, ymin, ymax)
        random_distribution.save(os.path.join(task_output_dir, "hole_distribution.json"))
        if debug:
            random_distribution.plot(None, xmin, xmax, ymin, ymax)
            plt.show()
    rps = RPSInterface(rps_ip)
    websocket = RPSWebSocket(rps_ip)
    websocket.connect()
    # Check how many have already been collected, and collect the remaining number of examples
    i = len(os.listdir(task_output_dir)) / 2 - 1    # 2 files per iteration, and 1x hole_distribution.json
    while i < num_examples:
        hole_pose_x, hole_pose_y = random_distribution.sample(1)[0]   # Fake hole position
        offset_x, offset_y = hole_pose_x - true_hole_pose.position.x, hole_pose_y - true_hole_pose.position.y
        # Set offset tag
        # value
        set_offset_tag(rps, NODE_ID_OFFSET_TAG, offset_x, offset_y)
        # Because the ACTUAL hole pose is fixed, I have to translate point_start by -hole_pose_x and -hole_pose_y
        # Also Spike Search starts above the hole, so translate up by 0.5 mm
        spike_point_from = torch.tensor(true_hole_pose.parameters()) + torch.tensor([-offset_x, -offset_y, 0.0005, 0.0, 0.0, 0.0, 0.0])
        if not fixed_spike_inputs:
            approach_point_from, move_linear_params, spike_search_params = random_inputs(spike_point_from, experiment_config)
        else:
            approach_point_from, move_linear_params, spike_search_params = fixed_inputs(spike_point_from, experiment_config)

        try:
            revision_hash, spike_search_template_data = execute_with_rps(rps, websocket, approach_point_from, move_linear_params, spike_search_params)
        except RuntimeError:
            continue
        torch.save(spike_search_params, os.path.join(task_output_dir, f"inputs_{revision_hash}.pt"))
        torch.save([offset_x, offset_y], os.path.join(task_output_dir, f"offset_{revision_hash}.pt"))
        i += 1


def collect_data_for_task(task_input_dirpath: str, output_dir: str, experiment_config: dict, debug=False):
    """
    This is a bit different from the spirals because the LAR does not store Spike Searches as one template, but
    rather as separate Move Linear / MLRC combinations.
    For training, I only care about the MLRCs anyway.
    Strategy:
    1. Get the nodeID for the MLRC in the Spike Search Relative from the ARTM
    2. For each training example, get all executions for this nodeID and revisionHash from the LAR
    """
    task_idx = int(os.path.basename(task_input_dirpath)[0])
    task_output_filepath = os.path.join(output_dir, f"{task_idx}.h5")
    simulator = StaticSimulator(GraphNodeMoveLinearRelativeContact(), experiment_config["sampling_interval"],
                                multiproc=False)
    rps = RPSInterface()
    program_data = ProgramData.from_dict(assert_ok(rps.get_program_structure()))
    spike_search_template = list(filter(lambda t: t.type == "Spike Search Relative",
                                        program_data.topLevelTemplates))[0]
    spike_search_controller_ids = [dp.controllerNodeId for dp in spike_search_template.dynamicProperties]
    mlrc_node_id = spike_search_controller_ids[-1]

    hole_centers = []
    inputs = []
    start_states = []
    simulations = []
    trajectories = []

    for task_input_filename in list(filter(lambda fn: fn.endswith(".pt") and fn.startswith("inputs"),
                                           os.listdir(task_input_dirpath))):
        revision_hash = os.path.splitext(task_input_filename)[0].split("_")[1]
        spike_search_params = torch.load(os.path.join(task_input_dirpath, task_input_filename))
        try:
            template_instance_id = template_instance_id_for_node_id(DB_CREDENTIALS, revision_hash, mlrc_node_id)
        except IndexError:
            continue
        mlrcs = trajectories_for_revision_hash(DB_CREDENTIALS, revision_hash, [template_instance_id])[0]
        if not mlrcs[-1].success_label:
            trajectory = mlrcs[-1]
        else:
            trajectory = random.choice(mlrcs)
        run_id = run_ids_for_revision(DB_CREDENTIALS, revision_hash)[0]
        offset_x, offset_y = torch.load(os.path.join(task_input_dirpath, f"offset_{revision_hash}.pt"))
        fanuc_correction_offset_z = 0.33    # Fanuc LAR data always has this offset in Z, see https://track.int.artiminds.com/browse/INROP-4482
        # tag = tags_for_run(DB_CREDENTIALS, run_id)[0]
        # offset_x, offset_y = list(map(float, tag[0].split(" ")))

        # Apply offsets to trajectory
        homogeneous_hole_offset = pose_euler_zyx_to_affine(
            torch.tensor([[offset_x, offset_y, 0.0, 0.0, 0.0, 0.0]])).squeeze().float()
        homogeneous_traj_offset = pose_euler_zyx_to_affine(
            torch.tensor([[offset_x, offset_y, fanuc_correction_offset_z, 0.0, 0.0, 0.0]])).squeeze().float()
        orig_trajectory = deepcopy(trajectory).to_tensor()
        trajectory.transform(homogeneous_traj_offset)
        trajectory_transformed = trajectory.to_tensor()

        # Additional correction for Fanuc: Fanuc has "jump-back" behavior at the end of MLRC trajectories
        # --> Must end the MLRC trajectory at its lowest pose along Z
        min_z_idx = torch.argmin(trajectory_transformed[:, 4])
        trajectory_transformed = trajectory_transformed[:min_z_idx+1]
        trajectory_transformed = Trajectory.from_tensor(trajectory_transformed).to_tensor(pad_to=experiment_config["trajectory_length"])  # Re-pad
        true_hole_center = torch.tensor(experiment_config["real_hole_pose"], dtype=torch.float32)
        hole_center = affine_transform(homogeneous_hole_offset.unsqueeze(0),
                                       true_hole_center.unsqueeze(0)).squeeze()
        hole_centers.append(hole_center)
        mlrc_params = torch.tensor([0.0, 0.0, spike_search_params[-6], 1.0, 0.0, 0.0, 0.0,
                                    *spike_search_params[-4:]], dtype=torch.float32)
        inputs.append(mlrc_params)
        start_state = trajectory_transformed[0, 2:9]
        start_states.append(start_state)
        trajectories.append(trajectory_transformed)
        sim = simulator.simulate(mlrc_params, start_state, experiment_config["trajectory_length"], cache=False).squeeze()
        simulations.append(sim)

        if debug and trajectory.success_label:
            plt.scatter([hc[0] for hc in hole_centers], [hc[1] for hc in hole_centers], color="gray", alpha=0.5)
            plt.scatter(hole_center[0], hole_center[1], color="black", marker="^")
            plt.scatter(true_hole_center[0], true_hole_center[1], color="black", marker="s")
            plt.plot(trajectory_transformed[:, 2], trajectory_transformed[:, 3], color="red", label="real")
            plt.plot(orig_trajectory[:, 2], orig_trajectory[:, 3], color="red", linestyle="--", label="real")
            plt.plot(sim[:, 2], sim[:, 3], color="green", label="sim")
            plt.show()

    save_tensors_to_file(task_output_filepath, {
        "inputs": torch.stack(inputs),
        "hole_centers": torch.stack(hole_centers),
        "start_states": torch.stack(start_states),
        "sim": torch.stack(simulations),
        "real": torch.stack(trajectories)
    })

def main(args):
    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)
    if args.command == "execute_random_spike_searches":
        print(f"Executing random spike searches for {args.num_tasks} tasks ({args.samples_per_task} each)")
        for task_idx in range(args.num_tasks):
            task_output_dir = os.path.join(args.output_dir, str(task_idx))
            if not os.path.exists(task_output_dir):
                os.makedirs(task_output_dir)
            execute_spike_searches_for_task(args.samples_per_task, experiment_config, args.rps_ip, task_output_dir,
                                            fixed_spike_inputs=args.fixed_inputs, debug=args.debug)
    else:
        task_dirpaths = [os.path.join(args.input_files_dir, filename) for filename in os.listdir(args.input_files_dir)]
        print(f"Collecting data for {len(task_dirpaths)} tasks from LAR")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        for task_dirpath in task_dirpaths:
            collect_data_for_task(task_dirpath, args.output_dir, experiment_config, debug=args.debug)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("command", type=str, choices=["execute_random_spike_searches", "gather_data_from_lar"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--experiment_config", type=str, help="JSON file containing experiment parameters")
    parser.add_argument("--output_dir", type=str)

    # execute_random_spike_searches
    parser.add_argument("--num_tasks", type=int)
    parser.add_argument("--samples_per_task", type=int)
    parser.add_argument("--rps_ip", type=str, default="127.0.0.1")
    parser.add_argument("--fixed_inputs", action="store_true")

    # gather_data_from_lar
    parser.add_argument("--input_files_dir", type=str)
    main(parser.parse_args())
