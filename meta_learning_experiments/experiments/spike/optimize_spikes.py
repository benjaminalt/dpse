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
import multiprocessing
import os
import shutil
import tempfile
from argparse import ArgumentParser
from copy import deepcopy

import torch
import numpy as np
import matplotlib.pyplot as plt
from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from natsort import natsorted
from spi.neural_programs.optimize_inputs_spi import optimize
from spi.neural_templates.program_primitive import ProgramPrimitive
from spi.visualization.poses import animate_2d_xy
from spi.visualization.spike_search import SpikeSearchAnimation

from meta_learning_experiments.experiments.common_utils import MetaDataset, finetune_on_task

from meta_learning_experiments.experiments.spike.spike_search import SpikeSearch
from meta_learning_experiments.experiments.spike.spike_search_tools import augment_dataset, SpikeSearchLoss, \
    spike_search_grid_xy_from_input_tensor

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)

    if os.path.exists(args.results_dir):
        shutil.rmtree(args.results_dir)
    os.makedirs(args.results_dir)

    meta_test_tasks = [os.path.join(args.test_data_dir, filename) for filename in natsorted(os.listdir(args.test_data_dir))]
    meta_test_set = MetaDataset(meta_test_tasks)
    meta_test_set = augment_dataset(MetaDataset(meta_test_tasks), experiment_config, 256)
    spi_iterations = 250
    spi_lr = 5e-5
    if args.show:
        for task_idx, task in enumerate(meta_test_set):
            optimize_for_task(args, experiment_config, spi_iterations, spi_lr, task, task_idx)
    else:
        pool = multiprocessing.Pool(2)
        pool.starmap(optimize_for_task, [(args, experiment_config, spi_iterations, spi_lr, task, task_idx) for task_idx, task in enumerate(meta_test_set)])


def optimize_for_task(args, experiment_config, spi_iterations, spi_lr, task, task_idx, adapt_steps, num_tests):
    pretrained_mlrc = ProgramPrimitive.load(args.trained_mlrc_dir)
    train_batch, valid_batch, test_batch = task.sample_train_validate_test(128, 128, 128, DEVICE)
    if args.model_type == "finetune":
        trained_nt, loss_history = finetune_on_task(train_batch, valid_batch, pretrained_mlrc, adapt_steps=adapt_steps, lr=5e-4,
                                                    invert_success_label=True)
    else:  # MAML
        # meta_learner = l2l.algorithms.MAML(spike_search_relative_nt,
        #                                    lr=spike_search_relative_nt.model_config["learning_rate"],
        #                                    first_order=True, allow_unused=True).to(device)
        # evaluation_error = fast_adapt_maml(train_batch, valid_batch, meta_learner, trajectory_loss,
        #                                    spike_search_relative_nt.model_config["adapt_steps"], debug=False)
        # print(f"Fast adaptation validation error: {evaluation_error.item():.4f}")
        # trained_nt = meta_learner.module
        raise NotImplementedError()
    temp_dir = tempfile.TemporaryDirectory()
    path = trained_nt.save(temp_dir.name, loss_history)
    # if args.show:
    #     plot_loss_history(path)
    test_inputs, test_hole_centers, test_mlrc_start_states, test_sim, test_real = test_batch
    task_results_dir = os.path.join(args.results_dir, str(task_idx))
    os.makedirs(task_results_dir)

    task_results = []

    for i in range(num_tests):
        finetuned_mlrc_nt = ProgramPrimitive.load(path, device=DEVICE)
        spike_search = SpikeSearch(16, finetuned_mlrc_nt, experiment_config)
        start_state = test_mlrc_start_states[i, :7].unsqueeze(0).to(DEVICE)
        input_tensor = SpikeSearch.make_inputs(experiment_config, spike_search.num_spikes, random=False).unsqueeze(0).to(DEVICE)

        initial_grid_xy = spike_search_grid_xy_from_input_tensor(input_tensor)
        loss_fn = SpikeSearchLoss(initial_grid_xy)

        xs, Ys, losses = [], [], []

        def callback(x, Y, loss):
            xs.append(x)
            Ys.append(Y)
            losses.append(loss)

        optimize(spike_search, spi_iterations, loss_fn, input_tensor, start_state, spi_lr, patience=50,
                 callback=callback)

        task_results.append({"x": xs,
                             "Y": Ys,
                             "loss": losses,
                             "hole_centers": test_hole_centers})

        # min_loss_idx = np.argmin(losses)
        # best_Y = Ys[min_loss_idx][0]
        # best_x = xs[min_loss_idx][0]
        # final_grid_xy = spike_search_grid_xy_from_input_tensor(best_x.unsqueeze(0))

        # # Before and after (trajectory Z)
        # if args.show:
        #     plt.title("Spike search trajectories before and after optimization")
        #     plt.plot(range(len(Ys[0][0])), Ys[0][0][:, 4], color="black", label="Before optimization")
        #     plt.plot(range(len(Ys[min_loss_idx][0])), Ys[min_loss_idx][0][:, 4], color="red", label="After optimization")
        #     plt.legend()
        #     plt.show()

        # fig, ax = plt.subplots(2, figsize=(5, 12))
        # ax[0].plot(range(len(losses)), losses, label="SPI loss")
        # ax[0].axvline(x=min_loss_idx, color="black", linestyle="--")
        # ax[0].set_xlabel("SGD step")
        # ax[0].legend()

        # ax[1].scatter(initial_grid_xy[0,:,0], initial_grid_xy[0,:,1], color="black", s=[2 for _ in range(final_grid_xy.shape[1])],
        #               label="Before optimization")
        # ax[1].scatter(test_hole_centers[:, 0].cpu(), test_hole_centers[:, 1].cpu(), color="gray", alpha=0.5,
        #               label="Hole centers")
        # ax[1].plot(best_Y[:, 2], best_Y[:, 3], color="red")
        # ax[1].scatter(final_grid_xy[0,:,0], final_grid_xy[0,:,1], color="red", s=[2 for _ in range(final_grid_xy.shape[1])],
        #               label="After optimization")
        # ax[1].legend()
        # if args.show:
        #     plt.show()
        # plt.savefig(os.path.join(task_results_dir, f"{i}.png"))
        # best_inputs = xs[min_loss_idx][0].tolist()
        # with open(os.path.join(task_results_dir, f"{i}.json"), "w") as optimal_inputs_file:
        #     json.dump(best_inputs, optimal_inputs_file)

        # fixed_poses = [Pose.from_parameters(hole_center.cpu()) for hole_center in test_hole_centers]
        # split_param_history = SpikeSearch.split_inputs(16, torch.stack(xs)[:, 0, :].cpu())
        # pose_tensors_over_time = torch.stack(split_param_history[::2])
        # pose_tensors_over_time = pose_tensors_over_time.permute(1, 0, 2)
        # moving_poses = [[Pose.from_parameters(pose_tensor) for pose_tensor in pose_tensors] for pose_tensors in
        #                 pose_tensors_over_time]
        # if args.monte_carlo_dropout:
        #     points = []
        #     for j in range(len(neural_program.neural_templates)):
        #         neural_program.neural_templates[j].train()
        #     for j in tqdm(range(len(optimized_param_history[0]))):
        #         points.append([])
        #         for k in range(100):
        #             with torch.no_grad():
        #                 combined_trajectory, internal_simulations = neural_program.predict([params[j] for params in optimized_param_history],
        #                                                                                    start_state,
        #                                                                                    environment_inputs_world=[test_hole_centers[i], test_hole_centers[i]])
        #             points[-1].append(combined_trajectory[-1, 2:4])

        # animate_2d_xy(moving_poses, fixed_poses, output_file=os.path.join(task_results_dir, f"{i}.mp4"),
        #               show=args.show)

        # spike_search_sim = deepcopy(spike_search)
        # for c in spike_search_sim.components:
        #     c.model = None
        # best_Y_sim, s_out = spike_search_sim(xs[min_loss_idx], start_state)
        # best_traj = Trajectory.from_tensor(best_Y_sim.squeeze())
        # min_z = min([pose.position.z for pose in best_traj.poses]) - 0.001
        # max_z = max([pose.position.z for pose in best_traj.poses]) + 0.001
        # a = SpikeSearchAnimation(best_traj, fixed_poses, moving_poses, (min_z, max_z))
        # if args.show:
        #     a.show()
        # a.save(os.path.join(task_results_dir, f"{task_idx}.gif"))
    return task_results


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    parser = ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["finetune", "maml"])
    parser.add_argument("trained_mlrc_dir", type=str)
    parser.add_argument("test_data_dir", type=str)
    parser.add_argument("experiment_config", type=str)
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--monte_carlo_dropout", action="store_true")
    main(parser.parse_args())
