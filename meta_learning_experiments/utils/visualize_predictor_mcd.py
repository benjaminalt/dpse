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

import os
from argparse import ArgumentParser

import torch
from spi.common.trajectory import Trajectory
from natsort import natsorted
from spi.utils.distributions import GaussianMixture

from meta_learning_experiments.experiments.common_utils import MetaDataset, finetune_on_task
from spi.neural_templates.program_primitive import ProgramPrimitive
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def plot_spiral_trajectories_mcd(program: ProgramPrimitive, test_batch, axes, color="red"):
    inputs, hole_centers, start_states, sim, real = test_batch
    program.train()
    with torch.no_grad():
        predictions = []
        for _ in tqdm(range(100)):
            Y, s_out = program(inputs, start_states, sim)
            predictions.append(Y)
    predictions = torch.stack(predictions)
    # Y is now a tensor of shape (100, batch_size, seq_len, y)
    for i in range(len(predictions)):
        traj_tensor_repadded = Trajectory.from_tensor(predictions[i, 0].detach().cpu()).to_tensor(pad_to=300)
        axes[0].plot(range(len(traj_tensor_repadded)), traj_tensor_repadded[:, 2], alpha=0.05, color=color, linewidth=2.0)
        axes[1].plot(range(len(traj_tensor_repadded)), traj_tensor_repadded[:, 4], alpha=0.05, color=color, linewidth=2.0)


def plot_spiral_ends_and_hole_centers(program: ProgramPrimitive, test_batch, hole_centers, fig,
                                      minx, maxx, miny, maxy, mcd=False):
    inputs, _, start_states, sim, real = test_batch
    if not mcd:
        with torch.no_grad():
            Y, _ = program(inputs, start_states, sim)
            # Y should now have shape (128, seq_len, y)
    else: # Monte Carlo dropout
        program.train()
        Y = []
        with torch.no_grad():
            for _ in tqdm(range(100)):
                y, _ = program(inputs, start_states, sim)
                Y.append(y)
        Y = torch.cat(Y, dim=0)
        # Y should now have shape (128*100, seq_len, y)

    def scatter_hist(x, y, ax, ax_histx, ax_histy, color, alpha):
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        if len(x) <= 1000:  # Gets too messy for > 1000 points
            ax.scatter(x, y, color=color, alpha=alpha)
        histx = ax_histx.hist(x, bins=20, color=color, alpha=alpha, density=True)
        histy = ax_histy.hist(y, bins=20, orientation='horizontal', color=color, alpha=alpha, density=True)

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    ax = fig.add_axes(rect_scatter)
    ax.set_xlim(left=minx, right=maxx)
    ax.set_ylim(bottom=miny, top=maxy)
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    successful = []
    failed = []
    for i in range(len(Y)):
        if Y[i, -1, 1] > 0.5:
            successful.append(Y[i].cpu())
        else:
            failed.append(Y[i].cpu())
    successful = torch.stack(successful).numpy()
    failed = torch.stack(failed).numpy()

    # Plot 30 random spirals
    perm = torch.randperm(Y.size(0))
    # ax.scatter(start_states[idx, 0].cpu().numpy(), start_states[idx, 1].cpu().numpy(), marker="x", color="black")
    for i in perm[:30]:
        ax.plot(Y[i, :, 2].cpu().numpy(), Y[i, :, 3].cpu().numpy(), linewidth=1, color="black", alpha=0.1)

    # Plot scatter and histogram for all end points and hole centers
    scatter_hist(hole_centers[:, 0].numpy(), hole_centers[:, 1].numpy(), ax, ax_histx, ax_histy,  color="gray", alpha=0.5)
    scatter_hist(successful[:, -1, 2], successful[:, -1, 3], ax, ax_histx, ax_histy, color="green", alpha=0.5)
    scatter_hist(failed[:, -1, 2], failed[:, -1, 3], ax, ax_histx, ax_histy, color="red", alpha=0.1)


def main(args):
    test_tasks = [os.path.join(args.test_data_dir, filename) for filename in natsorted(os.listdir(args.test_data_dir))]
    all_hole_centers = []
    for task_idx in tqdm(natsorted(os.listdir(args.train_inputs_dir))):
        hole_distribution = GaussianMixture.load(os.path.join(args.train_inputs_dir, task_idx, "hole_distribution.json"))
        all_hole_centers.extend(hole_distribution.sample(128))
    all_hole_centers = torch.tensor(all_hole_centers)
    xmin, ymin = torch.min(all_hole_centers, dim=0)[0]
    xmax, ymax = torch.max(all_hole_centers, dim=0)[0]
    # Plot distribution over trajectories before & after finetuning
    meta_test_set = MetaDataset(test_tasks)
    for task_idx, task in enumerate(meta_test_set):
        pretrained_spiral_search = ProgramPrimitive.load(args.pretrained_model_dir)
        train_batch, valid_batch, test_batch = task.sample_train_validate_test(128, 128, 128, device)
        hole_distribution = GaussianMixture.load(os.path.join(args.test_inputs_dir, str(task_idx), "hole_distribution.json"))
        test_hole_centers = torch.tensor(hole_distribution.sample(128))
        fig_pretrained = plt.figure(1)
        plot_spiral_ends_and_hole_centers(pretrained_spiral_search, test_batch, all_hole_centers, fig_pretrained, xmin, xmax,
                                          ymin, ymax)
        # fig, ax = plt.subplots(2, 2)
        # plot_mcd(pretrained_spiral_search, test_batch, ax[0])
        trained_nt, loss_history = finetune_on_task(train_batch, valid_batch, pretrained_spiral_search,
                                                    adapt_steps=128, lr=5e-5)
        fig_finetuned = plt.figure(2)
        plot_spiral_ends_and_hole_centers(trained_nt, test_batch, test_hole_centers, fig_finetuned, xmin, xmax,
                                          ymin, ymax, mcd=True)
        # plot_mcd(trained_nt, test_batch, ax[1], "blue")
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("pretrained_model_dir", type=str)
    parser.add_argument("test_data_dir", type=str)
    parser.add_argument("train_inputs_dir", type=str)
    parser.add_argument("test_inputs_dir", type=str)
    parser.add_argument("experiment_config", type=str)
    main(parser.parse_args())
