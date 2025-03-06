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
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from spi.visualization.spike_search import plot_spike_search_3d

from meta_learning_experiments.experiments.common_utils import MetaDataset


def main(args):
    print(f"Dataset contains {len(os.listdir(args.data_dir))} tasks")
    tasks = [os.path.join(args.data_dir, filename) for filename in os.listdir(args.data_dir)]
    dataset = MetaDataset(tasks)
    num_successful = 0
    samples_per_task = None
    for task_idx, task in tqdm(enumerate(dataset)):
        if samples_per_task is None:
            samples_per_task = len(task)
        successful_for_task = 0
        for i in range(len(task)):
            inputs, hole_center, start_state, sim, real = task[i]
            if real[-1, 1] > 0.5:
                successful_for_task += 1
        print(f"Proportion successful [{task_idx}]: {successful_for_task / len(task):.4f}")
        num_successful += successful_for_task

        train_batch, valid_batch, test_batch = task.sample_train_validate_test(128, 128, 128, torch.device("cpu"))
        test_inputs, test_hole_centers, test_start_states, test_sim, test_real = test_batch

        # Hole distribution
        # plt.scatter(test_hole_centers[:, 0], test_hole_centers[:, 1], color="gray", alpha=0.5)
        # plt.scatter(test_real[:, -1, 2], test_real[:, -1, 3], color=["green" if test_real[i, -1, 1] > 0.5 else "red" for i in range(len(test_real))])
        # plt.show()

        # 2D plots
        # for i in range(len(test_inputs)):
        #     print(f"Successful: {(test_real[i, -1, 1]) > 0.5}")
        #     fig, ax = plt.subplots(2, 2)
        #     ax[0,0].scatter([hc[0] for hc in test_hole_centers], [hc[1] for hc in test_hole_centers], color="gray", alpha=0.5)
        #     ax[0,0].scatter(test_hole_centers[i, 0], test_hole_centers[i, 1], color="black", marker="^")
        #     ax[0,0].plot(test_real[i, :, 2], test_real[i, :, 3], color="red", label="real")
        #     ax[0,0].plot(test_sim[i, :, 2], test_sim[i, :, 3], color="green", label="sim")
        #     ax[0,0].legend()
        #     ax[0, 1].plot(range(len(test_real[i])), test_real[i, :, 2], color="red", label="real (X)")
        #     ax[0, 1].plot(range(len(test_sim[i])), test_sim[i, :, 2], color="green", label="sim (X)")
        #     ax[0, 1].legend()
        #     ax[1, 1].plot(range(len(test_real[i])), test_real[i, :, 3], color="red", label="real (Y)")
        #     ax[1, 1].plot(range(len(test_sim[i])), test_sim[i, :, 3], color="green", label="sim (Y)")
        #     ax[1, 1].legend()
        #     ax[1, 0].plot(range(len(test_real[i])), test_real[i, :, 4], color="red", label="real (Z)")
        #     ax[1, 0].plot(range(len(test_sim[i])), test_sim[i, :, 4], color="green", label="sim (Z)")
        #     ax[1, 0].legend()
        #     plt.show()

        # 3D plot
        if args.problem_type == "spike":
            for i in range(len(test_inputs)):
                plot_spike_search_3d(test_hole_centers, test_hole_centers[i], test_real[i], test_sim[i])

    proportion_successful_total = num_successful / (len(dataset) * samples_per_task)
    print("----------")
    print(f"Proportion successful (all): {proportion_successful_total:.4f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("problem_type", type=str, choices=["spiral", "spike"])
    parser.add_argument("data_dir", type=str)
    main(parser.parse_args())
