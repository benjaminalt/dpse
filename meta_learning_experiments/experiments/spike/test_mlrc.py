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
from argparse import ArgumentParser
import learn2learn as l2l
from natsort import natsorted
from spi.neural_templates.neural_template_utils import normalize_traj
from spi.neural_templates.program_primitive import ProgramPrimitive
from utils.transformations import relative_to_absolute
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
import numpy as np

from spi.visualization.spike_search import plot_spike_search_2d, plot_spike_search_1d_xy

import meta_learning_experiments.experiments.common_utils
from meta_learning_experiments.experiments.common_utils import MetaDataset, trajectory_loss, fast_adapt_maml, \
    finetune_on_task, plot_success_prob_for_start_states
from meta_learning_experiments.experiments.spike.spike_search_tools import augment_dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def finetune_and_test(model_dir, data_dir: str, experiment_config: dict, samples_per_task=128, adapt_steps=2048,
                      learning_rate=5e-5, show=False, mc_dropout=False):
    test_tasks = [os.path.join(data_dir, filename) for filename in natsorted(os.listdir(data_dir))]
    # Data augmentation for test data generated in simulation: Sample failed MLRCs, because training data is sampled
    # only around the holes
    test_dataset = augment_dataset(MetaDataset(test_tasks), experiment_config, 2*samples_per_task)
    # test_dataset = MetaDataset(test_tasks)

    test_losses = []
    test_accs = []
    for task in test_dataset:
        nt = ProgramPrimitive.load(model_dir)
        train_batch, valid_batch, test_batch = task.sample_train_validate_test(samples_per_task, samples_per_task, samples_per_task, device)
        test_inputs, test_hole_centers, test_start_states, test_sim, test_real = test_batch
        if show:
            c = test_real[:, -1, 1].cpu()
            plt.scatter(test_start_states[:, 0].cpu(), test_start_states[:, 1].cpu(), c=c, vmin=0.0, vmax=1.0, cmap="RdYlGn")
            plt.scatter(test_hole_centers[:, 0].cpu(), test_hole_centers[:, 1].cpu(), color="gray", alpha=0.5)
            plt.title("Finetuning dataset")
            plt.show()
        trained_nt, loss_history = finetune_on_task(train_batch, valid_batch, nt, adapt_steps, learning_rate, plot_grads=False,
                                                    invert_success_label=True)
        if not mc_dropout:
            trained_nt.eval()
            with torch.no_grad():
                Y_pred, s_out_pred = nt(test_inputs, test_start_states, test_sim)
        else:
            #     trained_nt.train()
            #     with torch.no_grad():
            #         pred_real_test_world_samples = []
            #         for i in tqdm(range(100)):
            #             pred_real_test_norm, pred_sim_test_world = nt(test_inputs, test_hole_centers, test_start_states,
            #                                                           test_sim)
            #             pred_real_test_world = denormalize_outputs(pred_real_test_norm, nt.output_limits)
            #             pred_real_test_world_samples.append(pred_real_test_world)
            #     pred_real_test_world_samples = torch.stack(pred_real_test_world_samples)    #  This is now a batch of batches, shape (500, samples_per_task, 300, y)
            # # Invert success labels
            raise NotImplementedError()
        test_real[:, :, 1] = (~test_real[:, :, 1].bool()).float()
        if show:
            for i in range(1):#len(test_inputs)):
                fig, ax = plt.subplots(2, 2, figsize=(12, 10))

                # Training plot
                ax[0,0].plot(range(len(loss_history["train"])), loss_history["train"], label="Loss (train)")
                ax[0,0].plot(range(len(loss_history["val"])), loss_history["val"], label="Loss (val)")
                ax[0,0].axvline(x=np.argmin(loss_history["val"]))
                ax[0,0].set_xlabel("SGD step")
                ax[0,0].legend()

                if mc_dropout:
                    # plot_spike_search_2d_trajectory_distribution(ax[1, 0], test_hole_centers, test_hole_centers[i],
                    #                                              pred_real_test_world_samples[:, i], pred_sim_test_world[i],
                    #                                              test_real[i], test_sim[i])
                    # plot_spike_search_1d_xy_distribution(ax[0, 1], ax[1, 1], pred_real_test_world_samples[:, i],
                    #                                      pred_sim_test_world[i], test_real[i], test_sim[i])
                    raise NotImplementedError()
                else:
                    label_real = test_real[i].cpu()
                    pred_real = Y_pred[i].cpu()
                    hole_center = test_hole_centers[i].cpu()
                    c = "green" if pred_real[-1, 1] > 0.5 else "red"
                    # X
                    ax[0,1].plot(range(len(pred_real)), pred_real[:, 2], color=c, label="pred")
                    ax[0,1].plot(range(len(label_real)), label_real[:, 2], color="black", label="label")
                    # Y
                    ax[1,0].plot(range(len(pred_real)), pred_real[:, 3], color=c, label="pred")
                    ax[1,0].plot(range(len(label_real)), label_real[:, 3], color="black", label="label")
                    # Z
                    ax[1,1].plot(range(len(pred_real)), pred_real[:, 4], color=c, label="pred")
                    ax[1,1].plot(range(len(label_real)), label_real[:, 4], color="black", label="label")
                    ax[1,1].axhline(hole_center[2])
                plt.subplots_adjust(left=0.04, bottom=0.067, right=1, top=1)
                plt.show()

            # Test: Grid-sample start states across the entire space, color-code success prob. prediction
            start_states = []
            for x in np.linspace(0.422488, 0.426488, 20):
                for y in np.linspace(0.008449,  0.012449, 20):
                    start_states.append(torch.tensor([x, y, 0.0428629 + 0.003, 1.0, 0, 0, 0], dtype=torch.float32))
            start_states = torch.stack(start_states)
            dummy_inputs = torch.tensor([0.0, 0.0, -0.003, 1.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.1, 0.1],
                                        dtype=torch.float32).unsqueeze(0).repeat(len(start_states), 1)
            fig, ax = plt.subplots()
            plot_success_prob_for_start_states(nt, start_states, dummy_inputs, test_hole_centers, fig, ax)
            endpoints_failed = []
            endpoints_succeeded = []
            for endpoint in test_real[:, -1]:
                if endpoint[1]:
                    endpoints_succeeded.append(endpoint.cpu())
                else:
                    endpoints_failed.append(endpoint.cpu())
            endpoints_failed = torch.stack(endpoints_failed)
            endpoints_succeeded = torch.stack(endpoints_succeeded)
            ax.scatter(endpoints_succeeded[:, 2], endpoints_succeeded[:, 3], color="green", marker="^")
            ax.scatter(endpoints_failed[:, 2], endpoints_failed[:, 3], color="red", marker="^")
            plt.show()

        Y_label_norm = normalize_traj(test_real.to(nt.device), nt.limits_Y)
        Y_pred_norm = normalize_traj(Y_pred, nt.limits_Y)
        test_loss = trajectory_loss(Y_pred_norm, Y_label_norm)
        print(f"Test loss: {test_loss}")
        test_losses.append(test_loss.item())
        success_labels = [round(t.item()) for t in test_real[:, -1, 1]]
        success_outputs = [round(t.item()) for t in Y_pred[:, -1, 1]]
        success_acc = metrics.accuracy_score(success_labels, success_outputs)
        print(f"Success acc: {success_acc}")
        test_accs.append(success_acc)
    avg_test_loss = sum(test_losses)/len(test_losses)
    avg_success_acc = sum(test_accs)/len(test_accs)
    print(f"Mean meta_test validation loss: {avg_test_loss:.4f}, success accuracy: {avg_success_acc:.4f}")
    return avg_test_loss, avg_success_acc


def maml_and_test(model_dir, data_dir: str, samples_per_task=128, adapt_steps=2048, adapt_lr=0.01, show=False):
    test_tasks = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    random.shuffle(test_tasks)
    test_dataset = MetaDataset(test_tasks)
    test_losses = []
    test_accs = []
    nt = ProgramPrimitive.load(model_dir)
    maml = l2l.algorithms.MAML(nt, lr=adapt_lr, first_order=True, allow_unused=True).to(device)
    loss_fn = trajectory_loss

    for task_idx, task in enumerate(test_dataset):
        print(f"----- Task {task_idx}")
        # Run fast adaptation
        meta_learner = maml.clone()
        train_batch, valid_batch, test_batch = task.sample_train_validate_test(samples_per_task, device)
        test_inputs, test_hole_centers, test_start_states, test_sim, test_real = test_batch

        meta_learner.eval()
        with torch.no_grad():
            pred_real_test_norm, pred_sim_test_world = meta_learner(test_inputs, test_hole_centers, test_start_states, test_sim)
        real_test_norm = normalize_labels(test_real, nt.output_limits).to(device)
        test_loss = trajectory_loss(pred_real_test_norm, real_test_norm)
        print(f"Test loss before adaptation: {test_loss.item():.4f}")

        meta_learning_experiments.experiments.common_utils.train()
        evaluation_error = fast_adapt_maml(train_batch, valid_batch, meta_learner, loss_fn, adapt_steps, debug=False)
        test_losses.append(evaluation_error.item())
        print(f"Evaluation error during fast adaptation: {evaluation_error.item():.4f}")

        meta_learner.eval()
        with torch.no_grad():
            pred_real_test_norm, pred_sim_test_world = meta_learner(test_inputs, test_hole_centers, test_start_states, test_sim)
        pred_real_test_world = denormalize_outputs(pred_real_test_norm, nt.output_limits)
        if show:
            for i in range(len(pred_real_test_norm)):
                fig, ax = plt.subplots(2, 2, figsize=(12, 10))
                spike_points = test_inputs[i, :16 * 7].reshape((-1, 7))
                spike_points_world = relative_to_absolute(spike_points, test_start_states[i].unsqueeze(0))
                plot_spike_search_2d(ax[1, 0], spike_points_world, test_hole_centers, test_hole_centers[i], pred_real_test_world[i],
                                     pred_sim_test_world[i], test_real[i], test_sim[i])
                plot_spike_search_1d_xy(ax[0, 1], ax[1, 1], pred_real_test_world[i], pred_sim_test_world[i],
                                      test_real[i], test_sim[i])

                plt.subplots_adjust(left=0.04, bottom=0.067, right=1, top=1)
                plt.show()

        real_test_norm = normalize_labels(test_real, nt.output_limits).to(device)
        test_loss = trajectory_loss(pred_real_test_norm, real_test_norm)
        print(f"Test loss after adaptation: {test_loss.item():.4f}")
        test_losses.append(test_loss.item())
        success_labels = [round(t.item()) for t in test_real[:, -1, 1]]
        success_outputs = [round(t.item()) for t in pred_real_test_norm[:, -1, 1]]
        success_acc = metrics.accuracy_score(success_labels, success_outputs)
        test_accs.append(success_acc)
    avg_test_loss = sum(test_losses)/len(test_losses)
    avg_success_acc = sum(test_accs)/len(test_accs)
    print(f"Mean meta_test validation loss: {avg_test_loss:.4f}, success accuracy: {avg_success_acc:.4f}")
    return avg_test_loss, avg_success_acc


def main(args):
    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)
    if args.command == "finetune":
        if not args.gridsearch:
            finetune_and_test(args.model_dir, args.test_data_dir, experiment_config, samples_per_task=128, adapt_steps=512,
                              learning_rate=5e-4, show=args.show, mc_dropout=args.monte_carlo_dropout)
        else:
            results_filepath = "/home/bal/Projects/meta-learning-experiments/output/spike/gridsearch_no_maml.csv"
            with open(results_filepath, "w+") as results_file:
                results_file.write("samples_per_task,adapt_steps,learning_rate,avg_test_loss,avg_success_acc\n")
            for samples_per_task in [128, 64, 32, 16, 8]:
                for learning_rate in [0.001, 0.01, 0.1]:
                    for adapt_steps in [1024, 2048, 4096]:
                        avg_test_loss, avg_success_acc = finetune_and_test(args.model_dir, args.test_data_dir,
                                                                           samples_per_task=samples_per_task,
                                                                           adapt_steps=adapt_steps,
                                                                           learning_rate=learning_rate)
                        with open(results_filepath, "a+") as results_file:
                            line = f"{samples_per_task},{adapt_steps},{learning_rate},{avg_test_loss},{avg_success_acc}"
                            results_file.write(f"{line}\n")
    else:
        # Make sure the MAML hyperparameters are the same as during training
        with open(os.path.join(args.model_dir, "model_config.json")) as model_config_file:
            model_config = json.load(model_config_file)
        samples_per_task = model_config["batch_size"]
        adapt_steps = model_config["adapt_steps"]
        adapt_lr = model_config["learning_rate"]
        maml_and_test(args.model_dir, args.test_data_dir, samples_per_task=samples_per_task, adapt_steps=adapt_steps,
                      adapt_lr=adapt_lr, show=args.show)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("command", type=str, choices=["finetune", "maml"])
    parser.add_argument("model_dir", type=str)
    parser.add_argument("test_data_dir", type=str)
    parser.add_argument("experiment_config", type=str)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--gridsearch", action="store_true")
    parser.add_argument("--monte_carlo_dropout", action="store_true")
    main(parser.parse_args())
