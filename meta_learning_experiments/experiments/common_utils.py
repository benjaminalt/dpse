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
import random
from copy import deepcopy
from typing import List, Tuple

import learn2learn as l2l
import numpy as np
import torch
from spi.neural_programs.program_component import ProgramComponent
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import MultipleLocator
from spi.data.dataset import DirectoryDataset
from spi.neural_templates.neural_template_utils import normalize_traj, success_probability, trajectory_length

from scipy import stats
from sklearn import mixture, preprocessing
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from spi.utils.data_io import load_tensors_from_file

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")


class MetaDataset(Dataset):
    def __init__(self, task_data_files: List[str], **kwargs):
        super(MetaDataset, self).__init__(**kwargs)
        self.task_data_files = task_data_files

    def __len__(self):
        return len(self.task_data_files)

    def __getitem__(self, index):
        return TaskDataset(self.task_data_files[index])

    def sample(self):
        random_idx = random.randint(0, len(self) - 1)
        return self[random_idx]


class TaskDataset(Dataset):
    def __init__(self, data_filepath: str, **kwargs):
        super(TaskDataset, self).__init__(**kwargs)
        self.data = load_tensors_from_file(data_filepath, multiproc=False)
        self.num_samples = len(self.data["inputs"])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data["inputs"][index], self.data["hole_centers"][index], self.data["start_states"][index], \
               self.data["sim"][index], self.data["real"][index]

    def sample_train_validate_test(self, num_train, num_validate, num_test, device: torch.device=None):
        random_indices = torch.randperm(self.num_samples)
        train_batch = self.sample_using_indices(random_indices[:num_train], device)
        validate_batch = self.sample_using_indices(random_indices[num_train:num_train+num_validate], device)
        test_batch = self.sample_using_indices(random_indices[num_train+num_validate:num_train+num_validate+num_test], device)
        return train_batch, validate_batch, test_batch

    def sample_using_indices(self, indices, device: torch.device=None):
        inputs = self.data["inputs"][indices]
        hole_centers = self.data["hole_centers"][indices]
        start_states = self.data["start_states"][indices]
        sim = self.data["sim"][indices]
        real = self.data["real"][indices]
        if device is not None:
            inputs, hole_centers, start_states, sim, real = inputs.to(device), hole_centers.to(device), start_states.to(
                device), sim.to(device), real.to(device)
        return inputs, hole_centers, start_states, sim, real


def trajectory_loss(output_traj: torch.Tensor, label_traj: torch.Tensor) -> torch.Tensor:
    eos_loss = torch.nn.BCELoss()
    success_loss = torch.nn.BCELoss()
    traj_loss = torch.nn.MSELoss()
    return eos_loss(output_traj[:, :, 0], label_traj[:, :, 0]) \
           + success_loss(output_traj[:, :, 1], label_traj[:, :, 1]) \
           + traj_loss(output_traj[:, :, 2:], label_traj[:, :, 2:])


def fast_adapt_maml(train_batch, valid_batch, learner, loss_fn, adaptation_steps, debug=False, invert_success_label=False):
    """
    See https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py
    """
    # Adapt the model
    train_inputs, train_hole_centers, train_start_states, train_sim, train_real = train_batch
    valid_inputs, valid_hole_centers, valid_start_states, valid_sim, valid_real = valid_batch
    if invert_success_label:
        train_real[:, :, 1] = (~train_real[:, :, 1].bool()).float()
        valid_real[:, :, 1] = (~valid_real[:, :, 1].bool()).float()
    train_real_norm = normalize_traj(train_real, learner.limits_Y.to(device))
    valid_real_norm = normalize_traj(valid_real, learner.limits_Y.to(device))

    if debug:
        train_errors = []
    for step in range(adaptation_steps):
        Y_norm, s_out_norm = learner(train_inputs, train_start_states, train_sim, denormalize_out=False)
        train_error = loss_fn(Y_norm, train_real_norm)
        if debug:
            train_errors.append(train_error.item())
        learner.adapt(train_error)

    if debug:
        plt.plot(range(len(train_errors)), train_errors, label="Train loss")
        plt.legend()
        plt.show()
    # Evaluate the adapted model
    Y_norm, s_out_norm = learner(valid_inputs, valid_start_states, valid_sim, denormalize_out=False)
    valid_error = loss_fn(Y_norm, valid_real_norm)
    return valid_error


def fast_adapt_reptile(train_batch, valid_batch, learner, loss_fn, adaptation_steps, opt, debug=False,
                       invert_success_label=False):
    """
    https://github.com/learnables/learn2learn/blob/master/examples/vision/reptile_miniimagenet.py
    Only use the adaptation data to update parameters. (evaluation is only indicative.)
    """
    # Adapt the model
    train_inputs, train_hole_centers, train_start_states, train_sim, train_real = train_batch
    valid_inputs, valid_hole_centers, valid_start_states, valid_sim, valid_real = valid_batch
    if invert_success_label:
        train_real[:, :, 1] = (~train_real[:, :, 1].bool()).float()
        valid_real[:, :, 1] = (~valid_real[:, :, 1].bool()).float()
    train_real_normalized = normalize_traj(train_real, learner.limits_Y.to(device))
    valid_real_normalized = normalize_traj(valid_real, learner.limits_Y.to(device))

    if debug:
        train_errors = []
    for step in range(adaptation_steps):
        pred_real_train, pred_sim_train = learner(train_inputs, train_start_states, train_sim, denormalize_out=False)
        opt.zero_grad()
        train_error = loss_fn(pred_real_train, train_real_normalized)
        if debug:
            train_errors.append(train_error.item())
        train_error.backward()
        opt.step()

    if debug:
        plt.plot(range(len(train_errors)), train_errors, label="Train loss")
        plt.legend()
        plt.show()
    # Evaluate the adapted model
    pred_real_valid, pred_sim_valid = learner(valid_inputs, valid_start_states, valid_sim, denormalize_out=False)
    valid_error = loss_fn(pred_real_valid, valid_real_normalized)
    return valid_error


def compute_limits_meta_dataset(meta_dataset: MetaDataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Computing limits...")
    input_limits = None
    start_state_limits = None
    sim_limits = None
    real_limits = None

    for task_dataset in tqdm(meta_dataset, total=len(meta_dataset)):
        task_dataloader = DataLoader(task_dataset, batch_size=256, shuffle=False)
        for input_batch, hole_center_batch, start_states_batch, sim_batch, real_batch in task_dataloader:
            batch_input_limits = torch.stack((torch.min(input_batch, dim=0)[0], torch.max(input_batch, dim=0)[0]))
            batch_start_state_limits = torch.stack((torch.min(start_states_batch, dim=0)[0], torch.max(start_states_batch, dim=0)[0]))
            batch_sim_limits = torch.stack(
                (torch.min(sim_batch.view(-1, sim_batch.size(-1)), dim=0)[0], torch.max(sim_batch.view(-1, sim_batch.size(-1)), dim=0)[0]))
            batch_real_limits = torch.stack(
                (torch.min(real_batch.view(-1, real_batch.size(-1)), dim=0)[0], torch.max(real_batch.view(-1, real_batch.size(-1)), dim=0)[0]))
            if input_limits is None:
                input_limits = batch_input_limits
                start_state_limits = batch_start_state_limits
                sim_limits = batch_sim_limits
                real_limits = batch_real_limits
            else:
                input_limits = torch.stack((torch.min(batch_input_limits[0], input_limits[0]),
                                            torch.max(batch_input_limits[1], input_limits[1])))
                start_state_limits = torch.stack((torch.min(batch_start_state_limits[0], start_state_limits[0]),
                                                  torch.max(batch_start_state_limits[1], start_state_limits[1])))
                sim_limits = torch.stack((torch.min(batch_sim_limits[0], sim_limits[0]),
                                          torch.max(batch_sim_limits[1], sim_limits[1])))
                real_limits = torch.stack((torch.min(batch_real_limits[0], real_limits[0]),
                                           torch.max(batch_real_limits[1], real_limits[1])))

    # If min and max are identical, this causes trouble when scaling --> division by zero produces NaN
    # Set min to val - 1 and max to val + 1
    def disambiguate_identical_limits(limits: torch.Tensor):
        for dim_idx in range(limits.size(1)):
            if limits[0, dim_idx] == limits[1, dim_idx]:
                orig_value = limits[0, dim_idx].clone()
                limits[0, dim_idx] = orig_value - 1  # new min
                limits[1, dim_idx] = orig_value + 1  # new max

    disambiguate_identical_limits(input_limits)
    disambiguate_identical_limits(start_state_limits)
    disambiguate_identical_limits(sim_limits)
    disambiguate_identical_limits(real_limits)

    return input_limits, start_state_limits, sim_limits, real_limits


def compute_limits_directory_dataset(dataset: DirectoryDataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Computing limits...")
    input_limits = None
    start_state_limits = None
    sim_limits = None
    real_limits = None

    dataloader = DataLoader(dataset, batch_size=256, num_workers=1, shuffle=False)

    for hole_center_batch, input_batch, real_batch, sim_batch, start_states_batch in tqdm(dataloader):
        batch_input_limits = torch.stack((torch.min(input_batch, dim=0)[0], torch.max(input_batch, dim=0)[0]))
        batch_start_state_limits = torch.stack((torch.min(start_states_batch, dim=0)[0], torch.max(start_states_batch, dim=0)[0]))
        batch_sim_limits = torch.stack(
            (torch.min(sim_batch.view(-1, sim_batch.size(-1)), dim=0)[0], torch.max(sim_batch.view(-1, sim_batch.size(-1)), dim=0)[0]))
        batch_real_limits = torch.stack(
            (torch.min(real_batch.view(-1, real_batch.size(-1)), dim=0)[0], torch.max(real_batch.view(-1, real_batch.size(-1)), dim=0)[0]))
        if input_limits is None:
            input_limits = batch_input_limits
            start_state_limits = batch_start_state_limits
            sim_limits = batch_sim_limits
            real_limits = batch_real_limits
        else:
            input_limits = torch.stack((torch.min(batch_input_limits[0], input_limits[0]),
                                        torch.max(batch_input_limits[1], input_limits[1])))
            start_state_limits = torch.stack((torch.min(batch_start_state_limits[0], start_state_limits[0]),
                                              torch.max(batch_start_state_limits[1], start_state_limits[1])))
            sim_limits = torch.stack((torch.min(batch_sim_limits[0], sim_limits[0]),
                                      torch.max(batch_sim_limits[1], sim_limits[1])))
            real_limits = torch.stack((torch.min(batch_real_limits[0], real_limits[0]),
                                       torch.max(batch_real_limits[1], real_limits[1])))

    # If min and max are identical, this causes trouble when scaling --> division by zero produces NaN
    # Set min to val - 1 and max to val + 1
    def disambiguate_identical_limits(limits: torch.Tensor):
        for dim_idx in range(limits.size(1)):
            if limits[0, dim_idx] == limits[1, dim_idx]:
                orig_value = limits[0, dim_idx].clone()
                limits[0, dim_idx] = orig_value - 1  # new min
                limits[1, dim_idx] = orig_value + 1  # new max

    disambiguate_identical_limits(input_limits)
    disambiguate_identical_limits(start_state_limits)
    disambiguate_identical_limits(sim_limits)
    disambiguate_identical_limits(real_limits)

    return input_limits, start_state_limits, sim_limits, real_limits


def train_maml(model: ProgramComponent, data_dir: str, output_dir: str, num_epochs=100, tasks_per_epoch=16,
               samples_per_task=64, adapt_lr=0.01, meta_lr=0.001, adapt_steps=5, gridsearch=False,
               invert_success_label=False, first_order=True):
    """
    Adapted from https://github.com/learnables/learn2learn/blob/48e8c8b710e7d6a141ca33fe03cf1f13ca727315/examples/maml_sine.py
    """
    all_tasks = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    split_idx = int(0.9 * len(all_tasks))
    train_tasks = all_tasks[:split_idx]
    validate_tasks = all_tasks[split_idx:]
    all_meta_dataset = MetaDataset(all_tasks)
    train_meta_dataset = MetaDataset(train_tasks)
    validate_meta_dataset = MetaDataset(validate_tasks)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=first_order, allow_unused=True).to(device)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss_fn = trajectory_loss

    loss_history = {"train": [], "val": []}
    print(f"Training for {num_epochs} epochs")
    with torch.backends.cudnn.flags(enabled=False):
        for iteration in range(num_epochs):
            meta_train_error = 0.0
            meta_valid_error = 0.0
            opt.zero_grad()
            for task_idx in range(tasks_per_epoch):
                print(f"Task {task_idx + 1}/{tasks_per_epoch}")
                meta_learner = maml.clone()
                train_task = train_meta_dataset.sample()
                train_batch, valid_batch, test_batch = train_task.sample_train_validate_test(num_train=samples_per_task,
                                                                                             num_validate=samples_per_task,
                                                                                             num_test=samples_per_task,
                                                                                             device=device)
                evaluation_error = fast_adapt_maml(train_batch, valid_batch, meta_learner, loss_fn, adapt_steps,
                                                   invert_success_label=invert_success_label)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()

                meta_learner = maml.clone()
                valid_task = validate_meta_dataset.sample()
                train_batch, valid_batch, test_batch = valid_task.sample_train_validate_test(num_train=samples_per_task,
                                                                                             num_validate=samples_per_task,
                                                                                             num_test=samples_per_task,
                                                                                             device=device)
                evaluation_error = fast_adapt_maml(train_batch, valid_batch, meta_learner, loss_fn, adapt_steps,
                                                   invert_success_label=invert_success_label)
                meta_valid_error += evaluation_error.item()

            avg_meta_train_loss = meta_train_error / tasks_per_epoch
            loss_history["train"].append(avg_meta_train_loss)
            avg_meta_valid_loss = meta_valid_error / tasks_per_epoch
            loss_history["val"].append(avg_meta_valid_loss)
            print(f"{iteration}: Train error {avg_meta_train_loss:.4f}, val error {avg_meta_valid_loss:.4f}")

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / tasks_per_epoch)
            opt.step()

        if not gridsearch:
            model.save(output_dir, loss_history)

    return loss_history


def train_reptile(model: ProgramComponent, data_dir: str, output_dir: str, num_epochs=100, tasks_per_epoch=16,
                  samples_per_task=64, adapt_lr=0.01, meta_lr=0.001, adapt_steps=5, gridsearch=False, meta_decay=True,
                  invert_success_label=False):
    """
    Adapted from https://github.com/learnables/learn2learn/blob/master/examples/vision/reptile_miniimagenet.py
    """
    all_tasks = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    split_idx = int(0.9 * len(all_tasks))
    train_tasks = all_tasks[:split_idx]
    validate_tasks = all_tasks[split_idx:]
    all_meta_dataset = MetaDataset(all_tasks)
    train_meta_dataset = MetaDataset(train_tasks)
    validate_meta_dataset = MetaDataset(validate_tasks)
    opt = optim.Adam(model.parameters(), meta_lr)
    adapt_opt = optim.Adam(model.parameters(), adapt_lr, betas=(0, 0.999))
    adapt_opt_state = adapt_opt.state_dict()
    loss_fn = trajectory_loss

    model = model.to(device)

    loss_history = {"train": [], "val": []}
    print(f"Training for {num_epochs} epochs")
    for iteration in range(num_epochs):
        # anneal meta-lr
        if meta_decay:
            frac_done = float(iteration) / num_epochs
            new_lr = frac_done * meta_lr + (1 - frac_done) * meta_lr
            for pg in opt.param_groups:
                pg['lr'] = new_lr

        # zero-grad the parameters
        for p in model.parameters():
            p.grad = torch.zeros_like(p.data)

        meta_train_error = 0.0
        meta_valid_error = 0.0
        for _ in range(tasks_per_epoch):
            learner = deepcopy(model)
            adapt_opt = optim.Adam(learner.parameters(),
                                   lr=adapt_lr,
                                   betas=(0, 0.999))
            adapt_opt.load_state_dict(adapt_opt_state)
            train_task = train_meta_dataset.sample()
            train_batch, valid_batch, test_batch = train_task.sample_train_validate_test(num_train=samples_per_task,
                                                                                         num_validate=samples_per_task,
                                                                                         num_test=samples_per_task,
                                                                                         device=device)
            evaluation_error = fast_adapt_reptile(train_batch, valid_batch, learner, loss_fn, adapt_steps, adapt_opt,
                                                  invert_success_label=invert_success_label)
            adapt_opt_state = adapt_opt.state_dict()
            for p, l in zip(model.parameters(), learner.parameters()):
                p.grad.data.add_(-1.0, l.data)
            meta_train_error += evaluation_error.item()

            # Compute meta-validation loss
            learner = deepcopy(model)
            adapt_opt = optim.Adam(learner.parameters(),
                                   lr=adapt_lr,
                                   betas=(0, 0.999))
            adapt_opt.load_state_dict(adapt_opt_state)
            valid_task = validate_meta_dataset.sample()
            train_batch, valid_batch, test_batch = valid_task.sample_train_validate_test(num_train=samples_per_task,
                                                                                         num_validate=samples_per_task,
                                                                                         num_test=samples_per_task,
                                                                                         device=device)
            evaluation_error = fast_adapt_reptile(train_batch, valid_batch, learner, loss_fn, adapt_steps, adapt_opt,
                                                  invert_success_label=invert_success_label)
            meta_valid_error += evaluation_error.item()

        avg_meta_train_loss = meta_train_error / tasks_per_epoch
        loss_history["train"].append(avg_meta_train_loss)
        avg_meta_valid_loss = meta_valid_error / tasks_per_epoch
        loss_history["val"].append(avg_meta_valid_loss)
        print(f"{iteration}: Train error {avg_meta_train_loss:.4f}, val error {avg_meta_valid_loss:.4f}")

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / tasks_per_epoch).add_(p.data)
        opt.step()

    if not gridsearch:
        model.save(output_dir, loss_history)

    return loss_history


def finetune_on_task(train_batch, valid_batch, nt: ProgramComponent, adapt_steps=4096, lr=0.001, plot_grads=False,
                     invert_success_label=False):
    train_inputs, train_hole_centers, train_start_states, train_sim, train_real = train_batch
    valid_inputs, valid_hole_centers, valid_start_states, valid_sim, valid_real = valid_batch

    proportion_successful_train = train_real[:, -1, 1].mean().item()
    print(f"Proportion of successful executions in fine-tuning train set: {proportion_successful_train:.4f}")
    optim = torch.optim.Adam(nt.parameters(), lr=lr, weight_decay=1e-3)

    if invert_success_label:
        train_real[:, :, 1] = (~train_real[:, :, 1].bool()).float()
        valid_real[:, :, 1] = (~valid_real[:, :, 1].bool()).float()

    train_real_normalized = normalize_traj(train_real.to(nt.device), nt.limits_Y.to(nt.device))
    valid_real_normalized = normalize_traj(valid_real.to(nt.device), nt.limits_Y.to(nt.device))
    loss_history = {"train": [], "val": []}

    # Adaptation loop. It is important to prevent overfitting on the samples_per_task here, particularly
    # for large adapt_steps
    best_weights = nt.state_dict()
    best_test_loss = np.inf
    best_test_idx = 0
    for i in tqdm(range(adapt_steps)):
        nt.train()
        Y_pred, s_out_pred = nt(train_inputs, train_start_states, train_sim, denormalize_out=False)
        train_loss = trajectory_loss(Y_pred, train_real_normalized)
        optim.zero_grad()
        train_loss.backward()
        optim.step()
        loss_history["train"].append(train_loss.item())

        nt.eval()
        with torch.no_grad():
            Y_pred, s_out_pred = nt(valid_inputs, valid_start_states, valid_sim, denormalize_out=False)
        val_loss = trajectory_loss(Y_pred, valid_real_normalized)
        loss_history["val"].append(val_loss.item())
        if val_loss < best_test_loss:
            best_test_loss = val_loss.item()
            best_weights = nt.state_dict()
            best_test_idx = i
    nt.load_state_dict(best_weights)
    return nt, loss_history


def set_offset_tag(rps, node_id: int, offset_x: float, offset_y: float):
    raise NotImplementedError()


def plot_success_prob_for_start_states(nt, start_states, dummy_inputs, test_hole_centers, fig=None, ax=None):
    show = False
    if fig is None:
        fig, ax = plt.subplots()
        show = True
    nt.eval()
    with torch.no_grad():
        Y_pred, s_out_pred = nt(dummy_inputs, start_states, denormalize_out=True)
    xs = start_states[:, 0].cpu()
    ys = start_states[:, 1].cpu()
    cm = plt.get_cmap("RdYlGn")
    c = success_probability(Y_pred).cpu()
    ax.scatter(test_hole_centers[:, 0].cpu(), test_hole_centers[:, 1].cpu(), color="gray", alpha=0.5)
    scat = plt.scatter(xs, ys, c=c, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    fig.colorbar(scat)
    if show:
        plt.show()


def success_rate_loss(x: torch.Tensor, Y: torch.Tensor):
    return 1 - torch.mean(success_probability(Y)).clamp(0, 1)


def cycle_time_loss(x: torch.Tensor, Y: torch.Tensor):
    return trajectory_length(Y).mean()


def plot_hole_distribution_2d(ax, fig, hole_distribution, plot_in_mm=True, add_colorbar=True):
    # Fit GMM
    hole_samples = np.array(hole_distribution.sample(1000))
    gmm = mixture.GaussianMixture(6)
    scaler = preprocessing.MinMaxScaler()
    if plot_in_mm:  # Multiply by 1000
        points_scaled = scaler.fit_transform(np.array([hole[:2] * 1000 for hole in hole_samples]))
    else:   # Plot as-is
        points_scaled = scaler.fit_transform(np.array([hole[:2] for hole in hole_samples]))
    gmm.fit(points_scaled)
    x = np.linspace(-0.75, 1.25)
    y = np.linspace(-0.75, 1.25)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = np.exp(gmm.score_samples(XX))
    Z = Z.reshape(X.shape)
    Z[Z < 0.1] = np.nan  # Set to nan so I can use cmap.set_bad("white") for zero probabilities
    mins_maxes = scaler.inverse_transform([[-0.75, -0.75], [1.25, 1.25]])
    x = np.linspace(mins_maxes[0, 0], mins_maxes[1, 0])
    y = np.linspace(mins_maxes[0, 1], mins_maxes[1, 1])
    X, Y = np.meshgrid(x, y)
    cmap = cm.get_cmap("jet").copy()
    cmap.set_bad(color="white")
    contour = ax.contourf(X, Y, Z, 10, cmap=cmap, alpha=0.5)
    if add_colorbar:
        cbar = fig.colorbar(contour)
        cbar.ax.set_ylabel("$f_{H_t}$")
    return contour, mins_maxes


def torch_uniform_in_range(rng: torch.Tensor) -> torch.Tensor:
    if type(rng) != torch.Tensor:
        rng = torch.tensor(rng, dtype=torch.float32)
    if len(rng.size()) > 1:
        return (rng[0] + torch.rand(rng.size(-1)) * (rng[1] - rng[0])).float()
    return (rng[0] + torch.rand(1) * (rng[1] - rng[0])).float()
