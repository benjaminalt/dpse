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

import datetime
import time

import numpy as np
import torch

from spi.neural_programs.program_component import ProgramComponent
from spi.data.dataset import DirectoryDataset
from spi.neural_templates.neural_template_utils import normalize_traj
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_sgd(p: ProgramComponent, data_dir: str, output_dir: str, invert_success_label=False):
    assert p.model is not None and p.model_config is not None
    batch_size = p.model_config["batch_size"]
    all_data = DirectoryDataset(data_dir)

    split_idx = int(0.9 * len(all_data))
    training_data = DirectoryDataset(data_dir, end=split_idx)
    validation_data = DirectoryDataset(data_dir, start=split_idx)
    print("Training/validation split: {}/{}".format(len(training_data), len(validation_data)))
    train_loader = DataLoader(training_data, batch_size=batch_size, drop_last=False, pin_memory=True, num_workers=1)
    validate_loader = DataLoader(validation_data, batch_size=batch_size, drop_last=False, pin_memory=True, num_workers=1)

    optimizer = torch.optim.Adam(p.model.parameters(), lr=p.model_config["learning_rate"], weight_decay=1e-5)
    loss_functions = {
        "trajectory": [torch.nn.MSELoss(), 1],
        "eos": [torch.nn.BCELoss(), 1],
        "success": [torch.nn.BCELoss(), 1],
    }
    training_history = {"train": {key: [] for key in loss_functions.keys()},
                        "validate": {key: [] for key in loss_functions.keys()}}
    training_history["train"]["total"] = []
    training_history["validate"]["total"] = []
    best_weights = None
    best_validation_loss = np.inf

    def training_step(x: torch.Tensor, environment_inputs_world: torch.Tensor, s_in: torch.Tensor, Y_sim: torch.Tensor,
                      Y: torch.Tensor, optimizer, loss_functions, evaluate=False):
        optimizer.zero_grad()

        Y_pred_norm, s_out_pred_norm = p(x, s_in, Y_sim, denormalize_out=False)

        if invert_success_label:
            Y[:, :, 1] = (~Y[:, :, 1].bool()).float()
        Y = Y.to(p.device)
        Y_norm = normalize_traj(Y, p.limits_Y)

        trajectory_loss = loss_functions["trajectory"][0](Y_pred_norm[:, :, 2:], Y_norm[:, :, 2:]) * loss_functions["trajectory"][1]
        eos_loss = loss_functions["eos"][0](Y_pred_norm[:, :, 0], Y_norm[:, :, 0]) * loss_functions["eos"][1]
        success_loss = loss_functions["success"][0](Y_pred_norm[:, :, 1], Y_norm[:, :, 1]) * loss_functions["success"][1]
        loss = trajectory_loss + eos_loss + success_loss

        if not evaluate:
            loss.backward()
            optimizer.step()

        return loss.item(), trajectory_loss.item(), eos_loss.item(), success_loss.item()

    print("Training for {} epochs with batch size {}".format(p.model_config["epochs"],
                                                             p.model_config["batch_size"]))
    start = time.time()
    for epoch in range(p.model_config["epochs"]):
        print('{} Epoch {}/{}'.format("#" * 10, epoch + 1, p.model_config["epochs"]))
        epoch_start_time = time.time()
        total_train_losses = [0] * (len(loss_functions) + 1)
        total_validation_losses = [0] * (len(loss_functions) + 1)

        p.train()
        for environment_inputs, inputs, real, sim, start_states in tqdm(train_loader):
            losses = training_step(inputs, environment_inputs, start_states, sim, real, optimizer,
                                   loss_functions, evaluate=False)
            for i in range(len(losses)):
                total_train_losses[i] += losses[i]
        # Save training losses
        for i, loss_type in enumerate(["total", *loss_functions.keys()]):
            training_history["train"][loss_type].append(total_train_losses[i] / len(train_loader))

        print("Avg train loss: " + ", ".join([f'{training_history["train"][key][-1]:.6f}' for key in training_history["train"].keys()]))

        p.eval()
        with torch.no_grad():
            for environment_inputs, inputs, real, sim, start_states in tqdm(validate_loader):
                losses = training_step(inputs, environment_inputs, start_states, sim, real, optimizer,
                                       loss_functions, evaluate=True)
                for i in range(len(losses)):
                    total_validation_losses[i] += losses[i]
            # Save validation losses
            for i, loss_type in enumerate(["total", *loss_functions.keys()]):
                training_history["validate"][loss_type].append(total_validation_losses[i] / len(validate_loader))
            if training_history["validate"]["total"][-1] < best_validation_loss:
                best_weights = p.model.state_dict()
                best_validation_loss = training_history["validate"]["total"][-1]

        print("Avg val loss: " + ", ".join([f'{training_history["validate"][key][-1]:.6f}' for key in training_history["validate"].keys()]))
        print("Epoch {} took {:.2f}s".format(epoch + 1, time.time() - epoch_start_time))

    total_training_time = time.time() - start
    print("Training took {}".format(str(datetime.timedelta(seconds=total_training_time)).split(".")[0]))
    print("Setting model weights to minimize validation loss")
    p.model.load_state_dict(best_weights)
    p.save(output_dir, training_history)
