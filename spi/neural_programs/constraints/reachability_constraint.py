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

import argparse
import datetime
import json
import os
import time

import tables
import torch
import numpy as np
from torch.utils.data import DataLoader

from spi.neural_programs.constraints.constraint import Constraint
from spi.utils.pytorch_device import device
from spi.utils import transformations

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


class FCN(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super(FCN, self).__init__()
        self.input_size = 7     # 3D pos. + 4D ori.
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(2 * self.hidden_size, self.hidden_size) for _ in range(num_layers)])
        self.output_layer = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = torch.nn.SELU()(self.input_layer(x))
        x_prev = x
        for i in range(len(self.linears)):
            new_input = torch.cat((x_prev, x), dim=-1)
            x_prev = x
            x = torch.nn.SELU()(self.linears[i](new_input))
        x = self.output_layer(x)
        return x

    @staticmethod
    def load(filepath: str):
        params = torch.load(filepath, map_location=device)
        model = FCN(params["hidden_size"], params["num_layers"])
        model.load_state_dict(params["state_dict"])
        return model

    def save(self, filepath: str):
        torch.save({
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }, filepath)


class ReachabilityModel(object):
    def __init__(self, net: FCN, input_limits: torch.Tensor = None, output_limits: torch.Tensor = None):
        self.net = net
        self.input_limits = input_limits
        self.output_limits = output_limits

    @staticmethod
    def load(dirpath: str):
        params = torch.load(os.path.join(dirpath, "fcn.pt"), map_location=device)
        model = FCN(params["hidden_size"], params["num_layers"])
        model.load_state_dict(params["state_dict"])
        input_limits = torch.from_numpy(np.load(os.path.join(dirpath, "input_limits.npy")))
        output_limits = torch.from_numpy(np.load(os.path.join(dirpath, "output_limits.npy")))
        return ReachabilityModel(model, input_limits, output_limits)

    def save(self, training_history: list):
        output_dir = os.path.join(SCRIPT_DIR, "neural_models",
                                  f"{type(self).__name__}_({time.strftime('%Y%m%d-%H%M%S')})")
        os.makedirs(output_dir)
        self.net.save(os.path.join(output_dir, "fcn.pt"))
        np.save(os.path.join(output_dir, "input_limits.npy"), self.input_limits.numpy())
        np.save(os.path.join(output_dir, "output_limits.npy"), self.output_limits.numpy())
        with open(os.path.join(output_dir, "training_history.json"), "w") as json_file:
            json.dump(training_history, json_file)

    def train(self, raw_inputs: torch.Tensor, raw_labels: torch.Tensor, batch_size: int, lr: float, num_epochs: int):
        print(f"Training ReachabilityModel on device {device}")
        self.input_limits = torch.stack((torch.min(raw_inputs, dim=0)[0], torch.max(raw_inputs, dim=0)[0]))
        for dim in range(self.input_limits.size(-1)):
            if self.input_limits[0, dim] == self.input_limits[1, dim]:
                self.input_limits[0, dim] -= 1
                self.input_limits[1, dim] += 1
        self.output_limits = torch.stack((torch.min(raw_labels, dim=0)[0], torch.max(raw_labels, dim=0)[0]))
        normalized_inputs = transformations.scale(raw_inputs, self.input_limits[0], self.input_limits[1], -1, 1)
        normalized_labels = transformations.scale(raw_labels, self.output_limits[0], self.output_limits[1], -1, 1)
        validation_cutoff = int(0.9 * len(normalized_inputs))
        train_data = torch.utils.data.TensorDataset(normalized_inputs[:validation_cutoff],
                                                    normalized_labels[:validation_cutoff])
        validate_data = torch.utils.data.TensorDataset(normalized_inputs[validation_cutoff:],
                                                       normalized_labels[validation_cutoff:])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
        validate_loader = DataLoader(validate_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)

        self.net = self.net.to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        training_history = []
        start = time.time()

        for epoch in range(num_epochs):
            print('{} Epoch {}/{}'.format("#" * 10, epoch + 1, num_epochs))
            epoch_start_time = time.time()
            train_losses = []
            self.net.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                loss = loss_fn(self.net(inputs), labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = sum(train_losses) / len(train_losses)

            validate_losses = []
            self.net.eval()
            with torch.no_grad():
                for inputs, labels in validate_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    loss = loss_fn(self.net(inputs), labels)
                    validate_losses.append(loss.item())
            avg_validate_loss = sum(validate_losses) / len(validate_losses)
            training_history.append([[avg_train_loss], [avg_validate_loss]])
            print(f"Train: {avg_train_loss:.5}, validate: {avg_validate_loss:.5}")
            print("Epoch {} took {:.2f}s".format(epoch + 1, time.time() - epoch_start_time))

        total_training_time = time.time() - start
        print("Training took {}".format(str(datetime.timedelta(seconds=total_training_time)).split(".")[0]))
        self.save(training_history)

    def evaluate(self, raw_inputs: torch.Tensor) -> torch.Tensor:
        normalized_inputs = transformations.scale(raw_inputs, self.input_limits[0], self.input_limits[1], -1, 1)
        self.net.eval()
        normalized_outputs = self.net(normalized_inputs)
        outputs = transformations.scale(normalized_outputs, -1, 1, self.output_limits[0], self.output_limits[1])
        return outputs


class ReachabilityConstraintCartesian(Constraint):
    """
    Evaluate a neural net to compute a reachability score for a given cartesian pose in an environment.
    """
    def __init__(self, model: ReachabilityModel, weight: float = 1.0):
        self.model = model
        self.weight = weight

    @staticmethod
    def load(model_dirpath: str, weight: float = 1.0):
        return ReachabilityConstraintCartesian(ReachabilityModel.load(model_dirpath), weight)

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        reachabilities = self.model.evaluate(trajectory[:, :, 2:9])
        eps = 0.5
        return 1 / torch.clamp(reachabilities.min(), eps)     # Loss is high when reachability is low

    @staticmethod
    def from_dict(dic):
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()


def main(args):
    if args.command == "train":
        hidden_size = 4096
        num_layers = 6
        batch_size = 8192
        lr = 5e-6
        num_epochs = 500

        with tables.open_file(args.data_filepath) as data_file:
            raw_inputs = torch.from_numpy(data_file.root.inputs.read()).float()
            raw_labels = torch.from_numpy(data_file.root.labels.read()).unsqueeze(-1).float()
        model = ReachabilityModel(FCN(hidden_size, num_layers))
        model.train(raw_inputs, raw_labels, batch_size, lr, num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["train"])
    parser.add_argument("data_filepath", type=str)
    main(parser.parse_args())
