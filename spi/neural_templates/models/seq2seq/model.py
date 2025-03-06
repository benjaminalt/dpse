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

import random

import torch
from torch import nn


class ResidualGRU(nn.Module):
    """
    Maps an augmented trajectory (static inputs + traj.) to a residual trajectory
    """
    def __init__(self, static_input_size, sequence_input_size, environment_embedding_size=None,
                 hidden_size=64, output_size=2+7+6, num_layers=4, dropout_p=0.2):
        super(ResidualGRU, self).__init__()
        self.static_input_size = static_input_size
        self.sequence_input_size = sequence_input_size
        self.environment_embedding_size = environment_embedding_size
        self.use_env_embedding = environment_embedding_size is not None
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        input_size = static_input_size + environment_embedding_size if self.use_env_embedding else static_input_size
        self.input_layer = nn.Linear(input_size, 25)
        self.fcn1 = nn.Linear(sequence_input_size + 25, self.hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # self.ln2 = nn.LayerNorm(hidden_size)
        self.gru2 = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        # self.ln3 = nn.LayerNorm(hidden_size)
        self.gru3 = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        # self.ln4 = nn.LayerNorm(hidden_size)
        self.gru4 = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        # self.ln5 = nn.LayerNorm(hidden_size)
        self.fcn2 = nn.Linear(hidden_size, hidden_size)
        # self.ln6 = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        all_params = sum(p.numel() for p in self.parameters())
        print("ResidualGRU, total number of parameters: {}".format(all_params))

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, static_inputs, sequence, environment_embedding=None, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(static_inputs.size(0), static_inputs.device)
        inputs = torch.cat((static_inputs, environment_embedding), dim=-1) if self.use_env_embedding else static_inputs
        x = nn.SELU()(self.input_layer(inputs))
        inputs_sequence = x.unsqueeze(1).repeat(1, sequence.size(1), 1)
        augmented_sequence = torch.cat((inputs_sequence, sequence), dim=-1)
        x = nn.SELU()(self.fcn1(augmented_sequence))
        x_1, hidden = self.gru1(x, hidden)
        x_1 = nn.Dropout(self.dropout_p)(x_1)
        x_2, hidden = self.gru2(torch.cat((x, x_1), dim=-1), hidden)
        x_2 = nn.Dropout(self.dropout_p)(x_2)
        x_3, hidden = self.gru3(torch.cat((x_1, x_2), dim=-1), hidden)
        x_3 = nn.Dropout(self.dropout_p)(x_3)
        x_4, hidden = self.gru4(torch.cat((x_2, x_3), dim=-1), hidden)
        x_4 = nn.Dropout(self.dropout_p)(x_4)
        x = nn.SELU()(self.fcn2(x_4))
        x = self.output_layer(x)
        x[:, :, :2] = nn.Sigmoid()(x[:, :, :2])     # EOS & success probabilities
        return x

    def save(self, filepath: str):
        torch.save({
            "state_dict": self.state_dict(),
            "static_input_size": self.static_input_size,
            "sequence_input_size": self.sequence_input_size,
            "environment_embedding_size": self.environment_embedding_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "num_layers": self.num_layers,
            "dropout_p": self.dropout_p
        }, filepath)

    @staticmethod
    def load(filepath: str, device):
        params = torch.load(filepath, map_location=device)
        model = ResidualGRU(params["static_input_size"], params["sequence_input_size"], params["environment_embedding_size"],
                            params["hidden_size"], params["output_size"], params["num_layers"], params["dropout_p"])
        model.load_state_dict(params["state_dict"])
        return model


class DenseNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout_p):
        super(DenseNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.input_layer = nn.Linear(self.input_size * 500, self.hidden_size * 500)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size * 500, self.hidden_size * 500) for _ in range(self.num_layers)])
        self.output_layer = nn.Linear(self.hidden_size * 500, output_size * 500)
        all_params = sum(p.numel() for p in self.parameters())
        print("DenseNet, total number of parameters: {}".format(all_params))

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, x, hidden):
        x = x.reshape((x.size(0), x.size(1) * x.size(2)))  # flatten
        x = nn.SELU()(self.input_layer(x))
        for i in range(len(self.linear_layers)):
            x = nn.SELU()(self.linear_layers[i](x))
        x = self.output_layer(x)
        x = x.reshape((x.size(0), 500, -1))  # unflatten
        x[:, :, :2] = nn.Sigmoid()(x[:, :, :2])
        return x

    def save(self, filepath: str):
        torch.save({
            "state_dict": self.state_dict(),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout_p": self.dropout_p
        }, filepath)

    @staticmethod
    def load(filepath: str, device):
        params = torch.load(filepath, map_location=device)
        model = DenseNet(params["input_size"], params["output_size"], params["hidden_size"],
                         params["num_layers"], params["dropout_p"])
        model.load_state_dict(params["state_dict"])
        return model


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout_p):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.num_layers)
        self.decoder = AttnDecoderRNN(self.hidden_size, self.output_size, self.num_layers,
                                      self.dropout_p, max_length=500, attention=True)

    def save(self, filepath: str):
        torch.save({
            "state_dict": self.state_dict(),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout_p": self.dropout_p
        }, filepath)

    @staticmethod
    def load(filepath: str, device):
        params = torch.load(filepath, map_location=device)
        model = EncoderDecoder(params["input_size"], params["output_size"], params["hidden_size"],
                               params["num_layers"], params["dropout_p"])
        model.load_state_dict(params["state_dict"])
        return model

    def forward(self, x, teacher_forced_labels=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        encoder_hidden = self.encoder.init_hidden(batch_size, x.device)
        encoder_output, encoder_hidden = self.encoder(x, encoder_hidden)
        decoder_input = torch.zeros(batch_size, 1, self.decoder.output_size, device=x.device)
        decoder_hidden = encoder_hidden
        decoder_outputs = None
        if teacher_forced_labels is not None:
            # Teacher forcing: Feed the target as the next input
            for di in range(seq_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                if decoder_outputs is None:
                    decoder_outputs = decoder_output
                else:
                    decoder_outputs = torch.cat((decoder_outputs, decoder_output[:, -1, :].unsqueeze(1)), dim=1)
                decoder_input = teacher_forced_labels[:, di, :].unsqueeze(1)  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(seq_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                if decoder_outputs is None:
                    decoder_outputs = decoder_output
                else:
                    decoder_outputs = torch.cat((decoder_outputs, decoder_output[:, -1, :].unsqueeze(1)), dim=1)
                decoder_input = decoder_output.detach()  # detach from history as input
        x = decoder_outputs
        x[:, :, :2] = nn.Sigmoid()(x[:, :, :2])
        return x

    @staticmethod
    def scheduled_sampling(epoch_idx, n_epochs):
        """
        See arXiv:1506.03099.
        Decide whether or not to use teacher forcing with a probability decreasing during training.
        :return: True for teacher forcing, False otherwise
        """
        teacher_forcing_prob = 1 - epoch_idx / n_epochs
        return random.random() < teacher_forcing_prob


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.num_layers)

    def forward(self, inp, hidden):
        gru_input = self.linear(inp)
        output, hidden = self.gru(gru_input, hidden)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers,
                 dropout_p=0.1, max_length=500, attention=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attention = attention

        self.linear = nn.Linear(self.output_size, self.hidden_size)
        if self.attention:
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inp, hidden, encoder_outputs):
        batch_size = inp.size(0)

        embedded = self.dropout(nn.SELU()(self.linear(inp)))

        if self.attention:
            attn_inputs = torch.cat((embedded, hidden[-1, :, :].view(batch_size, 1, hidden.size(-1))), -1)
            attn_weights = torch.nn.Softmax(dim=-1)(self.attn(attn_inputs))
            attn_applied = torch.bmm(attn_weights, encoder_outputs)
            output = torch.cat((embedded, attn_applied), -1)
            output = self.attn_combine(output)
            output = torch.nn.SELU()(output)
        else:
            output = embedded

        output, hidden = self.gru(output, hidden)

        output = self.out(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
