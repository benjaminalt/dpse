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

import torch


def unpad_padded_sequence(seq: torch.Tensor):
    if len(seq.size()) > 2:
        raise ValueError("pad_padded_sequence: Don't expect batched tensors")
    end_idx = torch.cumsum(seq[:, 0] < 0.5, 0)[-1]
    return seq[:end_idx.int().item() + 1]


def pad_padded_sequence(seq: torch.Tensor, length: int):
    if len(seq.size()) > 2:
        raise ValueError("pad_padded_sequence: Don't expect batched tensors")
    orig_length = seq.size(0)
    if orig_length < length:
        num_paddings = length - orig_length
        padding = seq[-1,:].unsqueeze(0).repeat(num_paddings, 1)
        padded_seq = torch.cat((seq, padding), dim=0)
        padded_seq[orig_length:, 0] = 1.0
        return padded_seq
    cut_off_seq = seq[:length]
    cut_off_seq[-1, 0] = 1.0
    return cut_off_seq
