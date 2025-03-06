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

import math
import os

import tables
import torch
from natsort import natsorted
from torch.utils.data import IterableDataset
from itertools import cycle, chain, islice


class DirectoryDataset(IterableDataset):
    """
    Adapted from https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """
    def __init__(self, directory: str, start: int = 0, end: int = None):
        super(DirectoryDataset, self).__init__()
        self.start = start
        self.data_filepaths = list(map(lambda filename: os.path.join(directory, filename), natsorted(filter(lambda filename: filename.endswith(".h5"), os.listdir(directory)))))
        self.size = sum(map(self._count_data_in_file, self.data_filepaths))
        if end is None:
            self.end = self.size
        else:
            assert end <= self.size
            self.end = end
        assert self.start < self.end

    def _read_file(self, filepath: str):
        with tables.open_file(filepath) as data_file:
            nodes = sorted(data_file.list_nodes("/"), key=lambda node: node._v_name)
            for batch in data_file.list_nodes(f"/{nodes[0]._v_name}", classname="Array"):
                batch_data_dict = {node._v_name: data_file.get_node(f"/{node._v_name}/{batch.name}").read() for node in nodes}
                batch_data_list = [dict(zip(batch_data_dict, i)) for i in zip(*batch_data_dict.values())]
                for data_dict in batch_data_list:
                    data_list = [torch.from_numpy(data_dict[key]).float() for key in data_dict.keys()]
                    yield data_list

    def _count_data_in_file(self, filepath: str):
        length = 0
        with tables.open_file(filepath) as data_file:
            nodes = data_file.list_nodes("/")
            for batch in data_file.list_nodes(f"/{nodes[0]._v_name}", classname="Array"):
                length += len(batch)
        return length

    def _get_stream(self):
        paths = cycle(self.data_filepaths)
        return chain.from_iterable(map(self._read_file, paths))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:     # Single-process data loading
            iter_start = self.start
            iter_end = self.end
        else:                       # In a worker process
            per_worker = int(math.ceil(self.end - self.start) / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        # Now make an iterator for data between iter_start and iter_end
        return islice(self._get_stream(), iter_start, iter_end)

    def __len__(self):
        return self.end - self.start
