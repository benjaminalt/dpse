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

import multiprocessing
import os

import tables
from zipfile import ZipFile
import json
from typing import Tuple, List, Dict

import torch
from filelock import FileLock
from natsort import natsorted


def read_parameter_zipfile(parameterization_filepath: str) -> Tuple[List[str], List[dict]]:
    parameterizations = []
    revision_hashes = []
    with ZipFile(parameterization_filepath) as parameterization_zip:
        for parameterization_path in parameterization_zip.namelist():
            with parameterization_zip.open(parameterization_path) as parameterization_json:
                parameterizations.append(json.load(parameterization_json))
                revision_hashes.append(os.path.splitext(parameterization_path)[0])
    return revision_hashes, parameterizations


def add_parameterization_to_archive(parameterization: dict, revision_hash: str, output_filepath: str):
    with ZipFile(output_filepath, "a") as zf:
        json_str = json.dumps(parameterization)
        zf.writestr("{}.json".format(revision_hash), json_str)


def load_group(data_filepath, group_name, return_list):
    with tables.open_file(data_filepath) as data_file:
        try:
            for batch in data_file.list_nodes(group_name, classname="Array"):
                return_list.append(torch.from_numpy(batch.read()))
        except tables.exceptions.NoSuchNodeError:
            return


def save_tensors_to_file(data_filepath: str, named_tensors: Dict[str, torch.Tensor], lockfile: FileLock = None):
    def _save():
        if not os.path.exists(data_filepath):
            print(f"{data_filepath} does not exist, creating...")
            with tables.open_file(data_filepath, "w", title="Tensor data") as output_file:
                for key in named_tensors.keys():
                    output_file.create_group("/", key)
        with tables.open_file(data_filepath, "a", title="Tensor data") as output_file:
            for key, tensor in named_tensors.items():
                nodes = output_file.list_nodes(f"/{key}")
                batch_ids = natsorted([node.name for node in nodes])
                batch_id = str(int(batch_ids[-1]) + 1) if len(batch_ids) > 0 else "0"
                output_file.create_array(f"/{key}", batch_id, tensor.numpy())

    if lockfile is not None:
        with lockfile.acquire():
            _save()
    else:
        _save()


def load_tensors_from_file(data_filepath: str, keep_batches=False, multiproc=True) -> Dict[str, torch.Tensor]:
    if multiproc:
        manager = multiprocessing.Manager()

        with tables.open_file(data_filepath) as data_file:
            nodes = data_file.list_nodes("/")
            data = {node._v_name: manager.list() for node in nodes}

        processes = [multiprocessing.Process(target=load_group, args=(data_filepath, f"/{node_name}", node_list)) for node_name, node_list in data.items()]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
    else:
        with tables.open_file(data_filepath) as data_file:
            nodes = data_file.list_nodes("/")
            data = {node._v_name: [] for node in nodes}
        for node_name, node_list in data.items():
            load_group(data_filepath, f"/{node_name}", node_list)

    if not keep_batches:
        # Flatten data
        for key in data.keys():
            data[key] = torch.cat(list(data[key]), dim=0).float()
    return data

def load_tensors_from_dir(dirpath: str, keep_batches=False) -> Dict[str, torch.Tensor]:
    dicts = []
    for filename in filter(lambda fn: fn.endswith(".h5"), os.listdir(dirpath)):
        dicts.append(load_tensors_from_file(os.path.join(dirpath, filename), keep_batches))
    dict_of_tensors = {key: torch.cat([dic[key] for dic in dicts], dim=0) for key in dicts[0]}
    return dict_of_tensors
