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

from typing import List

def chunk(l, n):
    """
    Yield successive n-sized chunks from l.
    See https://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def dict_of_lists_to_list_of_dicts(dict_of_lists: dict) -> List[dict]:
    """
    https://www.geeksforgeeks.org/python-split-dictionary-of-lists-to-list-of-dictionaries/
    """
    return [dict(zip(dict_of_lists, i)) for i in zip(*dict_of_lists.values())]


def merge_list_of_dicts_of_lists(list_of_dicts_of_lists: List[dict]) -> List[list]:
    outer_list = []
    for dict_of_lists in list_of_dicts_of_lists:
        inner_list = []
        for key in dict_of_lists.keys():
            inner_list.extend(dict_of_lists[key])
        outer_list.append(inner_list)
    return outer_list
