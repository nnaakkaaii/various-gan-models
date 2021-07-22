import argparse
import math
import os
import re
from collections import defaultdict
from glob import glob
from statistics import mean
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch.utils.data as data

from . import base_dataset


def create_dataset(transform: Any, opt: argparse.Namespace) -> data.Dataset:
    return FilletShadow2ContourDataset(
        transform=transform, max_dataset_size=opt.max_dataset_size, z_min=opt.z_min, z_max=opt.z_max, solver_output_dir=opt.solver_output_dir)


def modify_dataset_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(transform_name='vanilla_numpy2tensor')
    parser.add_argument('--solver_output_dir', type=str, default=os.path.join('..', 'inputs', 'solver_output_0525'))
    parser.add_argument('--z_min', type=float, default=0)
    parser.add_argument('--z_max', type=float, default=5)
    return parser


class FilletShadow2ContourDataset(base_dataset.BaseDataset):
    def __init__(self,
                 transform: Any,
                 max_dataset_size: int,
                 z_min: float,
                 z_max: float,
                 solver_output_dir: str) -> None:
        self.file_list = get_file_list(solver_output_dir=solver_output_dir)
        super().__init__(max_dataset_size, len(self.file_list))
        self.transform = transform
        self.z_min = z_min
        self.z_max = z_max

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.file_list[idx]

        frd = Frd(path)
        colormap = ColorMap(
            x=frd.get_x(),
            s=frd.get_s(),
            z_min=self.z_min,
            z_max=self.z_max,
        )

        to_array = colormap.get_array()
        from_array = 0.299 * to_array[:, :, 2] + 0.587 * to_array[:, :, 1] + 0.114 * to_array[:, :, 0]

        from_tensor, to_tensor = self.transform(from_array, to_array)

        return {'A': from_tensor, 'B': to_tensor}


def get_file_list(solver_output_dir: str) -> List[str]:
    return sorted(glob(os.path.join(solver_output_dir, '**', 'FEMMeshNetgen.frd'), recursive=True))


pattern = r'\-?\d\.\d{5}E[\+|\-]\d{2}'


def remove_double_space(text: str) -> List[str]:
    text = text.strip()
    return [s for s in text.split(' ') if s != '']


class Frd:
    def __init__(self, path: str) -> None:
        """
        :param path:
        :raises IndexError: ファイルの欠損
        :raises ValueError: パースの失敗
        """
        with open(path, 'r') as f:
            self.__text = f.readlines()
        # 11
        x_index = 11
        # 3326
        x_num = int(remove_double_space(self.__text[x_index])[1])
        # 12, 3338
        self.__x_range = (1 + x_index, 1 + x_index + x_num)
        # 3339
        b_index = 2 + x_index + x_num
        # 3880
        b_num = 2 * int(remove_double_space(self.__text[b_index])[1])
        # 3340, 7220
        self.__b_range = (1 + b_index, 1 + b_index + b_num)
        # 7222
        d_index = 3 + b_index + b_num
        # 3326
        d_num = int(remove_double_space(self.__text[d_index])[3])
        # 7228, 10554
        self.__d_range = (6 + d_index, 6 + d_index + d_num)
        # 10556
        s_index = 8 + d_index + d_num
        # 3326
        s_num = int(remove_double_space(self.__text[s_index])[3])
        # 10564, 13890
        self.__s_range = (8 + s_index, 8 + s_index + s_num)
        # 13892
        e_index = 10 + s_index + s_num
        # 3326
        e_num = int(remove_double_space(self.__text[s_index])[3])
        # 13900, 17226
        self.__e_range = (8 + e_index, 8 + e_index + e_num)

    def get_x(self) -> Dict[int, List[float]]:
        lines = self.__text[self.__x_range[0]: self.__x_range[1]]
        x = {i: list(map(float, re.findall(pattern, line))) for i, line in enumerate(lines)}
        return x

    def get_d(self) -> Dict[int, List[float]]:
        lines = self.__text[self.__d_range[0]: self.__d_range[1]]
        d = {i: list(map(float, re.findall(pattern, line))) for i, line in enumerate(lines)}
        return d

    def get_s(self) -> Dict[int, List[float]]:
        lines = self.__text[self.__s_range[0]: self.__s_range[1]]
        s = {i: list(map(float, re.findall(pattern, line))) for i, line in enumerate(lines)}
        return s

    def get_e(self) -> Dict[int, List[float]]:
        lines = self.__text[self.__e_range[0]: self.__e_range[1]]
        e = {i: list(map(float, re.findall(pattern, line))) for i, line in enumerate(lines)}
        return e


class ColorMap:
    def __init__(self, x: Dict[int, List[float]], s: Dict[int, List[float]], z_min: float = 0, z_max: float = 5) -> None:
        use_keys = [k for k, v in x.items() if z_min <= v[2] <= z_max]
        self.map = {(x[k][0], x[k][1]): math.sqrt(s[k][0] ** 2 + s[k][1] ** 2) for k in use_keys}
        self.x_min = min(m[0] for m in self.map.keys())
        self.x_max = max(m[0] for m in self.map.keys())
        self.y_min = min(m[1] for m in self.map.keys())
        self.y_max = max(m[1] for m in self.map.keys())

    def get_array(self, width: int = 128, height: int = 128) -> np.ndarray:
        array = self._embed(width, height)
        array = self._complement(array)
        self._fillna(array)
        return array

    def _embed(self, width: int, height: int) -> np.ndarray:
        color_map = defaultdict(list)
        for (x, y), v in self.map.items():
            color_map[(
                int((width - 1) * (x - self.x_min) / (self.x_max - self.x_min)),
                int((height - 1) * (y - self.y_min) / (self.y_max - self.y_min))
            )].append(v)
        array = np.array([[np.nan for _ in range(width)] for _ in range(height)])
        for x, y in color_map.keys():
            array[x, y] = mean(color_map[x, y])
        return array

    @staticmethod
    def _complement(array: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(array)
        df = df.interpolate(
            method='linear',
            limit_area='inside',
        )
        return df.values

    @staticmethod
    def _fillna(array: np.ndarray) -> None:
        array[np.isnan(array)] = 0
        return
