import argparse
import json
import os
from typing import Any, Dict, List

from PIL import Image

from . import base_dataset


def create_dataset(transform: Any, opt: argparse.Namespace) -> base_dataset.BaseDataset:
    return TreeGravityShadow2ContourDataset(transform, opt.max_dataset_size, opt.img_dir, opt.filename_json_path)


def modify_dataset_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(transform_name='crop_and_random_flip')
    parser.add_argument('--filename_json_path', type=str, default=os.path.join('../dataloaders', 'inputs', 'tree_gravity_shadow2contour', 'filename_train.json'))
    parser.add_argument('--img_dir', type=str, default=os.path.join('../dataloaders', 'inputs', 'tree_gravity_shadow2contour', 'IMAGE_PAIRS_273x193px'))
    return parser


class TreeGravityShadow2ContourDataset(base_dataset.BaseDataset):
    def __init__(self,
                 transform: Any,
                 max_dataset_size: int,
                 img_dir: str,
                 filename_json_path: str) -> None:
        self.file_list: List[Dict[str, str]] = get_file_list(
            filename_json_path=filename_json_path,
            img_dir=img_dir,
        )
        super().__init__(max_dataset_size, len(self.file_list))
        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path_dict = self.file_list[idx]

        from_path = path_dict['A']
        to_path = path_dict['B']

        from_img = Image.open(from_path).convert('L')
        to_img = Image.open(to_path)

        from_tensor, to_tensor = self.transform(from_img, to_img)

        return {'a': from_tensor, 'b': to_tensor, 'a_paths': from_path, 'b_paths': to_path}


def get_file_list(filename_json_path: str, img_dir: str) -> List[Dict[str, str]]:
    with open(filename_json_path, 'r') as f:
        filename_dict = json.load(f)

    ret: List[Dict[str, str]] = []
    for value in filename_dict.values():
        path_a = os.path.join(img_dir, value[0])
        path_b = os.path.join(img_dir, value[1])
        if os.path.isfile(path_a) and os.path.isfile(path_b):
            ret.append({
                'A': path_a,
                'B': path_b,
            })
        else:
            print(f'ファイル欠損:{path_a}, {path_b}')

    return ret
