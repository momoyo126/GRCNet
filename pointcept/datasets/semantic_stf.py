"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SemanticSTFDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        split2seq = dict(
            train=['train'],
            val=['val'],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 5)
        coord = scan[:, :3]
        strength = scan[:, 3].reshape([-1, 1])/255.

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled",
            1: 0,  # "car",
            2: 1,  # "bicycle",
            3: 2,  # "motorcycle",
            4: 3,  # "truck",
            5: 4,  # "other-vehicle",
            6: 5,  # "person",
            7: 6,  # "bicyclist",
            8: 7,  # "motorcyclist",
            9: 8,  # "road",
            10: 9,  # "parking",
            11: 10,  # "sidewalk",
            12: 11,  # "other-ground",
            13: 12,  # "building",
            14: 13,  # "fence",
            15: 14,  # "vegetation",
            16: 15,  # "trunk",
            17: 16,  # "terrain",
            18: 17,  # "pole",
            19: 18,  # "traffic-sign",
            20: ignore_index  # "invalid"
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {  # inverse of previous map
            ignore_index: ignore_index,  # "unlabeled", and others ignored
            0: 1,  # "car"
            1: 2,  # "bicycle"
            2: 3,  # "motorcycle"
            3: 4,  # "truck"
            4: 5,  # "other-vehicle"
            5: 6,  # "person"
            6: 7,  # "bicyclist"
            7: 8,  # "motorcyclist"
            8: 9,  # "road"
            9: 10,  # "parking"
            10: 11,  # "sidewalk"
            11: 12,  # "other-ground"
            12: 13,  # "building"
            13: 14,  # "fence"
            14: 15,  # "vegetation"
            15: 16,  # "trunk"
            16: 17,  # "terrain"
            17: 18,  # "pole"
            18: 19  # "traffic-sign"
        }
        return learning_map_inv
