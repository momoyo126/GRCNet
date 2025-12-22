#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import time

import numpy as np
import math
import random

import torch
from scipy.spatial.transform import Rotation as R
from .transform import TRANSFORMS
import os
import matplotlib.pyplot as plt


@TRANSFORMS.register_module()
class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""

    def __init__(self, H=64, W=1024, fov_up=3.0, fov_down=-25.0, DA=False, flip_sign=False, rot=False,
                 drop_points=False):
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = DA
        self.flip_sign = flip_sign
        self.rot = rot
        self.drop_points = drop_points

        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def do_range_projection(self, data_dict):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        self.points = data_dict['coord']
        self.remissions = data_dict['strength'].reshape(-1)

        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1) + 1e-4
        # print("points min:", np.min(self.points, axis=0),"points max:", np.max(self.points, axis=0),'depth max min:',np.max(depth),np.min(depth))

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.indices = indices
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)
        data_dict['proj_range'] = self.proj_range
        # data_dict['proj_range'] = self.fill_empty_points_with_max(self.proj_range)
        data_dict['proj_xyz'] = self.proj_xyz
        data_dict['proj_remission'] = self.proj_remission
        data_dict['proj_idx'] = self.proj_idx
        data_dict['proj_mask'] = self.proj_mask
        data_dict['unproj_range'] = self.unproj_range
        data_dict['proj_x'] = self.proj_x
        data_dict['proj_y'] = self.proj_y
        data_dict['order'] = np.copy(order)
        return data_dict

    def __call__(self, data_dict):
        data_dict = self.do_range_projection(data_dict)
        return data_dict

    def save(self, name, image):
        path0 = name
        figsize = (2048 / 100, 64 / 100)  # 将尺寸从像素转换为英寸，这里假设 100 DPI
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap='inferno', aspect='auto', vmin=0, vmax=1)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path0, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()


@TRANSFORMS.register_module()
class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""

    def __init__(self, sensor, mode='train', max_classes=300):
        H = sensor["img_prop"]["height"]
        W = sensor["img_prop"]["width"]
        fov_up = sensor.fov_up
        fov_down = sensor.fov_down
        self.mode = mode
        self.sensor_img_means = np.array(sensor.img_means)
        self.sensor_img_stds = np.array(sensor.img_stds)
        super(SemLaserScan, self).__init__(H, W, fov_up, fov_down)
        self.reset()

        # make semantic colors
        sem_color_dict = self.get_color_map()
        if sem_color_dict:
            # if I have a dict, make it
            max_sem_key = 0
            for key, data in sem_color_dict.items():
                if key + 1 > max_sem_key:
                    max_sem_key = key + 1
            self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
            for key, value in sem_color_dict.items():
                self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
        else:
            # otherwise make random
            max_sem_key = max_classes
            self.sem_color_lut = np.random.uniform(low=0.0,
                                                   high=1.0,
                                                   size=(max_sem_key, 3))
            # force zero to a gray-ish color
            self.sem_color_lut[0] = np.full((3), 0.1)

        # make instance colors
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)

    def reset(self):
        """ Reset scan members. """
        super(SemLaserScan, self).reset()

        # semantic labels
        self.sem_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # instance labels
        # self.inst_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        # self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # projection color with semantic labels
        self.proj_sem_label = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.int32)  # [H,W]  label
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                       dtype=float)  # [H,W,3] color

        # projection color with instance labels
        # self.proj_inst_label = np.full((self.proj_H, self.proj_W),-1,
        #                                 dtype=np.int32)  # [H,W]  label
        # self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
        #                                 dtype=float)  # [H,W,3] color

    def colorize(self):
        """ Colorize pointcloud with the color of each semantic label
        """
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

        self.inst_label_color = self.inst_color_lut[self.inst_label]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    def do_label_projection(self, data_dict):
        # only map colors to labels that exist
        self.sem_label = data_dict['segment']
        mask = self.proj_idx >= 0

        # semantics
        # self.proj_sem_label[self.proj_y,self.proj_x] = self.sem_label[self.indices]
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        # self.proj_sem_color[self.proj_y,self.proj_x] = self.sem_color_lut[self.sem_label[self.indices]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]
        data_dict['proj_sem_label'] = self.proj_sem_label
        data_dict['proj_sem_color'] = self.proj_sem_color

        # instances
        # self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
        # self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
        return data_dict

    def get_color_map(self):
        return None

    def __call__(self, data_dict):
        self.reset()
        data_dict = self.do_range_projection(data_dict)
        proj_mask = data_dict['proj_mask']

        proj = np.concatenate((data_dict['proj_range'][np.newaxis, :, :], data_dict['proj_xyz'].transpose(2, 0, 1),
                               data_dict['proj_remission'][np.newaxis, :, :]))
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.astype(float)
        data_dict['proj'] = proj[np.newaxis, :, :, :]
        data_dict['proj_remission'] = proj[np.newaxis, 4:, :, :]
        data_dict['proj_range_remission'] = proj[np.newaxis, [0, 4], :, :]
        data_dict['origin_offset'] = np.array([data_dict['proj_y'].shape[0]])

        return data_dict
