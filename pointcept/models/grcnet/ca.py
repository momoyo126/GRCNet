import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse
import torchsparse.nn as spnn
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset
from torchsparse import SparseTensor, PointTensor
from .utils import range_to_point, point_to_voxel, voxel_to_point, point_to_voxel_segment, range_to_voxel_pxpy
from .MultiHeadAttention import MultiHeadAttention
import os


class CA(nn.Module):

    def __init__(self, dim, N, num_heads=8):
        super().__init__()
        self.init_prototypes(N, dim)

        self.attn_w_q_1 = nn.Linear(dim, dim)
        self.attn_w_k_1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=1)
        self.attn_w_v_1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=1)
        self.attn_multihead_attn_1 = MultiHeadAttention(dim, num_heads)

        self.attn_w_q_2 = nn.Linear(dim, dim)
        self.attn_w_k_2 = nn.Linear(dim, dim)
        self.attn_w_v_2 = nn.Linear(dim, dim)
        self.attn_multihead_attn_2 = MultiHeadAttention(dim, num_heads)

        self.attn_point_transforms = nn.Sequential(
            nn.Linear(dim + dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )


    def forward(self, voxel_tensor, range_z, range_mu, fusion_z, down4c):
        # print(voxel_tensor.F.shape,range_z.shape,range_mu.shape,fusion_z.shape,down4c.shape)
        b, c, h, w = down4c.shape
        # query: prototypes [n, C] -> [B,n,C]
        # key: down4c [B,C,H,W] -> [B,HW,C]
        # value: down4c [B,C,H,W] -> [B,HW,C]
        q_1 = self.attn_w_q_1(self.attn_prototypes).unsqueeze(0).repeat(b, 1, 1)  # [B,n,C]
        if not self.training:
            range_z = range_mu
        k_1 = self.attn_w_k_1(range_z).view(b, c, -1).transpose(1, 2)  # [B,HW,C]
        v_1 = self.attn_w_v_1(range_z).view(b, c, -1).transpose(1, 2)
        attn_output_1 = self.attn_multihead_attn_1(q_1, k_1, v_1)  # [B,n,C]
        attn_z = attn_output_1

        offset = batch2offset(voxel_tensor.C[:, 3]).detach()
        feat_padded, mask = self.feat_padding(voxel_tensor.F, offset)
        q_2 = self.attn_w_q_2(feat_padded.detach())  # [B,Length,C]
        k_2 = self.attn_w_k_2(attn_z)  # [B,n,C]
        v_2 = self.attn_w_v_2(attn_z)
        # attn_output_2 = self.attn_multihead_attn_2(q_2,k_2,v_2)
        attn_output_2 = self.attn_multihead_attn_2(q_2, k_2, v_2, mask) + feat_padded.detach()  # [B,N',C]

        assert not torch.any(torch.isnan(attn_output_2)), 'attn_output2'
        v4 = self.feat_unpadding(attn_output_2, offset)
        new_fusion_z = self.attn_point_transforms(torch.cat([fusion_z, v4], dim=1))
        return new_fusion_z

    def init_prototypes(self, N, dim):
        self.attn_prototypes = nn.Parameter(torch.empty(N, dim))
        nn.init.xavier_normal_(self.attn_prototypes)
        print(self.attn_prototypes)

    def feat_padding(self, feat, offset):
        # feat: [N,C]
        # offset: [B,]
        B = offset.shape[0]
        C = feat.shape[-1]
        offset_ = torch.cat([torch.tensor([0]).to(offset.device), offset])
        max_length = (offset_[1:] - offset_[:-1]).max().item()
        feats_padded = torch.zeros((B, max_length, C), device=feat.device)
        mask = torch.zeros((B, max_length), dtype=torch.bool, device=feat.device)
        for i in range(B):
            start = offset_[i]
            end = offset_[i + 1]
            length = end - start
            feats_padded[i, :length] = feat[start:end]
            mask[i, :length] = 1

        return feats_padded, mask

    def feat_unpadding(self, feat, offset):
        feats_unpadded = []
        B = offset.shape[0]
        offset_ = torch.cat([torch.tensor([0]).to(offset.device), offset])
        for i in range(B):
            length = offset_[i + 1] - offset_[i]
            feats_unpadded.append(feat[i, :length])
        return torch.cat(feats_unpadded, dim=0)