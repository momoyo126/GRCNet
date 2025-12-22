import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset
from .utils import range_to_point


class CIC(nn.Module):

    def __init__(self, dim, kl=[0.001, 0.001,0.001,0.001]):
        super().__init__()
        self.kl = kl
        self.voxel_mu = nn.Linear(dim, dim)
        self.voxel_sigma = nn.Linear(dim, dim)
        self.range_mu = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=1)
        self.range_sigma = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=1)
        self.softplus = nn.Softplus()
        self.point_transforms = nn.Sequential(
            nn.Linear(dim + dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
        )

    def forward(self, voxel_tensor, range_feat, proj_x, proj_y):
        range_mu = self.range_mu(range_feat)
        range_sigma = self.softplus(self.range_sigma(range_feat)) + 1e-5  # [B,C,H,W]
        range_sample = torch.randn_like(range_mu, device=range_mu.device)
        if self.training:
            range_z = range_mu + range_sigma * range_sample
        else:
            range_z = range_mu

        voxel_mu = self.voxel_mu(voxel_tensor.F)
        voxel_sigma = self.softplus(self.voxel_sigma(voxel_tensor.F)) + 1e-5
        voxel_sample = torch.randn_like(voxel_mu, device=voxel_mu.device)
        if self.training:
            voxel_z = voxel_mu + voxel_sigma * voxel_sample
        else:
            voxel_z = voxel_mu
        # voxel_z = voxel_mu

        r2v_mu = range_to_point(range_mu, proj_x, proj_y, batch2offset(voxel_tensor.C[:, 3]))
        r2v_sigma = range_to_point(range_sigma, proj_x, proj_y, batch2offset(voxel_tensor.C[:, 3]))
        r2v_z = range_to_point(range_z, proj_x, proj_y, batch2offset(voxel_tensor.C[:, 3]))

        # print(r2v_sigma.mean(dim=0).cpu().numpy(),voxel_sigma.mean(dim=0).cpu().numpy())

        # 计算模态权重
        _range_sigma = torch.exp(1 / r2v_sigma.mean(dim=1, keepdim=True))
        _voxel_sigma = torch.exp(1 / voxel_sigma.mean(dim=1, keepdim=True))
        range_w = _range_sigma / (_range_sigma + _voxel_sigma)
        voxel_w = _voxel_sigma / (_range_sigma + _voxel_sigma)

        # 加权融合
        r2v4 = range_w * r2v_z
        x4F = voxel_w * voxel_z
        # x4F = voxel_w * voxel_tensor.F
        fusion_z = self.point_transforms(torch.cat([x4F, r2v4], dim=1))
        zero_mu = torch.zeros_like(x4F)
        one_sigma = torch.ones_like(x4F)

        if self.training:
            loss_kl_vn = self.kl_divergence_diag_gaussians(voxel_mu, voxel_sigma, zero_mu, one_sigma)
            loss_kl_rn = self.kl_divergence_diag_gaussians(r2v_mu, r2v_sigma, zero_mu, one_sigma)
            # print(loss_kl_vn.detach().cpu().numpy(), loss_kl_rn.detach().cpu().numpy())
    
            loss_kl_vr = -self.kl_divergence_diag_gaussians(voxel_mu, voxel_sigma, r2v_mu, r2v_sigma)
            if loss_kl_vr < -100:
                loss_kl_vr *= 0
            loss_kl_rv = -self.kl_divergence_diag_gaussians(r2v_mu, r2v_sigma, voxel_mu, voxel_sigma)
            if loss_kl_rv < -100:
                loss_kl_rv *= 0
            # print(loss_kl_vn.detach().cpu().numpy(), loss_kl_rn.detach().cpu().numpy(),
            #       loss_kl_rv.detach().cpu().numpy(), loss_kl_vr.detach().cpu().numpy(),
            #       '-------',voxel_mu.mean().detach().cpu().numpy(),voxel_sigma.mean().detach().cpu().numpy(),
            #       range_mu.mean().detach().cpu().numpy(),range_sigma.mean().detach().cpu().numpy())
    
            loss = self.kl[0] * loss_kl_vn + self.kl[1] * loss_kl_rn + self.kl[2] * loss_kl_vr + self.kl[3] * loss_kl_rv
            # loss = self.kl[0] * loss_kl_vn + self.kl[1] * loss_kl_rn
        else:
            loss = 0

        return fusion_z, range_z, range_mu, loss

    def kl_divergence_diag_gaussians(self, mu1, sigma1, mu2, sigma2):
        """
        计算每对多元高斯分布之间的KL散度，支持批量操作

        参数:
        mu1: 第一个高斯分布的均值，形状为 [B, C, H, W] (torch.Tensor) 或者 [N, C]
        sigma1: 第一个高斯分布的标准差，形状为 [B, C, H, W] (torch.Tensor) 或者 [N, C]
        mu2: 第二个高斯分布的均值，形状为 [B, C, H, W] (torch.Tensor) 或者 [N, C]
        sigma2: 第二个高斯分布的标准差，形状为 [B, C, H, W] (torch.Tensor) 或者 [N, C]

        返回值:
        KL散度，形状为[B, H, W] (torch.Tensor)  或者 [N,]
        """
        # 计算sigma的平方，即协方差矩阵的对角元素
        # print(sigma1)
        # print(sigma2)
        eps = 1e-5
        sigma1_squared = sigma1 ** 2 + eps
        sigma2_squared = sigma2 ** 2 + eps

        # 计算KL散度的各项
        term1 = sigma1_squared / sigma2_squared
        term2 = (mu2 - mu1) ** 2 / sigma2_squared
        term3 = torch.log(sigma2_squared) - torch.log(sigma1_squared)

        kl = 0.5 * (term1 + term2 - 1 + term3)
        # print(kl.shape)

        # 对维度 C 求和，得到形状为 [B, H, W] 的张量或者 [N,] 的张量
        kl_sum = kl.sum(dim=1)
        # print('1:',kl_sum.detach().cpu().numpy())
        kl_loss = kl_sum.mean()
        # print('2:',kl_loss.detach().cpu().numpy())
        return kl_loss