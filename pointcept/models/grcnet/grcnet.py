import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse
import torchsparse.nn as spnn

from torchsparse import SparseTensor, PointTensor
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset
from .cic import CIC
from .ca import CA
from .utils import computer_pxpy, range_to_voxel_pxpy_test


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class InvertedResidual(nn.Module):
    def __init__(self, in_filters, out_filters, stride, expand_ratio=6, norm='batchnorm'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(in_filters * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_filters == out_filters
        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d
        elif norm == 'instancenorm':
            self.norm = nn.InstanceNorm2d

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_filters, 1, 1, 0, bias=False),
                self.norm(out_filters),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_filters, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_filters, 1, 1, 0, bias=False),
                self.norm(out_filters),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, bottleneck=2, pooling=True, drop_out=True,
                 kernel_size=(3, 3), norm='batchnorm', expand_ratio=6):
        super(InvResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        hidden_dim = int((in_filters + out_filters) / 2)
        if bottleneck == 2:
            self.invresBlock1 = nn.Sequential(
                InvertedResidual(in_filters, hidden_dim, 1, expand_ratio=expand_ratio, norm=norm),
                InvertedResidual(hidden_dim, out_filters, 1, expand_ratio=expand_ratio, norm=norm))
        elif bottleneck == 1:
            self.invresBlock1 = InvertedResidual(in_filters, out_filters, 1, expand_ratio=expand_ratio, norm=norm)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        resA = self.invresBlock1(x)
        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


@MODELS.register_module("GRCNet")
class GRCNet(nn.Module):

    def __init__(self, nclasses=19, norm='instancenorm', invresbottleneck=2, cr=0.5, kl=[0.001, 0.001, 0.0005, 0.0005], num_proto=32):
        super().__init__()

        self.nclasses = nclasses
        self.in_channels = 4

        # minknet/voxel
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(self.in_channels, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.voxel_stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.voxel_stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1))

        self.voxel_stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.voxel_stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        # rangeview channel:[2,32,64,128,256,256]
        self.downCntx = ResContextBlock(2, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.range_resBlock1 = InvResBlock(32, 2 * 32, 0.2, bottleneck=invresbottleneck, pooling=True, drop_out=False,
                                           norm=norm)
        self.range_resBlock2 = InvResBlock(2 * 32, 2 * 2 * 32, 0.2, bottleneck=invresbottleneck, pooling=True,
                                           norm=norm)
        self.range_resBlock3 = InvResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, bottleneck=invresbottleneck, pooling=True,
                                           norm=norm)
        self.range_resBlock4 = InvResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, bottleneck=invresbottleneck, pooling=False,
                                           norm=norm)
        self.range_transform = nn.Conv2d(256, cs[4], kernel_size=(1, 1), stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(nn.Linear(cs[8], self.nclasses))

        self.CIC = CIC(cs[4], kl=kl)
        self.CA = CA(cs[4], num_proto)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        feat[:, 3] = 0  # ignore the strength
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        rangeview = input_dict['proj_range_remission']
        px, py = computer_pxpy(feat[:,:3].clone().detach(), 2048)
        
        coord = torch.cat(
            [grid_coord.int(), batch.unsqueeze(-1).int()], dim=1
        ).contiguous()
        z0 = PointTensor(None, coord)

        # rangeview encoder
        downCntx = self.downCntx(rangeview)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.range_resBlock1(downCntx)
        down1c, down1b = self.range_resBlock2(down0c)
        down2c, down2b = self.range_resBlock3(down1c)
        down4c = self.range_resBlock4(down2c)
        down4c = self.range_transform(down4c)
        down4c = self.relu(down4c)

        # voxel encoder
        x = SparseTensor(feat, coord)
        x0 = self.stem(x)
        x1 = self.voxel_stage1(x0)
        x2 = self.voxel_stage2(x1)
        x3 = self.voxel_stage3(x2)
        x4 = self.voxel_stage4(x3)

        px, py = range_to_voxel_pxpy_test(x4,z0,px, py)

        fusion_z, range_z, range_mu, loss = self.CIC(x4, down4c, px, py)
        fusion_z = self.CA(x4, range_z, range_mu, fusion_z, down4c)

        x4.F = fusion_z

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = self.classifier(y4.F)

        return out, loss