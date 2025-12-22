import torch
import torchsparse.nn.functional as F
# import range_utils.nn.functional as rnf
import time
import numpy as numpy
from torch_scatter import scatter_max
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
from torch.nn.functional import grid_sample, one_hot

__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point', 'range_to_point']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1,
    )

    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(
        torch.floor(new_float_coord),
        idx_query,
        counts,
    )
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)

    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
        x.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=x.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C)
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


def resample_grid_stacked(range_feat, px, py, offsets, grid_sample_mode='bilinear'):
    '''
    :param range_feat: BCHW
    :param px,py: [N,2]
    :param offsets: the list of offset
    :return:
    '''
    r2p = []
    for batch, offset in enumerate(offsets):
        if batch == 0:
            p_x = px[0: offsets[batch]]
            p_y = py[0: offsets[batch]]
        else:
            p_x = px[offsets[batch - 1]: offsets[batch]]
            p_y = py[offsets[batch - 1]: offsets[batch]]
        pxpy = torch.stack([p_x, p_y], dim=1).to(px[0].device).unsqueeze(0).unsqueeze(0)
        resampled = grid_sample(range_feat[batch].unsqueeze(0), pxpy, mode=grid_sample_mode, align_corners=True)
        one_resampled = resampled.squeeze().transpose(0, 1)  # NxC
        r2p.append(one_resampled)
    return torch.cat(r2p, dim=0)


def range_to_point(feature_map, px, py, offsets, grid_sample_mode='bilinear'):
    """convert 2d range feature map to points feature"""
    return resample_grid_stacked(feature_map, px, py, offsets, grid_sample_mode)


def range_to_voxel_pxpy(x, z, px, py):
    pc_hash = F.sphash(
        torch.cat([
            torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
            z.C[:, -1].int().view(-1, 1)
        ], 1))
    sparse_hash = F.sphash(x.C)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    # print(idx_query.shape, idx_query.max(),x.C.shape[0])
    sorted_hash, sorted_indices = torch.sort(idx_query)

    _, count = torch.unique(sorted_hash, return_counts=True)
    cumsum_count = torch.cumsum(count, dim=0)
    # print(count.shape,counts.shape,cumsum_count.max(),z.C.shape)
    cumsum_count_shifted = torch.cat(
        (torch.tensor([0], dtype=cumsum_count.dtype, device=x.F.device), cumsum_count[:-1]))
    random_ints = torch.randint(0, count.max().item(), (count.size(0),)).to(x.F.device)
    idx_select = cumsum_count_shifted + random_ints % count
    # print(idx_select.max())
    idx_unique = sorted_indices[idx_select]
    new_px = px[idx_unique]
    new_py = py[idx_unique]

    return new_px, new_py


def range_to_voxel_pxpy_test(x, z, px, py):
    pc_hash = F.sphash(
        torch.cat([
            torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
            z.C[:, -1].int().view(-1, 1)
        ], 1))
    sparse_hash = F.sphash(x.C)
    idx_query = F.sphashquery(pc_hash, sparse_hash)

    sorted_hash, sorted_indices = torch.sort(idx_query)

    counts = F.spcount(sorted_hash.int(), x.C.shape[0])

    cumsum_count = torch.cumsum(counts, dim=0)
    cumsum_count_shifted = torch.cat(
        (torch.tensor([0], dtype=cumsum_count.dtype, device=x.F.device), cumsum_count[:-1]))
    random_ints = torch.randint(0, counts.max().item(), (counts.size(0),)).to(x.F.device)
    idx_select = cumsum_count_shifted + random_ints % counts

    valid_indices = counts > 0
    idx_select = idx_select[valid_indices]
    idx_unique = sorted_indices[idx_select]

    # 初始化 new_px 和 new_py
    new_px = torch.ones(x.C.shape[0], device=x.F.device)  # 设置为 1
    new_py = torch.ones(x.C.shape[0], device=x.F.device)  # 设置为 1
    new_px[valid_indices] = px[idx_unique]
    new_py[valid_indices] = py[idx_unique]

    return new_px, new_py


def computer_pxpy(coord, w=512):
    fov_up = torch.tensor((3.0 / 180.0) * torch.pi)  # field of view up in rad
    fov_down = torch.tensor((-25.0 / 180.0) * torch.pi)  # field of view down in rad
    fov = torch.abs(fov_down) + torch.abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = torch.norm(coord, p=2, dim=1) + 1e-4

    # get scan components
    scan_x = coord[:, 0]
    scan_y = coord[:, 1]
    scan_z = coord[:, 2]

    # get angles of all points
    yaw = -torch.atan2(scan_y, scan_x)
    pitch = torch.asin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / torch.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + torch.abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_W = w
    proj_H = 64
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = torch.floor(proj_x)
    proj_x = torch.clamp(proj_x, 0, proj_W - 1).long()  # in [0,W-1]

    proj_y = torch.floor(proj_y)
    proj_y = torch.clamp(proj_y, 0, proj_H - 1).long()  # in [0,H-1]

    proj_x = 2.0 * (proj_x / (proj_W - 1) - 0.5)
    proj_y = 2.0 * (proj_y / (proj_H - 1) - 0.5)
    return proj_x, proj_y



def point_to_voxel_segment(x, z, segment, mode='random'):
    pc_hash = F.sphash(
        torch.cat([
            torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
            z.C[:, -1].int().view(-1, 1)
        ], 1))
    sparse_hash = F.sphash(x.C)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    sorted_hash, sorted_indices = torch.sort(idx_query)
    _, count = torch.unique(sorted_hash, return_counts=True)
    if mode == 'random':
        cumsum_count = torch.cumsum(count, dim=0)
        cumsum_count_shifted = torch.cat(
            (torch.tensor([0], dtype=cumsum_count.dtype, device=x.F.device), cumsum_count[:-1]))
        random_ints = torch.randint(0, count.max().item(), (count.size(0),)).to(x.F.device)
        idx_select = cumsum_count_shifted + random_ints % count
        idx_unique = sorted_indices[idx_select]
        new_segment = segment[idx_unique]
    elif mode == 'most':
        num_voxels = count.size(0)
        num_classes = segment.max() + 1 + 1

        voxel_class_count = torch.zeros((num_voxels, num_classes), device=segment.device)
        voxel_indices = torch.arange(num_voxels, device=segment.device).repeat_interleave(count)
        voxel_class_count.scatter_add_(0, voxel_indices.view(-1, 1).expand(-1, num_classes),
                                       one_hot(segment[sorted_indices] + 1, num_classes).float())
        new_segment = voxel_class_count.argmax(dim=1) - 1
    return new_segment


def voxel_to_voxel_coord(x, z):
    pc_hash = F.sphash(
        torch.cat([
            torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
            z.C[:, -1].int().view(-1, 1)
        ], 1))
    sparse_hash = F.sphash(x.C)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    sorted_hash, sorted_indices = torch.sort(idx_query)
    _, count = torch.unique(sorted_hash, return_counts=True)
    cumsum_count = torch.cumsum(count, dim=0)
    cumsum_count_shifted = torch.cat(
        (torch.tensor([0], dtype=cumsum_count.dtype, device=x.F.device), cumsum_count[:-1]))
    random_ints = torch.randint(0, count.max().item(), (count.size(0),)).to(x.F.device)
    idx_select = cumsum_count_shifted + random_ints % count
    idx_unique = sorted_indices[idx_select]
    new_coord = z.F[:, :3][idx_unique]
    return new_coord


def voxel_to_voxel_idx(x, z):
    pc_hash = F.sphash(
        torch.cat([
            torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
            z.C[:, -1].int().view(-1, 1)
        ], 1))
    sparse_hash = F.sphash(x.C)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    sorted_hash, sorted_indices = torch.sort(idx_query)
    _, count = torch.unique(sorted_hash, return_counts=True)
    cumsum_count = torch.cumsum(count, dim=0)
    cumsum_count_shifted = torch.cat(
        (torch.tensor([0], dtype=cumsum_count.dtype, device=x.F.device), cumsum_count[:-1]))
    random_ints = torch.randint(0, count.max().item(), (count.size(0),)).to(x.F.device)
    idx_select = cumsum_count_shifted + random_ints % count
    idx_unique = sorted_indices[idx_select]
    return idx_unique


def voxel_to_bev(x):
    feat = x.F
    xyb_coord = x.C[:, [0, 1, 3]]
    unique_coords, inverse_indices = torch.unique(xyb_coord, dim=0, return_inverse=True)
    max_feat, _ = scatter_max(feat, inverse_indices, dim=0)
    z_column = torch.zeros((unique_coords.shape[0], 1), dtype=unique_coords.dtype, device=unique_coords.device)
    coords = torch.cat((unique_coords[:, :2], z_column, unique_coords[:, 2:]), dim=1)
    _, indices = torch.sort(coords[:, 3], dim=0)
    new_tensor = SparseTensor(max_feat[indices], coords[indices])
    return new_tensor


def bev_to_voxel(x, bev):
    voxel_coord = x.C
    z_column = torch.zeros((voxel_coord.shape[0], 1), dtype=voxel_coord.dtype, device=voxel_coord.device)
    coords = torch.cat((voxel_coord[:, :2], z_column, voxel_coord[:, 3:]), dim=1)
    voxel_hash = F.sphash(coords)
    bev_hash = F.sphash(bev.C)
    idx_query = F.sphashquery(voxel_hash, bev_hash)
    bev_to_voxel_feat = bev.F[idx_query]
    return bev_to_voxel_feat