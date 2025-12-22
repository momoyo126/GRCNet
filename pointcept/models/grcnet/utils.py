import torch
import torchsparse.nn.functional as F
from torch.nn.functional import grid_sample



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

    new_px = torch.ones(x.C.shape[0], device=x.F.device)
    new_py = torch.ones(x.C.shape[0], device=x.F.device)
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