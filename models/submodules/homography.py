# Differentiable Homography
import numpy as np
import torch
import torch.nn.functional as F
import math
from torch.distributions.normal import Normal


# for training FNET
def est_costvolume_F(d_center, ref_feat, nghbr_feat, R, t, is_valid, cam_intrins):
    # d_center:         (1, D, 1, 1)
    # ref_feat:         (N, F, H/4, W/4)
    # nghbr_feat:       (NxV, F, H/4, W/4)
    B, _, H, W = ref_feat.shape
    _, D, _, _ = d_center.shape
    n_views = int(nghbr_feat.shape[0] / B)
    device = ref_feat.device

    cost_volume = torch.zeros(B, D, H, W, device=device)

    for i_batch in range(B):
        IntM = cam_intrins['intM'][i_batch, :, :].to(device)                                # 3, 3
        Ray2D = cam_intrins['unit_ray_array_2D'][i_batch, :, :].to(device)                  # 3, h*w

        ref_feat_ = ref_feat[i_batch, ...].unsqueeze(0)                                     # 1, F, h, w
        ref_feat_ = ref_feat_.repeat(D, 1, 1, 1)                                            # D, F, h, w
        ref_mv_cost = torch.zeros(D, H, W, device=device)                                   # D, h, w

        for i_view in range(n_views):
            if is_valid[i_batch, i_view].item() == 1:
                term1_pix = IntM.matmul(t[i_batch, i_view, :]).reshape(3, 1)                # 3, 1
                term2_pix = IntM.matmul(R[i_batch, i_view, :, :]).matmul(Ray2D)             # 3, h*w

                # things to warp
                nghbr_feat_ = nghbr_feat[B * i_view + i_batch, ...].unsqueeze(0)            # 1, F, h, w
                nghbr_feat_ = nghbr_feat_.repeat(D, 1, 1, 1)                                # D, F, h, w

                # compute cost (D, H, W)
                cost = _compute_cost_F(ref_feat_, nghbr_feat_, d_center, term1_pix, term2_pix, device)

                ref_mv_cost = ref_mv_cost + cost

        cost_volume[i_batch, :, :, :] = ref_mv_cost

    cost_volume = cost_volume / float(n_views)
    cost_volume = F.softmax(cost_volume, dim=1)
    return cost_volume


def _compute_cost_F(ref_feat_, nghbr_feat_, d_center, term1_pix, term2_pix, device):
    D, _, H, W = ref_feat_.shape

    # pixel coordinates
    src_coords = torch.zeros(D, H, W, 2, device=device)                         # D, H, W, 2
    term2_pix = term2_pix.unsqueeze(0).repeat(D, 1, 1)                          # D, 3, H*W
    P_src_pix = term1_pix.unsqueeze(0) + term2_pix * d_center.reshape(D, 1, 1)  # (1, 3, 1) + (D, 3, H*W) * (D, 1, 1) = (D, 3, H*W)
    P_src_pix = P_src_pix / (P_src_pix[:, 2, :].unsqueeze(1) + 1e-10)           # (D, 3, H*W)

    # pixel coordinates - normalized
    src_coords[:, :, :, 0] = P_src_pix[:, 0, :].reshape(D, H, W)
    src_coords[:, :, :, 1] = P_src_pix[:, 1, :].reshape(D, H, W)
    v_center = ref_feat_.shape[2] / 2.
    u_center = ref_feat_.shape[3] / 2.
    src_coords[:, :, :, 0] = (src_coords[:, :, :, 0] - u_center) / u_center
    src_coords[:, :, :, 1] = (src_coords[:, :, :, 1] - v_center) / v_center
    src_coords[src_coords > 10.0] = 10.0
    src_coords[src_coords < -10.0] = -10.0
    
    # nghbr_feat warping (D, F, H, W)
    nghbr_feat_warped = F.grid_sample(nghbr_feat_, src_coords, mode='bilinear', padding_mode='zeros', align_corners=False)

    # feat cost
    feat_cost = torch.sum((ref_feat_ * nghbr_feat_warped), axis=1)  # (D, H/4, W/4)

    return feat_cost


# consistency weighting
def est_costvolume_CW(d_volume, ref_feat, nghbr_feat, ref_gmms, nghbr_gmms,
                      R, t, is_valid, cam_intrins, thres):
    B, D, H, W = d_volume.shape
    n_views = int(nghbr_feat.shape[0] / B)
    nghbr_mu, nghbr_sigma = torch.split(nghbr_gmms, 1, dim=1)  # BxV, 1, H, W
    device = ref_feat.device

    cost_volume = torch.zeros(B, D, H, W, device=device)

    for i_batch in range(B):
        IntM = cam_intrins['intM'][i_batch, :, :].to(device)                                                            # 3, 3
        Ray2D = cam_intrins['unit_ray_array_2D'][i_batch, :, :].to(device)                                              # 3, HxW

        ref_feat_ = ref_feat[i_batch, ...].unsqueeze(0)                                                                 # 1, F, H, W
        ref_feat_ = ref_feat_.repeat(D, 1, 1, 1)                                                                        # D, F, H, W
        ref_mv_cost = torch.zeros(D, H, W, device=device)                                                                   # D, H, W

        for i_view in range(n_views):
            if is_valid[i_batch, i_view].item() == 1:
                idm = torch.eye(3, device=device)
                term1_cam = idm.matmul(t[i_batch, i_view, :]).reshape(3, 1)                                             # 3, 1
                term2_cam = idm.matmul(R[i_batch, i_view, :, :]).matmul(Ray2D)                                          # 3, HxW
                term1_pix = IntM.matmul(t[i_batch, i_view, :]).reshape(3, 1)                                            # 3, 1
                term2_pix = IntM.matmul(R[i_batch, i_view, :, :]).matmul(Ray2D)                                         # 3, HxW

                # things to warp
                nghbr_feat_ = nghbr_feat[B * i_view + i_batch, ...].unsqueeze(0)                                        # D, F, H, W
                nghbr_feat_ = nghbr_feat_.repeat(D, 1, 1, 1)
                nghbr_mu_ = nghbr_mu[B * i_view + i_batch, ...].unsqueeze(0)                                            # D, 1, H, W
                nghbr_mu_ = nghbr_mu_.repeat(D, 1, 1, 1)
                nghbr_sigma_ = nghbr_sigma[B * i_view + i_batch, ...].unsqueeze(0)                                      # D, 1, H, W
                nghbr_sigma_ = nghbr_sigma_.repeat(D, 1, 1, 1)

                # compute cost (D, H, W)
                weighted_cost = _compute_cost_CW(ref_feat_, nghbr_feat_, nghbr_mu_, nghbr_sigma_, d_volume[i_batch, ...],
                                                term1_cam, term2_cam, term1_pix, term2_pix, device, thres)

                ref_mv_cost = ref_mv_cost + weighted_cost

        cost_volume[i_batch, :, :, :] = ref_mv_cost

    cost_volume = cost_volume / float(n_views)
    return cost_volume


def _compute_cost_CW(ref_feat_, nghbr_feat_, nghbr_mu_, nghbr_sigma_, d_volume,
                     term1_cam, term2_cam, term1_pix, term2_pix, device, thres):

    D, H, W = d_volume.shape

    # pixel coordinates
    src_coords = torch.zeros(D, H, W, 2, device=device)                                                                 # D, H, W, 2
    term2_pix = term2_pix.unsqueeze(0).repeat(D, 1, 1)                                                                  # D, 3, HxW
    P_src_pix = term1_pix.unsqueeze(0) + term2_pix * d_volume.reshape(D, 1, -1)                                         # (1,3,1) + (D,3,HxW) * (D,1,HxW) = (D,3,HxW)
    P_src_pix = P_src_pix / (P_src_pix[:, 2, :].unsqueeze(1) + 1e-10)                                                   # D, 3, HxW

    # depth coordinates
    term2_cam = term2_cam.unsqueeze(0).repeat(D, 1, 1)                                                                  # D, 3, HxW
    P_src_cam = term1_cam.unsqueeze(0) + term2_cam * d_volume.reshape(D, 1, -1)                                         # (1,3,1) + (D,3,HxW) * (D,1,HxW) = (D,3,HxW)
    depth_volume_warped = P_src_cam[:, 2, :].reshape(D, H, W)                                                           # D, H, W

    # pixel coordinates - normalized
    src_coords[:, :, :, 0] = P_src_pix[:, 0, :].reshape(D, H, W)
    src_coords[:, :, :, 1] = P_src_pix[:, 1, :].reshape(D, H, W)
    v_center = ref_feat_.shape[2] / 2.
    u_center = ref_feat_.shape[3] / 2.
    src_coords[:, :, :, 0] = (src_coords[:, :, :, 0] - u_center) / u_center
    src_coords[:, :, :, 1] = (src_coords[:, :, :, 1] - v_center) / v_center
    src_coords[src_coords > 10.0] = 10.0
    src_coords[src_coords < -10.0] = -10.0

    nghbr_feat_warped = F.grid_sample(nghbr_feat_, src_coords, mode='bilinear', padding_mode='zeros', align_corners=False)
    nghbr_mu_warped = F.grid_sample(nghbr_mu_, src_coords, mode='bilinear', padding_mode='zeros', align_corners=False)
    nghbr_sigma_warped = F.grid_sample(nghbr_sigma_, src_coords, mode='bilinear', padding_mode='zeros', align_corners=False)

    # feat cost
    feat_cost = torch.sum((ref_feat_ * nghbr_feat_warped), axis=1)  # (D, H, W)

    depth_diff = torch.abs(depth_volume_warped - nghbr_mu_warped[:, 0, :, :])
    binary_prob = (depth_diff < (nghbr_sigma_warped[:, 0, :, :] * thres)).double()
    weighted_cost = feat_cost * binary_prob

    return weighted_cost

