import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

from models.DNET import DNET
from models.FNET import FNET
import utils.utils as utils
import models.submodules.homography as homography


# upsample coarse depth map via learned upsampling
def upsample_depth_via_mask(depth, up_mask, k):
    # depth: low-resolution depth (B, 2, H, W)
    # up_mask: (B, 9*k*k, H, W)
    N, o_dim, H, W = depth.shape
    up_mask = up_mask.view(N, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    up_depth = F.unfold(depth, [3, 3], padding=1)       # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
    up_depth = up_depth.view(N, o_dim, 9, 1, 1, H, W)   # (B, 2, 3*3, 1, 1, H, W)
    up_depth = torch.sum(up_mask * up_depth, dim=2)     # (B, 2, k, k, H, W)

    up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)       # (B, 2, H, k, W, k)
    return up_depth.reshape(N, o_dim, k*H, k*W)         # (B, 2, kH, kW)


# load checkpoint
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v
    model.load_state_dict(load_dict)
    return model


# GNET
class GNET(nn.Module):
    def __init__(self, ch_in, ch_out=2):
        super(GNET, self).__init__()
        h_dim = 128
        self.gnet = nn.Sequential(
            nn.Conv2d(ch_in, h_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, ch_out, 1)
        )

    def forward(self, cost_volume, ref_gmm):
        # ref_gmm: initial prediction (N, 2, H, W)
        mu_0, sigma_0 = torch.split(ref_gmm, 1, dim=1)

        # mu_1:     (mu_new - mu_0) / sigma_0
        # sigma_1:  sigma_new / sigma_0
        d_output = self.gnet(cost_volume)
        mu_1, sigma_1 = torch.split(d_output, 1, dim=1)

        mu_new = mu_0 + (mu_1 * sigma_0)
        sigma_new = (F.elu(sigma_1) + 1.0 + 1e-10) * sigma_0
        mv_gmm = torch.cat([mu_new, sigma_new], dim=1)                        # B, 3 x N_c, H, W
        return mv_gmm


class MAGNET(nn.Module):
    def __init__(self, args):
        super(MAGNET, self).__init__()
        self.args = args

        # load DNET
        print('loading DNET...{}'.format(args.DNET_ckpt))
        self.d_net = DNET(args, dnet=False)
        self.d_net = load_checkpoint(args.DNET_ckpt, self.d_net)
        for param in self.d_net.parameters():
            param.requires_grad = False
        self.d_net.eval()

        # load FNET
        print('loading FNET... {}'.format(args.FNET_ckpt))
        self.f_net = FNET(args)
        self.f_net = load_checkpoint(args.FNET_ckpt, self.f_net)
        for param in self.f_net.parameters():
            param.requires_grad = False
        self.f_net.eval()

        # hyperparameters
        self.sampling_range = args.MAGNET_sampling_range        # beta in paper / defines the sampling range
        self.n_samples = args.MAGNET_num_samples                # N_s in paper / number of samples
        self.weighting = args.MAGNET_mvs_weighting              # If it is "CW5", it means "use consistency weighting and set kappa to 5"
        self.train_iter = args.MAGNET_num_train_iter            # N_iter during training
        self.test_iter = args.MAGNET_num_test_iter              # N_iter during testing
        self.dpv_height = args.dpv_height                       # height of the cost volume (1/4 of the original height)
        self.dpv_width = args.dpv_width                         # width of the cost volume (1/4 of the original width)

        self.k_list = self.depth_sampling()
        self.downsample_ratio = args.downsample_ratio

        # GNet
        dnet_fdim = 256
        self.g_net = GNET(ch_in=dnet_fdim + self.n_samples, ch_out=2)

        # Learned upsampling
        h_dim = 128
        self.mask_head = nn.Sequential(
            nn.Conv2d(dnet_fdim, h_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, 9 * self.downsample_ratio * self.downsample_ratio, 1)
        )
        self.upsample_depth = upsample_depth_via_mask

    def depth_sampling(self):
        from scipy.special import erf
        from scipy.stats import norm
        P_total = erf(self.sampling_range / np.sqrt(2))             # Probability covered by the sampling range
        idx_list = np.arange(0, self.n_samples + 1)
        p_list = (1 - P_total)/2 + ((idx_list/self.n_samples) * P_total)
        k_list = norm.ppf(p_list)
        k_list = (k_list[1:] + k_list[:-1])/2
        return list(k_list)

    def forward(self, ref_img, nghbr_imgs, nghbr_poses, is_valid, cam_intrins, mode='train'):
        B = ref_img.shape[0]

        with torch.no_grad():
            # D-Net forward pass
            mono_gmms, x_d3 = self.d_net(torch.cat((ref_img, nghbr_imgs), dim=0))     # N+NxV x 2 x H/4 x W/4
            mono_gmms = mono_gmms.detach()
            ref_gmms = mono_gmms[:B, ...]
            x_d3 = x_d3[:B, ...]
            nghbr_gmms = mono_gmms[B:, ...]

            # F-Net forward pass
            feat_4 = self.f_net(torch.cat((ref_img, nghbr_imgs), dim=0))
            ref_feat_4 = feat_4[:B, ...]
            nghbr_feat_4 = feat_4[B:, ...]

        # Multi-view matching
        Rs_src = nghbr_poses[:, :, :3, :3]                                 # N, V, 3, 3
        ts_src = nghbr_poses[:, :, :3, 3]                                  # N, V, 3

        pred_list = [ref_gmms]
        for itr in range(self.train_iter) if mode == 'train' else range(self.test_iter):

            # Depth sampling
            ref_mu, ref_sigma = torch.split(pred_list[-1].detach(), 1, dim=1)   # B, 1, H, W
            depth_volume = [ref_mu + ref_sigma * k for k in self.k_list]
            depth_volume = torch.cat(depth_volume, dim=1)                       # B, N_samples, H, W

            # Multi-view matching
            thres = int(self.weighting.split('CW')[1])
            cost_volume = homography.est_costvolume_CW(
                depth_volume, ref_feat_4, nghbr_feat_4,
                ref_gmms, nghbr_gmms,
                Rs_src, ts_src, is_valid, cam_intrins, thres
            )

            # G-Net forward pass
            gnet_input = torch.cat([cost_volume.detach(), x_d3], dim=1)
            new_pred = self.g_net(gnet_input, pred_list[-1].detach())
            pred_list.append(new_pred)

        # Upsampling
        mask = self.mask_head(x_d3)
        pred_list = [self.upsample_depth(pred, mask, self.downsample_ratio) for pred in pred_list[1:]]

        return pred_list


# When training F-Net
class MAGNET_F(nn.Module):
    def __init__(self, args):
        super(MAGNET_F, self).__init__()
        self.f_net = FNET(args)

    def forward(self, ref_img, nghbr_imgs, nghbr_poses, is_valid, cam_intrins, d_center):
        B = ref_img.shape[0]

        # F-Net forward pass
        feat_4 = self.f_net(torch.cat((ref_img, nghbr_imgs), dim=0))
        ref_feat_4 = feat_4[:B, ...]                                       # N   x F x H/4 x W/4
        nghbr_feat_4 = feat_4[B:, ...]                                     # NxV x F x H/4 x W/4

        # DP-Net forward pass
        Rs_src = nghbr_poses[:, :, :3, :3]                                 # N, V, 3, 3
        ts_src = nghbr_poses[:, :, :3, 3]                                  # N, V, 3

        # Cost-volume computation
        cost_volume = homography.est_costvolume_F(
            d_center, ref_feat_4, nghbr_feat_4,
            Rs_src, ts_src, is_valid, cam_intrins
        )

        return cost_volume


