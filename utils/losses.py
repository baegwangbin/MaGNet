# loss functions
import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss for training D-Net
class DnetLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_type = args.loss_fn

    def forward(self, pred, gt_depth, gt_depth_mask):
        if self.loss_type == 'gaussian':
            gt_depth = gt_depth[gt_depth_mask]
            mu, var = torch.split(pred, 1, dim=1)  # (B, 1, H, W)
            mu = mu[gt_depth_mask]
            var = var[gt_depth_mask]
            var[var < 1e-10] = 1e-10

            nll = (torch.square(mu - gt_depth) / (2 * var)) + (0.5 * torch.log(var))
            return torch.mean(nll)
        else:
            raise Exception


# Loss for training MaGNet (D-Net and F-Net fixed)
class MagnetLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_type = args.loss_fn 
        self.gamma = args.loss_gamma

    def forward(self, pred_list, gt_depth, gt_depth_mask):
        if self.loss_type == 'gaussian':
            gt_depth = gt_depth[gt_depth_mask]
            n_predictions = len(pred_list)
            loss = 0.0
            for i in range(n_predictions):
                i_weight = self.gamma ** (n_predictions - i - 1)

                mu, sigma = torch.split(pred_list[i], 1, dim=1)
                mu = mu[gt_depth_mask]
                sigma = sigma[gt_depth_mask]
                var = torch.square(sigma)
                var[var < 1e-10] = 1e-10

                nll = (torch.square(mu - gt_depth) / (2 * var)) + (0.5 * torch.log(var))
                loss = loss + i_weight * torch.mean(nll)
            return loss
        else:
            raise Exception