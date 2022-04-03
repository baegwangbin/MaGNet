import torch
import torch.nn as nn
import torch.nn.functional as F


# F-Net
class FNET(nn.Module):
    def __init__(self, args):
        super(FNET, self).__init__()
        self.args = args

        # Define model
        if args.FNET_architecture == 'PSM-Net':
            from models.submodules.F_psmnet import PSMNet
            self.f_net = PSMNet(feature_dim=args.FNET_feature_dim)
        else:
            raise Exception

    def forward(self, img):
        return self.f_net(img)
