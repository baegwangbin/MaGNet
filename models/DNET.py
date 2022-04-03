import torch
import torch.nn as nn
import torch.nn.functional as F


# D-Net
class DNET(nn.Module):
    def __init__(self, args, dnet=True):
        super(DNET, self).__init__()
        self.args = args

        # Define activation
        if args.output_type == 'R':
            self.activation = self.activation_none
        elif args.output_type == 'G':
            if dnet:
                self.activation = self.activation_G
            else:
                self.activation = self.activation_G_magnet
        else:
            raise Exception

        # Define model
        if 'DenseDepth' in args.DNET_architecture:
            # DenseDepth_BN
            _, BN = args.DNET_architecture.split('_')
            from models.submodules.D_dense_depth import DenseDepth
            self.d_net = DenseDepth(n_bins=args.output_dim, 
                                    downsample_ratio=args.downsample_ratio, 
                                    learned_upsampling=True,
                                    BN=BN=='BN',
                                    dnet=dnet)
        else:
            raise Exception

        # load and fix weights
        if args.DNET_fix_encoder_weights == "AdaBins_fix":
            model_path = "./ckpts/AdaBins_kitti_encoder.pt"
            print('loading AdaBins weights... {}'.format(model_path))

            pre_model_dict = torch.load(model_path)['model']
            model_dict = self.d_net.encoder.state_dict()
            pre_model_dict_feat = {k: v for k, v in pre_model_dict.items() if k in model_dict}
            model_dict.update(pre_model_dict_feat)

            self.d_net.encoder.load_state_dict(pre_model_dict_feat)
            for param in self.d_net.encoder.parameters():
                param.requires_grad = False

    def forward(self, img, **kwargs):
        return self.activation(self.d_net(img, **kwargs))

    def activation_none(self, out):
        return out

    def activation_G(self, out):
        mu, var = torch.split(out, 1, dim=1)
        var = F.elu(var) + 1.0 + 1e-10
        out = torch.cat([mu, var], dim=1)  # (N, 2, H, W)
        return out

    def activation_G_magnet(self, outs):
        mu, var = torch.split(outs[0], 1, dim=1)
        var = F.elu(var) + 1.0 + 1e-10
        stdev = torch.sqrt(var)
        out = torch.cat([mu, stdev], dim=1)  # (N, 2, H, W)
        return out, outs[1]
