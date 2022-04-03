import torch
import torch.nn as nn
import torch.nn.functional as F


# EfficientNet B5
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        basemodel_name = 'tf_efficientnet_b5_ap'
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


# Decoder block with batch norm
class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


# Decoder block with group norm + weight standardization
class UpSampleGN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleGN, self).__init__()
        self._net = nn.Sequential(Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU(),
                                  Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


# Conv2d with weight standardization
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Upsample depth via bilinear upsampling
def upsample_depth_via_bilinear(depth, up_mask, downsample_ratio):
    return F.interpolate(depth, scale_factor=downsample_ratio, mode='bilinear', align_corners=True)


# Upsample depth via learned upsampling
def upsample_depth_via_mask(depth, up_mask, downsample_ratio):
    # depth: low-resolution depth (B, 2, H, W)
    # up_mask: (B, 9*k*k, H, W)
    k = downsample_ratio

    N, o_dim, H, W = depth.shape
    up_mask = up_mask.view(N, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    up_depth = F.unfold(depth, [3, 3], padding=1)       # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
    up_depth = up_depth.view(N, o_dim, 9, 1, 1, H, W)   # (B, 2, 3*3, 1, 1, H, W)
    up_depth = torch.sum(up_mask * up_depth, dim=2)     # (B, 2, k, k, H, W)

    up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)       # (B, 2, H, k, W, k)
    return up_depth.reshape(N, o_dim, k*H, k*W)         # (B, 2, kH, kW)


# Decoder
class Decoder(nn.Module):
    def __init__(self, num_classes, downsample_ratio, learned_upsampling, BN, dnet):
        super(Decoder, self).__init__()
        features = 2048
        bottleneck_features = 2048
        self.downsample_ratio = downsample_ratio
        self.dnet = dnet

        if BN:
            print('using BatchNorm')
            UpSample = UpSampleBN
        else:
            print('using GroupNorm')
            UpSample = UpSampleGN

        # decoder architecture
        if self.downsample_ratio == 8:
            i_dim = features // 4
            h_dim = 128
            self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=0)
            self.up1 = UpSample(skip_input=features // 1 + 176, output_features=features // 2)
            self.up2 = UpSample(skip_input=features // 2 + 64, output_features=features // 4)

        elif self.downsample_ratio == 4:
            i_dim = features // 8
            h_dim = 128
            self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=0)
            self.up1 = UpSample(skip_input=features // 1 + 176, output_features=features // 2)
            self.up2 = UpSample(skip_input=features // 2 + 64, output_features=features // 4)
            self.up3 = UpSample(skip_input=features // 4 + 40, output_features=features // 8)

        elif self.downsample_ratio == 2:
            i_dim = features // 16
            h_dim = 128
            self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=0)
            self.up1 = UpSample(skip_input=features // 1 + 176, output_features=features // 2)
            self.up2 = UpSample(skip_input=features // 2 + 64, output_features=features // 4)
            self.up3 = UpSample(skip_input=features // 4 + 40, output_features=features // 8)
            self.up4 = UpSample(skip_input=features // 8 + 24, output_features=features // 16)

        else:
            raise Exception('downsample ratio invalid')

        # depth prediction 
        self.depth_head = nn.Sequential(
            nn.Conv2d(i_dim, h_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, num_classes, 1),
        )

        # upsampling
        if learned_upsampling:
            self.mask_head = nn.Sequential(
                nn.Conv2d(i_dim, h_dim, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
                nn.Conv2d(h_dim, 9 * self.downsample_ratio * self.downsample_ratio, 1)
            )
            self.upsample_depth = upsample_depth_via_mask
        else:
            self.mask_head = lambda a: None
            self.upsample_depth = upsample_depth_via_bilinear

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        if self.downsample_ratio == 8:
            x_d0 = self.conv2(x_block4)
            x_d1 = self.up1(x_d0, x_block3)
            x_feat = self.up2(x_d1, x_block2)
        elif self.downsample_ratio == 4:
            x_d0 = self.conv2(x_block4)
            x_d1 = self.up1(x_d0, x_block3)
            x_d2 = self.up2(x_d1, x_block2)
            x_feat = self.up3(x_d2, x_block1)
        elif self.downsample_ratio == 2:
            x_d0 = self.conv2(x_block4)
            x_d1 = self.up1(x_d0, x_block3)
            x_d2 = self.up2(x_d1, x_block2)
            x_d3 = self.up3(x_d2, x_block1)
            x_feat = self.up4(x_d3, x_block0)
        else:
            raise Exception('downsample ratio invalid')

        depth = self.depth_head(x_feat)

        if self.dnet:
            mask = self.mask_head(x_feat)
            up_depth = self.upsample_depth(depth, mask, self.downsample_ratio)
            return up_depth
        else:
            # if used as a part of MaGNet, do not upsample and also return the feature-map
            return depth, x_feat


# D-Net
class DenseDepth(nn.Module):
    def __init__(self, n_bins, downsample_ratio, learned_upsampling, BN=True, dnet=True):
        super(DenseDepth, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(n_bins, downsample_ratio, learned_upsampling, BN, dnet)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        return self.decoder.parameters()

