# dataloader for 7-Scenes / when testing D-Net
import os
import random
import glob

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


class SevenScenesLoader(object):
    def __init__(self, args, mode):
        self.t_samples = SevenScenesLoadPreprocess(args, mode)
        self.data = DataLoader(self.t_samples, 1, shuffle=False, num_workers=1)


class SevenScenesLoadPreprocess(Dataset):
    def __init__(self, args, mode):
        self.args = args

        # Test set by Long et al. (CVPR 21)
        with open("./data_split/sevenscenes_long_test.txt", 'r') as f:
            self.filenames = f.readlines()

        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset_path = args.dataset_path

        # img resolution
        self.img_H = args.input_height  # 480
        self.img_W = args.input_width   # 640

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        scene_name, seq_id, img_idx = self.filenames[idx].split(' ')
        seq_id = int(seq_id)
        img_idx = int(img_idx)

        scene_dir = self.dataset_path + '/{}/seq-%02d/'.format(scene_name) % seq_id

        # img path and depth path
        img_path = scene_dir + '/frame-%06d.color.png' % img_idx
        depth_path = scene_dir + '/frame-%06d.depth.png' % img_idx

        # read img and depth
        img = Image.open(img_path).convert("RGB").resize(size=(self.img_W, self.img_H), resample=Image.BILINEAR)
        depth_gt = Image.open(depth_path).resize(size=(self.img_W, self.img_H), resample=Image.NEAREST)

        # img to tensor
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        img = self.normalize(img)

        depth_gt = np.array(depth_gt)[:, :, np.newaxis]
        depth_gt[depth_gt == 65535] = 0.0                       # filter out invalid depth
        depth_gt = depth_gt.astype(np.float32) / 1000.0         # from mm to m
        depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1)  # (1, H, W)

        sample = {'img': img,
                  'depth': depth_gt,
                  'scene_name': '%s_seq-%02d' % (scene_name, seq_id),
                  'img_idx': str(img_idx)}

        return sample
