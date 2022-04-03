# dataloader for ScanNet / when training & testing F-Net MaGNet
import os
import random
import glob

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json


def _read_ExtM_from_txt(fpath_txt):
    ExtM = np.eye(4)
    with open(fpath_txt, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for ir, row in enumerate(ExtM):
        row_content = content[ir].split()
        row = np.asarray([float(x) for x in row_content])
        ExtM[ir, :] = row
    ExtM = np.linalg.inv(ExtM)  #cam2world -> world2cam
    return ExtM


def _read_IntM_from_txt(fpath_txt):
    IntM = np.eye(4)
    with open(fpath_txt, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for ir, row in enumerate(IntM):
        row_content = content[ir].split()
        row = np.asarray([float(x) for x in row_content])
        IntM[ir, :] = row
    return IntM


class ScannetLoader(object):
    def __init__(self, args, mode):
        self.t_samples = ScannetLoadPreprocess(args, mode)

        if mode == 'train':
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.t_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.t_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=self.train_sampler)

        else:
            self.data = DataLoader(self.t_samples, 1, shuffle=False, num_workers=1)


class ScannetLoadPreprocess(Dataset):
    def __init__(self, args, mode):
        self.args = args
        if mode == 'train':
            with open("./data_split/scannet_train.txt", 'r') as f:
                self.filenames = f.readlines()
                self.scans = 'scans'
        elif mode == 'rob_test':
            with open("./data_split/scannet_rob_test.txt", 'r') as f:
                self.filenames = f.readlines()
                self.scans = 'scans_test'
        elif mode == 'long_test':
            with open("./data_split/scannet_long_test.txt", 'r') as f:
                self.filenames = f.readlines()
                self.scans = 'scans_test'
        else:
            raise Exception('mode not recognized')

        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset_path = args.dataset_path

        # local window
        self.window_radius = args.MAGNET_window_radius
        self.n_views = args.MAGNET_num_source_views
        self.frame_interval = self.window_radius // (self.n_views // 2)
        self.img_idx_center = self.n_views // 2

        # window_idx_list
        self.window_idx_list = list(range(-self.n_views // 2, (self.n_views // 2) + 1))
        self.window_idx_list = [i * self.frame_interval for i in self.window_idx_list]

        # image resolution
        self.img_H = args.input_height  # 480
        self.img_W = args.input_width   # 640
        self.dpv_H = args.dpv_height    # 120
        self.dpv_W = args.dpv_width     # 160

        # ray array
        self.ray_array = self.get_ray_array()

        # dict containing the raw width and height of each scene
        with open('./data/scannet_raw_WH.json', 'r') as f:
            self.raw_WH_dict = json.load(f)

    def __len__(self):
        return len(self.filenames)

    # ray array used to back-project depth-map into camera-centered coordinates
    def get_ray_array(self):
        ray_array = np.ones((self.dpv_H, self.dpv_W, 3))
        x_range = np.arange(self.dpv_W)
        y_range = np.arange(self.dpv_H)
        x_range = np.concatenate([x_range.reshape(1, self.dpv_W)] * self.dpv_H, axis=0)
        y_range = np.concatenate([y_range.reshape(self.dpv_H, 1)] * self.dpv_W, axis=1)
        ray_array[:, :, 0] = x_range + 0.5
        ray_array[:, :, 1] = y_range + 0.5
        return ray_array

    # get camera intrinscs
    def get_cam_intrinsics(self, scene_dir, scene_name):
        intrins_path = scene_dir + '/intrinsic/intrinsic_color.txt'
        raw_W, raw_H = self.raw_WH_dict[scene_name]

        # original intrinsic matrix (4X4)
        IntM_ = _read_IntM_from_txt(intrins_path)

        # updated intrinsic matrix
        IntM = np.zeros((3, 3))
        IntM[2, 2] = 1.
        IntM[0, 0] = IntM_[0, 0] * (self.dpv_W / raw_W)
        IntM[1, 1] = IntM_[1, 1] * (self.dpv_H / raw_H)
        IntM[0, 2] = IntM_[0, 2] * (self.dpv_W / raw_W)
        IntM[1, 2] = IntM_[1, 2] * (self.dpv_H / raw_H)

        # pixel to ray array
        pixel_to_ray_array = np.copy(self.ray_array)
        pixel_to_ray_array[:, :, 0] = ((pixel_to_ray_array[:, :, 0] * (raw_W / self.dpv_W))
                                       - IntM_[0, 2]) / IntM_[0, 0]
        pixel_to_ray_array[:, :, 1] = ((pixel_to_ray_array[:, :, 1] * (raw_H / self.dpv_H))
                                       - IntM_[1, 2]) / IntM_[1, 1]

        pixel_to_ray_array_2D = np.reshape(np.transpose(pixel_to_ray_array, axes=[2, 0, 1]), [3, -1])   # (3, H*W)
        pixel_to_ray_array_2D = torch.from_numpy(pixel_to_ray_array_2D.astype(np.float32))

        cam_intrinsics = {
            'unit_ray_array_2D': pixel_to_ray_array_2D,
            'intM': torch.from_numpy(IntM.astype(np.float32)),
        }
        return cam_intrinsics

    def __getitem__(self, idx):
        scene_name, img_idx = self.filenames[idx].split(' ')
        img_idx = int(img_idx)
        scene_dir = self.dataset_path + '/{}/{}/'.format(self.scans, scene_name)

        # identify the neighbor views
        img_idx_list = []
        for i in self.window_idx_list:
            if os.path.exists(scene_dir + '/color/{}.jpg'.format(img_idx + i)):
                img_idx_list.append(img_idx + i)
            else:
                img_idx_list.append(img_idx - i - np.sign(i) * int(self.frame_interval * 0.5))

        # cam intrinsics
        cam_intrins = self.get_cam_intrinsics(scene_dir, scene_name)

        # color augmentation
        color_aug = False
        if 'train' in self.mode and self.args.data_augmentation_color:
            if random.random() > 0.5:
                color_aug = True
                aug_gamma = random.uniform(0.9, 1.1)
                aug_brightness = random.uniform(0.75, 1.25)
                aug_colors = np.random.uniform(0.9, 1.1, size=3)

        # data array
        data_array = []
        for i in range(self.n_views + 1):
            cur_idx = str(img_idx_list[i])
            img_path = scene_dir + '/color/{}.jpg'.format(cur_idx)
            dmap_path = scene_dir + '/depth/{}.png'.format(cur_idx)
            pose_path = scene_dir + '/pose/{}.txt'.format(cur_idx)

            # read img
            img = Image.open(img_path).convert("RGB").resize(size=(self.img_W, self.img_H), resample=Image.BILINEAR)
            img = np.array(img).astype(np.float32) / 255.0      # (H, W, 3)
            if color_aug:
                img = self.augment_image(img, aug_gamma, aug_brightness, aug_colors)
            img = torch.from_numpy(img).permute(2, 0, 1)        # (3, H, W)
            img = self.normalize(img)

            # read dmap (only for the ref img)
            if i == self.img_idx_center:
                gt_dmap = Image.open(dmap_path).resize(size=(self.img_W, self.img_H), resample=Image.NEAREST)
                gt_dmap = np.array(gt_dmap)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
                gt_dmap = gt_dmap / 1000.0
                gt_dmap = torch.from_numpy(gt_dmap).permute(2, 0, 1)  # (1, H, W)
            else:
                gt_dmap = 0.0

            # read pose
            extM = _read_ExtM_from_txt(pose_path)

            data_dict = {
                'img': img,
                'gt_dmap': gt_dmap,
                'extM': extM,
                'scene_name': scene_name,
                'img_idx': cur_idx
            }
            data_array.append(data_dict)

        return data_array, cam_intrins

    def augment_image(self, image, gamma, brightness, colors):
        # gamma augmentation
        image_aug = image ** gamma

        # brightness augmentation
        image_aug = image_aug * brightness

        # color augmentation
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
