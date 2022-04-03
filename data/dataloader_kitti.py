# dataloader for KITTI / when training & testing F-Net and MaGNet
import os
import random
import glob

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pykitti


class KittiLoader(object):
    def __init__(self, args, mode):
        self.t_samples = KittiLoadPreprocess(args, mode)

        if mode == 'eigen_train' or mode == 'official_train':
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


class KittiLoadPreprocess(Dataset):
    def __init__(self, args, mode):
        self.args = args
        if mode == 'eigen_train':
            with open("./data_split/kitti_eigen_train.txt", 'r') as f:
                self.filenames = f.readlines()
        elif mode == 'eigen_test':
            with open("./data_split/kitti_eigen_test.txt", 'r') as f:
                self.filenames = f.readlines()
        elif mode == 'official_train':
            with open("./data_split/kitti_official_train.txt", 'r') as f:
                self.filenames = f.readlines()
        elif mode == 'official_test':
            with open("./data_split/kitti_official_test.txt", 'r') as f:
                self.filenames = f.readlines()
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
        self.img_H = args.input_height  # 352
        self.img_W = args.input_width   # 1216
        self.dpv_H = args.dpv_height    # 88
        self.dpv_W = args.dpv_width     # 304

        # ray array
        self.ray_array = self.get_ray_array()

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
    def get_cam_intrinsics(self, p_data):
        raw_img_size = p_data.get_cam2(0).size
        raw_W = int(raw_img_size[0])
        raw_H = int(raw_img_size[1])

        top_margin = int(raw_H - 352)
        left_margin = int((raw_W - 1216) / 2)

        # original intrinsic matrix (4X4)
        IntM_ = p_data.calib.K_cam2

        # updated intrinsic matrix
        IntM = np.zeros((3, 3))
        IntM[2, 2] = 1.
        IntM[0, 0] = IntM_[0, 0] * (self.dpv_W / float(self.img_W))
        IntM[1, 1] = IntM_[1, 1] * (self.dpv_H / float(self.img_H))
        IntM[0, 2] = (IntM_[0, 2] - left_margin) * (self.dpv_W / float(self.img_W))
        IntM[1, 2] = (IntM_[1, 2] - top_margin) * (self.dpv_H / float(self.img_H))

        # pixel to ray array
        pixel_to_ray_array = np.copy(self.ray_array)
        pixel_to_ray_array[:, :, 0] = ((pixel_to_ray_array[:, :, 0] * (self.img_W / float(self.dpv_W)))
                                       - IntM_[0, 2] + left_margin) / IntM_[0, 0]
        pixel_to_ray_array[:, :, 1] = ((pixel_to_ray_array[:, :, 1] * (self.img_H / float(self.dpv_H)))
                                       - IntM_[1, 2] + top_margin) / IntM_[1, 1]

        pixel_to_ray_array_2D = np.reshape(np.transpose(pixel_to_ray_array, axes=[2, 0, 1]), [3, -1])
        pixel_to_ray_array_2D = torch.from_numpy(pixel_to_ray_array_2D.astype(np.float32))

        cam_intrinsics = {
            'unit_ray_array_2D': pixel_to_ray_array_2D,
            'intM': torch.from_numpy(IntM.astype(np.float32))
        }
        return cam_intrinsics

    def __getitem__(self, idx):
        date, drive, mode, img_idx = self.filenames[idx].split(' ')
        img_idx = int(img_idx)
        scene_name = '%s_drive_%s_sync' % (date, drive)

        # identify the neighbor views
        img_idx_list = [img_idx + i for i in self.window_idx_list]
        p_data = pykitti.raw(self.dataset_path + '/rawdata', date, drive, frames=img_idx_list)

        # cam intrinsics
        cam_intrins = self.get_cam_intrinsics(p_data)

        # color augmentation
        color_aug = False
        if 'train' in self.mode and self.args.data_augmentation_color:
            if random.random() > 0.5:
                color_aug = True
                aug_gamma = random.uniform(0.9, 1.1)
                aug_brightness = random.uniform(0.9, 1.1)
                aug_colors = np.random.uniform(0.9, 1.1, size=3)

        # data array
        data_array = []
        for i in range(self.n_views + 1):
            cur_idx = img_idx_list[i]

            # read img
            img_name = '%010d.png' % cur_idx
            img_path = self.dataset_path + '/rawdata/{}/{}/image_02/data/{}'.format(date, scene_name, img_name)
            img = Image.open(img_path).convert("RGB")

            # kitti benchmark crop
            height = img.height
            width = img.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            img = img.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # to tensor
            img = np.array(img).astype(np.float32) / 255.0      # (H, W, 3)
            if color_aug:
                img = self.augment_image(img, aug_gamma, aug_brightness, aug_colors)
            img = torch.from_numpy(img).permute(2, 0, 1)        # (3, H, W)
            img = self.normalize(img)

            # read dmap (only for the ref img)
            if i == self.img_idx_center:
                dmap_path = self.dataset_path + '/{}/{}/proj_depth/groundtruth/image_02/{}'.format(mode, scene_name,
                                                                                                   img_name)
                gt_dmap = Image.open(dmap_path).crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                gt_dmap = np.array(gt_dmap)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
                gt_dmap = gt_dmap / 256.0
                gt_dmap = torch.from_numpy(gt_dmap).permute(2, 0, 1)  # (1, H, W)
            else:
                gt_dmap = 0.0

            # read extM
            pose = p_data.oxts[i].T_w_imu
            M_imu2cam = p_data.calib.T_cam2_imu
            extM = np.matmul(M_imu2cam, np.linalg.inv(pose))

            data_dict = {
                'img': img,
                'gt_dmap': gt_dmap,
                'extM': extM,
                'scene_name': scene_name,
                'img_idx': str(img_idx),
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