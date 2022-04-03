# dataloader for ScanNet / when training & testing DNet
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

        # Rob test set by DORN (CVPR 18)
        elif mode == 'rob_test':
            with open("./data_split/scannet_rob_test.txt", 'r') as f:
                self.filenames = f.readlines()
                self.scans = 'scans_test'
        
        # Test set by Long et al. (CVPR 21)
        elif mode == 'long_test':
            with open("./data_split/scannet_long_test.txt", 'r') as f:
                self.filenames = f.readlines()
                self.scans = 'scans_test'
        else:
            raise Exception('mode not recognized')

        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.data_augmentation_rotate_degree = 2.5
        self.dataset_path = args.dataset_path

        # img resolution
        self.img_H = args.input_height  # 480
        self.img_W = args.input_width   # 640
        self.crop_H = args.crop_height  # 416
        self.crop_W = args.crop_width   # 544

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        scene_name, img_idx = self.filenames[idx].split(' ')
        img_idx = int(img_idx)
        scene_dir = self.dataset_path + '/{}/{}/'.format(self.scans, scene_name)
        img_path = scene_dir + '/color/{}.jpg'.format(img_idx)
        depth_path = scene_dir + '/depth/{}.png'.format(img_idx)

        # read img and depth
        img = Image.open(img_path).convert("RGB").resize(size=(self.img_W, self.img_H), resample=Image.BILINEAR)
        depth_gt = Image.open(depth_path).resize(size=(self.img_W, self.img_H), resample=Image.NEAREST)

        if self.mode == 'train':
            # data augmentation - rotate
            if self.args.data_augmentation_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.data_augmentation_rotate_degree
                img = img.rotate(random_angle, resample=Image.BILINEAR)
                depth_gt = depth_gt.rotate(random_angle, resample=Image.NEAREST)

            # data augmentation - flip
            if self.args.data_augmentation_flip:
                if random.random() > 0.5:
                    img = TF.hflip(img)
                    depth_gt = TF.hflip(depth_gt)

            # img and depth to array
            img = np.array(img).astype(np.float32) / 255.0                      # (H, W, 3)
            depth_gt = np.array(depth_gt)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
            depth_gt = depth_gt / 1000.0

            # data augmentation - random crop
            if self.args.data_augmentation_crop:
                img, depth_gt = self.random_crop(img, depth_gt, self.crop_H, self.crop_W)

            # data augmentation - color
            if self.args.data_augmentation_color:
                if random.random() > 0.5:
                    img = self.augment_image(img)

        else:
            # img and depth to array
            img = np.array(img).astype(np.float32) / 255.0                      # (H, W, 3)
            depth_gt = np.array(depth_gt)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
            depth_gt = depth_gt / 1000.0

        # img and depth to tensor
        img = torch.from_numpy(img).permute(2, 0, 1)            # (3, H, W)
        img = self.normalize(img)
        depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1)  # (1, H, W)

        sample = {'img': img,
                  'depth': depth_gt,
                  'scene_name': scene_name,
                  'img_idx': str(img_idx)}

        return sample

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
