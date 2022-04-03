import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

import utils.utils as utils
from utils.losses import MagnetLoss
from models.MAGNET import MAGNET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn.functional as F
import time
import random


def validate(model, args, test_loader, device, vis_dir=None):
    if args.dataset_name == 'nyu' or args.dataset_name == 'scannet':
        d_max, e_max = 5.0, 0.5
    else:
        d_max, e_max = 60.0, 3.0

    with torch.no_grad():
        metrics = utils.RunningAverageDict()

        for data_array, cam_intrins in tqdm(test_loader, desc=f"Loop: Validation"):

            cur_batch_size = data_array[0]['img'].size()[0]
            ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(data_array, cur_batch_size)

            ref_img = ref_dat['img'].to(device)
            gt_dmap = ref_dat['gt_dmap'].to(device)
            gt_dmap[gt_dmap > args.max_depth] = 0.0

            nghbr_imgs = [nghbr_dat['img'].to(device) for nghbr_dat in nghbr_dats]
            nghbr_imgs = torch.cat(nghbr_imgs, dim=0)
            nghbr_poses = nghbr_poses.to(device)

            # forward pass
            pred_list = model(ref_img, nghbr_imgs, nghbr_poses, is_valid, cam_intrins, mode='test')

            pred_dmap, pred_stdev = torch.split(pred_list[-1], 1, dim=1)        # (B, 1, H, W)

            gt_dmap = gt_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 1)
            pred_dmap = pred_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()    # (B, H, W, 1)
            pred_stdev = pred_stdev.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 1)

            gt_dmap = gt_dmap[0, :, :, 0]
            pred_dmap = pred_dmap[0, :, :, 0]
            pred_var = np.square(pred_stdev[0, :, :, 0])

            valid_mask = np.logical_and(gt_dmap > args.min_depth, gt_dmap < args.max_depth)
            if args.garg_crop or args.eigen_crop:
                assert args.dataset_name == 'kitti_eigen'
                gt_height, gt_width = gt_dmap.shape
                eval_mask = np.zeros(valid_mask.shape)
                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.eigen_crop:
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)

            # masking
            pred_dmap[pred_dmap < args.min_depth] = args.min_depth
            pred_dmap[pred_dmap > args.max_depth] = args.max_depth
            pred_dmap[np.isinf(pred_dmap)] = args.max_depth
            pred_dmap[np.isnan(pred_dmap)] = args.min_depth

            metrics.update(utils.compute_depth_errors(gt_dmap[valid_mask], pred_dmap[valid_mask], pred_var[valid_mask]))

        return metrics.get_value()


########################################################################################################################

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    # directory
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--exp_dir', required=True, type=str)
    parser.add_argument('--visible_gpus', required=True, type=str)

    # output
    parser.add_argument('--output_dim', default=2, type=int)
    parser.add_argument('--output_type', default='G', type=str)
    parser.add_argument('--downsample_ratio', default=4, type=int)

    # DNET architecture
    parser.add_argument('--DNET_architecture', type=str, default='DenseDepth_BN', help='{DenseDepth_BN, DenseDepth_GN}')
    parser.add_argument("--DNET_fix_encoder_weights", type=str, default='None', help='None or AdaBins_fix')
    parser.add_argument("--DNET_ckpt", required=True, type=str)

    # FNET architecture
    parser.add_argument('--FNET_architecture', type=str, default='PSM-Net')
    parser.add_argument('--FNET_feature_dim', type=int, default=64)
    parser.add_argument("--FNET_ckpt", required=True, type=str)

    # Multi-view matching hyper-parameters
    parser.add_argument('--MAGNET_sampling_range', type=int, default=3)
    parser.add_argument('--MAGNET_num_samples', type=int, default=5)
    parser.add_argument('--MAGNET_mvs_weighting', type=str, default='CW5')
    parser.add_argument('--MAGNET_num_train_iter', type=int, default=3)
    parser.add_argument('--MAGNET_num_test_iter', type=int, default=3)
    parser.add_argument('--MAGNET_window_radius', type=int, default=10)
    parser.add_argument('--MAGNET_num_source_views', type=int, default=4)

    # dataset
    parser.add_argument("--dataset_name", required=True, type=str, help="{kitti, scannet}")
    parser.add_argument("--dataset_path", required=True, type=str, help="path to the dataset")
    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--dpv_height', type=int, help='input height', default=120)
    parser.add_argument('--dpv_width', type=int, help='input width', default=160)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

    # dataset - crop
    parser.add_argument('--do_kb_crop', default=True, help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--eigen_crop', default=False, help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', default=False, help='if set, crops according to Garg  ECCV16', action='store_true')

    # dataset - augmentation
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")

    # todo - best iter
    parser.add_argument("--MAGNET_ckpt", default='', type=str)

    # read arguments from txt file
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.num_threads = 1
    args.mode = 'online_eval'

    # experiment directory
    args.exp_dir = args.exp_dir + '/{}/'.format(args.exp_name)
    print(args.exp_dir)
    args.exp_test_dir = args.exp_dir + '/test/'       # store test images
    args.exp_log_dir = args.exp_dir + '/log/'         # store log
    utils.make_dir_from_list([args.exp_dir, args.exp_test_dir, args.exp_log_dir])

    # test
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(args.visible_gpus))
    args.distributed = False
    device = torch.device('cuda:0')

    # define & load model
    model = MAGNET(args).to(device)
    print('loading checkpoint... {}'.format(args.MAGNET_ckpt))
    model = utils.load_checkpoint(args.MAGNET_ckpt, model)
    model.eval()
    print('loading checkpoint... / done')


    # define test_loader
    if args.dataset_name == 'scannet':
        from data.dataloader_scannet import ScannetLoader
        test_loader = ScannetLoader(args, 'long_test').data
    elif args.dataset_name == '7scenes':
        from data.dataloader_7scenes import SevenScenesLoader
        test_loader = SevenScenesLoader(args, 'test').data
    elif args.dataset_name == 'kitti_official':
        from data.dataloader_kitti import KittiLoader
        test_loader = KittiLoader(args, 'official_test').data
    elif args.dataset_name == 'kitti_eigen':
        from data.dataloader_kitti import KittiLoader
        test_loader = KittiLoader(args, 'eigen_test').data
    else:
        raise Exception('dataset should be one of \{scannet, 7scenes, kitti_official, kitti_eigen\}')

    # measure accuracy    
    metrics = validate(model, args, test_loader, device, vis_dir=None)
    first_line = 'dataset: %s / d_min: %s / d_max: %s / ckpt_path: %s' % (args.dataset_name, args.min_depth, args.max_depth, args.MAGNET_ckpt)
    utils.log_metrics(args.exp_log_dir + '/test_acc.txt', metrics, first_line)

