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
from models.DNET import DNET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def validate(model, args, test_loader, device='cpu', vis_dir=None):
    if args.dataset_name == 'nyu' or args.dataset_name == 'scannet':
        d_max, e_max = 5.0, 0.5
    else:
        d_max, e_max = 60.0, 3.0

    with torch.no_grad():
        metrics = utils.RunningAverageDict()

        for t_data_dict in tqdm(test_loader, desc="Loop: Validation"):

            # data to device
            img = t_data_dict['img'].to(device)
            gt_dmap = t_data_dict['depth'].to(device)

            # forward pass
            out = model(img)

            pred_dmap, pred_var = torch.split(out, 1, dim=1)  # (B, 1, H, W)

            gt_dmap = gt_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()    # (B, H, W, 1)
            pred_dmap = pred_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()      # (B, H, W, 1)
            pred_var = pred_var.detach().cpu().permute(0, 2, 3, 1).numpy()      # (B, H, W, 1)

            gt_dmap = gt_dmap[0, :, :, 0]
            pred_dmap = pred_dmap[0, :, :, 0]
            pred_var = pred_var[0, :, :, 0]
            pred_stdev = np.sqrt(pred_var)

            valid_mask = np.logical_and(gt_dmap > args.min_depth, gt_dmap < args.max_depth)
            if args.garg_crop or args.eigen_crop:
                assert args.dataset_name == 'kitti_eigen'
                gt_height, gt_width = gt_dmap.shape
                eval_mask = np.zeros(valid_mask.shape)
                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.eigen_crop:
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)
            invalid_mask = np.logical_not(valid_mask)

            # masking
            pred_dmap[pred_dmap < args.min_depth] = args.min_depth
            pred_dmap[pred_dmap > args.max_depth] = args.max_depth
            pred_dmap[np.isinf(pred_dmap)] = args.max_depth
            pred_dmap[np.isnan(pred_dmap)] = args.min_depth

            metrics.update(utils.compute_depth_errors(gt_dmap[valid_mask], pred_dmap[valid_mask], pred_var[valid_mask]))

        return metrics.get_value()


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    # directory
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--exp_dir', required=True, type=str)
    parser.add_argument('--visible_gpus', required=True, type=str)

    # output
    parser.add_argument('--output_dim', required=True, type=int, help='{1, 2}')
    parser.add_argument('--output_type', required=True, type=str, help='{R, G}')
    parser.add_argument('--downsample_ratio', type=int, default=4)

    # DNET - model architecture
    parser.add_argument('--DNET_architecture', required=True, type=str, help='{DenseDepth_BN, DenseDepth_GN}')
    parser.add_argument("--DNET_fix_encoder_weights", type=str, default='None', help='None or AdaBins_fix')

    # dataset
    parser.add_argument("--dataset_name", required=True, type=str, help="{kitti_eigen, kitti_official, scannet, 7scenes}")
    parser.add_argument("--dataset_path", required=True, type=str, help="path to the dataset")
    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--crop_height', type=int, help='input height', default=416)
    parser.add_argument('--crop_width', type=int, help='input width', default=544)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

    # dataset - crop
    parser.add_argument('--do_kb_crop', default=True, help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--eigen_crop', default=False, help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', default=False, help='if set, crops according to Garg ECCV16', action='store_true')

    # dataset - augmentation
    parser.add_argument("--data_augmentation_flip", default=True, action="store_true")
    parser.add_argument("--data_augmentation_crop", default=True, action="store_true")
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")
    parser.add_argument("--data_augmentation_rotate", default=True, action="store_true")

    # ckpt path
    parser.add_argument("--ckpt_path", required=True, type=str)

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
    model = DNET(args).to(device)
    print('loading checkpoint... {}'.format(args.ckpt_path))
    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()
    print('loading checkpoint... / done')

    # define test_loader
    if args.dataset_name == 'scannet':
        from data.dataloader_scannet_D import ScannetLoader
        test_loader = ScannetLoader(args, 'long_test').data
    elif args.dataset_name == '7scenes':
        from data.dataloader_7scenes_D import SevenScenesLoader
        test_loader = SevenScenesLoader(args, 'test').data
    elif args.dataset_name == 'kitti_official':
        from data.dataloader_kitti_D import KittiLoader
        test_loader = KittiLoader(args, 'official_test').data
    elif args.dataset_name == 'kitti_eigen':
        from data.dataloader_kitti_D import KittiLoader
        test_loader = KittiLoader(args, 'eigen_test').data
    else:
        raise Exception('dataset should be one of \{scannet, 7scenes, kitti_official, kitti_eigen\}')

    # test accuracy
    metrics = validate(model, args, test_loader, device, vis_dir=None)
    first_line = 'dataset: %s / d_min: %s / d_max: %s / ckpt_path: %s' % (args.dataset_name, args.min_depth, args.max_depth, args.ckpt_path)
    utils.log_metrics(args.exp_log_dir + '/test_acc.txt', metrics, first_line)
