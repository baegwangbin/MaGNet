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


def train(model, args, device):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    should_write = ((not args.distributed) or args.rank == 0)
    if should_write:
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # dataloader
    if args.dataset_name == 'scannet':
        from data.dataloader_scannet import ScannetLoader
        train_loader = ScannetLoader(args, 'train').data
        test_loader = ScannetLoader(args, 'long_test').data
    elif args.dataset_name == 'kitti_eigen':
        from data.dataloader_kitti import KittiLoader
        train_loader = KittiLoader(args, 'eigen_train').data
        test_loader = KittiLoader(args, 'eigen_test').data
    elif args.dataset_name == 'kitti_official':
        from data.dataloader_kitti import KittiLoader
        train_loader = KittiLoader(args, 'official_train').data
        test_loader = KittiLoader(args, 'official_test').data
    else:
        raise Exception

    # define loss
    loss_fn = MagnetLoss(args)

    # optimizer
    m = model.module if args.multigpu else model
    optimizer = optim.AdamW(m.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    # lr scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, epochs=args.n_epochs,
                                              steps_per_epoch=len(train_loader))

    # cudnn setting
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    # start training
    total_iter = 0
    model.train()
    for epoch in range(args.n_epochs):
        if args.rank == 0:
            t_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train",
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=len(train_loader))
        else:
            t_loader = train_loader

        for data_array, cam_intrins in t_loader:
            optimizer.zero_grad()
            total_iter += args.batch_size_orig

            # data to device
            cur_batch_size = data_array[0]['img'].size()[0]
            ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(data_array, cur_batch_size)

            ref_img = ref_dat['img'].to(device)
            gt_dmap = ref_dat['gt_dmap'].to(device)
            gt_dmap[gt_dmap > args.max_depth] = 0.0
            gt_dmap_mask = gt_dmap > args.min_depth

            nghbr_imgs = [nghbr_dat['img'].to(device) for nghbr_dat in nghbr_dats]
            nghbr_imgs = torch.cat(nghbr_imgs, dim=0)
            nghbr_poses = nghbr_poses.to(device)

            # forward pass
            pred_list = model(ref_img, nghbr_imgs, nghbr_poses, is_valid, cam_intrins, mode='train')

            # compute & display loss
            loss = loss_fn(pred_list, gt_dmap, gt_dmap_mask)

            loss_ = float(loss.data.cpu().numpy())
            if args.rank == 0:
                t_loader.set_description(f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train. Loss: {'%.5f' % loss_}")
                t_loader.refresh()

            # back-propagate
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # train visualization
            if should_write and ((total_iter % args.visualize_every) < args.batch_size_orig):
                utils.visualize_MaG(args, ref_img, gt_dmap, gt_dmap_mask, pred_list, total_iter)

            # validation loop
            if should_write and ((total_iter % args.validate_every) < args.batch_size_orig):
                model.eval()
                metrics = validate(model, args, test_loader, device)
                utils.log_metrics(args.eval_acc_txt, metrics, 'total_iter: {}'.format(total_iter))

                target_path = args.exp_model_dir + '/checkpoint_iter_%010d.pt' % total_iter
                torch.save({"model": model.state_dict(),
                            "iter": total_iter}, target_path)
                model.train()

    if should_write:
        model.eval()
        metrics = validate(model, args, test_loader, device)
        utils.log_metrics(args.eval_acc_txt, metrics, 'total_iter: {}'.format(total_iter))

        target_path = args.exp_model_dir + '/checkpoint_iter_%010d.pt' % total_iter
        torch.save({"model": model.state_dict(),
                    "iter": total_iter}, target_path)

    return model


def validate(model, args, test_loader, device='cpu'):
    with torch.no_grad():
        metrics = utils.RunningAverageDict()

        for data_array, cam_intrins in tqdm(test_loader, desc=f"Loop: Validation",
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):

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


# main worker
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # define model
    model = MAGNET(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    train(model, args, device=args.gpu)


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
    parser.add_argument("--DNET_fix_encoder_weights", type=str, default='None', help='ImageNet_fix, AdaBins, AdaBins_fix')
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

    # loss function
    parser.add_argument('--loss_fn', default='gaussian', type=str)
    parser.add_argument('--loss_gamma', default=0.8, type=float)

    # training
    parser.add_argument('--n_epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--validate_every', default=5000, type=int, help='validation period')
    parser.add_argument('--visualize_every', default=1000, type=int, help='visualization period')
    parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
    parser.add_argument("--workers", default=4, type=int, help="Number of workers for data loading")

    # optimizer setup
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--lr', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final_div_factor', default=10000, type=float, help="final div factor for lr")

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

    # read arguments from txt file
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.num_threads = args.workers
    args.mode = 'train'

    # create experiment directory
    args.exp_dir = args.exp_dir + '/{}/'.format(args.exp_name)
    print(args.exp_dir)
    args.exp_model_dir = args.exp_dir + '/models/'    # store model checkpoints
    args.exp_test_dir = args.exp_dir + '/test/'       # store test images
    args.exp_vis_dir = args.exp_dir + '/vis/'         # store training images
    args.exp_log_dir = args.exp_dir + '/log/'         # store log
    utils.make_dir_from_list([args.exp_dir, args.exp_model_dir, args.exp_test_dir, args.exp_vis_dir, args.exp_log_dir])

    # log
    utils.save_args(args, args.exp_log_dir + '/params.txt')  # save experiment parameters
    args.eval_acc_txt = args.exp_log_dir + '/eval_acc.txt'

    # train
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(args.visible_gpus))
    
    args.world_size = 1
    args.rank = 0
    nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')
        port = np.random.randint(15000, 16000)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    args.batch_size_orig = args.batch_size

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
