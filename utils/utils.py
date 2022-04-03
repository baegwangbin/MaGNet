# utils
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#########
# Utils #
#########


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


def save_args(args, filename):
    with open(filename, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))


def write_to_log(txt_filename, msg):
    with open(txt_filename, 'a') as f:
        f.write('{}\n'.format(msg))


def makedir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def make_dir_from_list(dirpath_list):
    for dirpath in dirpath_list:
        makedir(dirpath)


def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v
    model.load_state_dict(load_dict)
    return model


######################
# Data preprocessing #
######################


def split_data_array(data_array):
    n_frames = len(data_array)
    ref_idx = n_frames // 2
    ref_dat = data_array[ref_idx]
    nghbr_dats = [data_array[idx] for idx in range(n_frames) if idx != ref_idx]
    return ref_dat, nghbr_dats


def data_preprocess(data_array, cur_batch_size):
    # 1. Split data array
    ref_dat, nghbr_dats = split_data_array(data_array)
    num_views = len(nghbr_dats)

    # 2. Obtain pose
    nghbr_poses = torch.zeros((cur_batch_size, num_views, 4, 4))
    is_valid = torch.ones((cur_batch_size, num_views), dtype=torch.int)
    ref_extM = ref_dat['extM']  # batch_size X 4 X 4
    nghbr_extMs = [nghbr_dat['extM'] for nghbr_dat in nghbr_dats]  # list of (batch_size X 4 X 4)
    for i in range(cur_batch_size):
        ext_ref = ref_extM[i, :, :]
        if torch.isnan(ext_ref.min()):
            is_valid[i, :] = 0
        else:
            for j in range(num_views):
                ext_nghbr = nghbr_extMs[j][i, :, :]
                if torch.isnan(ext_nghbr.min()):
                    is_valid[i, j] = 0
                else:
                    nghbr_pose = ext_nghbr.mm(torch.from_numpy(np.linalg.inv(ext_ref)))
                    if torch.isnan(nghbr_pose.min()):
                        is_valid[i, j] = 0
                    else:
                        nghbr_poses[i, j, :, :] = nghbr_pose

    return ref_dat, nghbr_dats, nghbr_poses, is_valid


##############
# Evaluation #
##############


def compute_depth_errors(gt, pred, var=None):
    thresh = np.maximum((gt / pred), (pred / gt))

    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_diff = np.mean(np.abs(gt - pred))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    irmse = (1/gt - 1/pred) ** 2
    irmse = np.sqrt(irmse.mean())

    if var is not None:
        var[var < 1e-6] = 1e-6
        nll = 0.5 * (np.log(var) + np.log(2*np.pi) + (np.square(gt - pred) / var))
        nll = np.mean(nll)
    else:
        nll = 0.0

    return dict(a1=a1, a2=a2, a3=a3,
                abs_diff=abs_diff,
                abs_rel=abs_rel, sq_rel=sq_rel,
                rmse=rmse, log_10=log_10, irmse=irmse,
                rmse_log=rmse_log, silog=silog,
                nll=nll)


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def log_metrics(txt_path, metrics, first_line):
    print('{}'.format(first_line))
    print("abs_rel abs_diff sq_rel rmse rmse_log irmse log_10 silog a1 a2 a3 NLL")
    print("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
        metrics['abs_rel'], metrics['abs_diff'],
        metrics['sq_rel'], metrics['rmse'],
        metrics['rmse_log'], metrics['irmse'],
        metrics['log_10'], metrics['silog'],
        metrics['a1'], metrics['a2'], metrics['a3'],
        metrics['nll']))

    with open(txt_path, 'a') as f:
        f.write('{}\n'.format(first_line))
        f.write("abs_rel abs_diff sq_rel rmse rmse_log irmse log_10 silog a1 a2 a3 NLL\n")
        f.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n\n" % (
            metrics['abs_rel'], metrics['abs_diff'],
            metrics['sq_rel'], metrics['rmse'],
            metrics['rmse_log'], metrics['irmse'],
            metrics['log_10'], metrics['silog'],
            metrics['a1'], metrics['a2'], metrics['a3'],
            metrics['nll']))


#################
# Visualization #
#################


__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
def unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out


# visualize during training (DNET)
def visualize_D(args, img, gt_dmap, gt_dmap_mask, out, total_iter):
    if args.dataset_name == 'scannet':
        d_max = 5.0
        e_max = 0.5
    else:
        d_max = 60.0
        e_max = 3.0

    pred_dmap, pred_var = torch.split(out, 1, dim=1)  # (B, 1, H, W)
    pred_stdev = torch.sqrt(pred_var)

    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()[0, ...]                     # (H, W, 3)
    gt_dmap = gt_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]         # (H, W)
    pred_dmap = pred_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]     # (H, W)
    pred_stdev = pred_stdev.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]   # (H, W)

    # save image
    target_path = '%s/%08d_img.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, unnormalize(img))

    # gt dmap
    target_path = '%s/%08d_gt_dmap.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, gt_dmap, vmin=0.0, vmax=d_max, cmap='jet')

    # pred dmap
    target_path = '%s/%08d_pred_dmap.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, pred_dmap, vmin=0.0, vmax=d_max, cmap='jet')

    # pred emap
    pred_emap = np.abs(pred_dmap - gt_dmap)
    pred_emap[gt_dmap < args.min_depth] = 0.0
    pred_emap[gt_dmap > args.max_depth] = 0.0
    target_path = '%s/%08d_pred_emap.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, pred_emap, vmin=0.0, vmax=e_max, cmap='Reds')

    # pred stdev
    target_path = '%s/%08d_pred_stdev.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, pred_stdev, vmin=0.0, vmax=e_max, cmap='Reds')


# visualize during training (FNET)
def visualize_F(args, img, gt_dmap, gt_dmap_mask, pred_dmap, total_iter):
    if args.dataset_name == 'scannet':
        d_max = 5.0
        e_max = 0.5
    else:
        d_max = 60.0
        e_max = 3.0

    # upsample
    pred_dmap = F.interpolate(pred_dmap, size=[img.shape[2], img.shape[3]], mode='nearest')

    # to numpy array
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()[0, ...]                    # (H, W, 3)
    gt_dmap = gt_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]        # (H, W)
    pred_dmap = pred_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]    # (H, W)

    # save image
    target_path = '%s/%08d_img.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, unnormalize(img))

    # gt dmap
    target_path = '%s/%08d_gt_dmap.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, gt_dmap, vmin=0.0, vmax=d_max, cmap='jet')

    # pred dmap
    target_path = '%s/%08d_pred_dmap.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, pred_dmap, vmin=0.0, vmax=d_max, cmap='jet')

    # pred emap
    pred_emap = np.abs(pred_dmap - gt_dmap)
    pred_emap[gt_dmap < args.min_depth] = 0.0
    pred_emap[gt_dmap > args.max_depth] = 0.0
    target_path = '%s/%08d_pred_emap.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, pred_emap, vmin=0.0, vmax=e_max, cmap='Reds')


# visualize during training (MAGNET)
def visualize_MaG(args, img, gt_dmap, gt_dmap_mask, pred_list, total_iter):
    if args.dataset_name == 'nyu' or args.dataset_name == 'scannet':
        d_max = 5.0
        e_max = 0.5
    else:
        d_max = 60.0
        e_max = 3.0
        
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()[0, ...]                # (H, W, 3)
    gt_dmap = gt_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]    # (H, W)

    # save image
    target_path = '%s/%08d_img.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, unnormalize(img))

    # gt dmap
    target_path = '%s/%08d_gt_dmap.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, gt_dmap, vmin=0.0, vmax=d_max, cmap='jet')

    for i in range(len(pred_list)):
        pred_dmap, pred_stdev = torch.split(pred_list[i], 1, dim=1)  # (B, 1, H, W)

        pred_dmap = pred_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]     # (H, W)
        pred_stdev = pred_stdev.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]     # (H, W)

        # pred dmap
        target_path = '%s/%08d_pred_dmap_iter%02d.jpg' % (args.exp_vis_dir, total_iter, i)
        plt.imsave(target_path, pred_dmap, vmin=0.0, vmax=d_max, cmap='jet')

        # pred emap
        pred_emap = np.abs(pred_dmap - gt_dmap)
        pred_emap[gt_dmap < args.min_depth] = 0.0
        pred_emap[gt_dmap > args.max_depth] = 0.0
        target_path = '%s/%08d_pred_emap_iter%02d.jpg' % (args.exp_vis_dir, total_iter, i)
        plt.imsave(target_path, pred_emap, vmin=0.0, vmax=e_max, cmap='Reds')

        # pred stdev
        target_path = '%s/%08d_pred_stdev_iter%02d.jpg' % (args.exp_vis_dir, total_iter, i)
        plt.imsave(target_path, pred_stdev, vmin=0.0, vmax=e_max, cmap='Reds')

