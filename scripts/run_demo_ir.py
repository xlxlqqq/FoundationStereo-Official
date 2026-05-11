import os
import sys
import argparse
import imageio
import torch
import logging
import cv2
import numpy as np
import open3d as o3d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo

def read_pfm(path):
    """Read PFM file (disparity map format)"""
    with open(path, 'rb') as f:
        header = f.readline().decode('latin-1').strip()
        if header not in ('PF', 'Pf'):
            raise Exception('Not a PFM file')
        dims = f.readline().decode('latin-1').strip()
        width, height = map(int, dims.split())
        scale = float(f.readline().decode('latin-1').strip())
        data = np.fromfile(f, '<f') if scale < 0 else np.fromfile(f, '>f')
        data = np.flipud(data.reshape(height, width))
        return data

if __name__=="__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_id', default='000001', type=str, help='frame ID (6-digit)')
    parser.add_argument('--dataset_dir', default=f'{code_dir}/../data/D435i_FOD_Dataset', type=str)
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str)
    parser.add_argument('--out_dir', default=f'{code_dir}/../output_ir', type=str)
    parser.add_argument('--scale', default=0.5, type=float, help='downsize the image by scale')
    parser.add_argument('--valid_iters', type=int, default=32)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")

    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    # 读取IR图像（单通道灰度图）
    left_ir_path = os.path.join(args.dataset_dir, 'left_ir', f'{args.frame_id}.png')
    right_ir_path = os.path.join(args.dataset_dir, 'right_ir', f'{args.frame_id}.png')
    disp_gt_path = os.path.join(args.dataset_dir, 'disparity', f'{args.frame_id}.pfm')
    
    logging.info(f"Reading IR images: {left_ir_path}")
    img0 = imageio.imread(left_ir_path)
    img1 = imageio.imread(right_ir_path)
    
    # IR图像是单通道，转换为三通道（复制到RGB）
    if len(img0.shape) == 2:
        img0 = np.stack([img0, img0, img0], axis=-1)
        img1 = np.stack([img1, img1, img1], axis=-1)
    
    logging.info(f"IR image shape: {img0.shape}")
    
    # 缩放图像
    img0 = cv2.resize(img0, fx=args.scale, fy=args.scale, dsize=None)
    img1 = cv2.resize(img1, fx=args.scale, fy=args.scale, dsize=None)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()

    # 转换为tensor
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    # 推理
    with torch.cuda.amp.autocast(True):
        disp_pred = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    disp_pred = padder.unpad(disp_pred.float())
    disp_pred = disp_pred.data.cpu().numpy().reshape(H, W)
    
    # 读取ground truth disparity
    disp_gt = read_pfm(disp_gt_path)
    disp_gt = cv2.resize(disp_gt, fx=args.scale, fy=args.scale, dsize=None)
    
    # 可视化
    vis_pred = vis_disparity(disp_pred)
    vis_gt = vis_disparity(disp_gt)
    
    # 合并显示：原始图像 + 预测视差 + GT视差
    img0_gray = cv2.cvtColor(img0_ori, cv2.COLOR_RGB2GRAY)
    img0_gray_rgb = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2RGB)
    
    combined = np.concatenate([img0_gray_rgb, vis_pred, vis_gt], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis_{args.frame_id}.png', combined)
    
    # 保存视差图
    np.save(f'{args.out_dir}/disp_pred_{args.frame_id}.npy', disp_pred)
    np.save(f'{args.out_dir}/disp_gt_{args.frame_id}.npy', disp_gt)
    
    # 计算误差指标
    valid_mask = (disp_gt > 0) & (disp_gt < 100) & (disp_pred > 0) & (disp_pred < 100)
    if np.sum(valid_mask) > 0:
        abs_diff = np.abs(disp_pred[valid_mask] - disp_gt[valid_mask])
        rmse = np.sqrt(np.mean(abs_diff ** 2))
        mae = np.mean(abs_diff)
        bad_pix = np.mean(abs_diff > 3) * 100
        logging.info(f"Frame {args.frame_id}: RMSE={rmse:.3f}, MAE={mae:.3f}, BadPix(>3px)={bad_pix:.2f}%")
    
    logging.info(f"Results saved to {args.out_dir}")