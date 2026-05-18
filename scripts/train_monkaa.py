# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import logging
import time
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# Optional matplotlib support for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class DummyWriter:
    """Simple replacement for SummaryWriter when tensorboard is not available"""
    def add_scalar(self, *args, **kwargs):
        pass
    def close(self):
        pass

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
from Utils import set_logging_format, set_seed

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

class MonkaaLiteDataset(Dataset):
    def __init__(self, dataset_dir, split='train', transform=None, img_scale=1.0):
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        self.img_scale = img_scale
        
        self.left_dir = os.path.join(dataset_dir, 'frames_finalpass')
        self.disp_dir = os.path.join(dataset_dir, 'disparity')
        
        self.samples = []
        
        for scene in os.listdir(self.left_dir):
            scene_left_dir = os.path.join(self.left_dir, scene, 'left')
            scene_right_dir = os.path.join(self.left_dir, scene, 'right')
            scene_disp_dir = os.path.join(self.disp_dir, scene, 'right')
            
            if not os.path.exists(scene_left_dir) or not os.path.exists(scene_right_dir) or not os.path.exists(scene_disp_dir):
                continue
            
            for filename in os.listdir(scene_left_dir):
                if filename.endswith('.png'):
                    frame_id = filename[:-4]
                    left_path = os.path.join(scene_left_dir, filename)
                    right_path = os.path.join(scene_right_dir, filename)
                    disp_path = os.path.join(scene_disp_dir, f'{frame_id}.pfm')
                    
                    if os.path.exists(right_path) and os.path.exists(disp_path):
                        self.samples.append({
                            'left': left_path,
                            'right': right_path,
                            'disp': disp_path
                        })
        
        split_idx = int(len(self.samples) * 0.8)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        logging.info(f"Loaded {len(self.samples)} {split} samples from monkaa_lite dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        import cv2
        left = cv2.imread(sample['left']).astype(np.float32)
        right = cv2.imread(sample['right']).astype(np.float32)
        disp = read_pfm(sample['disp']).astype(np.float32)
        
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        
        if self.img_scale != 1.0:
            h, w = left.shape[:2]
            new_h, new_w = int(h * self.img_scale), int(w * self.img_scale)
            new_h = ((new_h + 31) // 32) * 32
            new_w = ((new_w + 31) // 32) * 32
            left = cv2.resize(left, (new_w, new_h))
            right = cv2.resize(right, (new_w, new_h))
            disp = cv2.resize(disp, (new_w, new_h))
        
        if self.transform:
            left, right, disp = self.transform(left, right, disp)
        
        left = torch.from_numpy(left).permute(2, 0, 1)
        right = torch.from_numpy(right).permute(2, 0, 1)
        disp = torch.from_numpy(disp)
        mask = torch.ones_like(disp)
        
        return left, right, disp, mask

def compute_metrics(pred_disp, gt_disp, mask):
    """Compute stereo matching metrics"""
    valid = (mask > 0) & (gt_disp > 0) & (gt_disp < 200) & (pred_disp > 0) & (pred_disp < 200)
    
    if valid.sum() == 0:
        return {
            'epe': torch.tensor(0.0),
            'l1': torch.tensor(0.0),
            'd1_3px': torch.tensor(0.0),
            'd1_5pct': torch.tensor(0.0),
            'valid_ratio': torch.tensor(0.0)
        }
    
    diff = torch.abs(pred_disp[valid] - gt_disp[valid])
    epe = torch.sqrt(torch.mean(diff ** 2))
    l1 = torch.mean(diff)
    d1_3px = (diff > 3).float().mean()
    d1_5pct = (diff > 0.05 * gt_disp[valid]).float().mean()
    valid_ratio = valid.float().mean()
    
    return {
        'epe': epe,
        'l1': l1,
        'd1_3px': d1_3px,
        'd1_5pct': d1_5pct,
        'valid_ratio': valid_ratio
    }

def compute_edge_mask_from_disp(disp, threshold=1.0):
    """从视差图计算边缘掩码（深度不连续区域）"""
    grad_x = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    grad_y = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])
    
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
    
    edge_mask = (grad_x > threshold) | (grad_y > threshold)
    return edge_mask.float()


def train_one_epoch(model, train_loader, optimizer, epoch, args, writer, scaler=None):
    model.train()
    total_loss = 0.0
    total_epe = 0.0
    total_l1 = 0.0
    total_d1_3px = 0.0
    total_d1_5pct = 0.0
    total_edge_loss = 0.0
    total_samples = 0
    
    start_time = time.time()
    accum_iter = 0
    
    use_earr = getattr(model, 'use_earr', False)
    edge_weight = 2.0
    non_edge_weight = 0.5
    edge_supervision_weight = 0.1
    
    for batch_idx, (left, right, disp_gt, mask) in enumerate(train_loader):
        left = left.cuda().float()
        right = right.cuda().float()
        disp_gt = disp_gt.cuda().float()
        mask = mask.cuda().float()
        
        if args.mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast(True):
                if use_earr:
                    init_disp, disp_preds, edge_map, edge_loss = model(
                        left, right, iters=args.train_iters, 
                        low_memory=args.low_memory,
                        return_edge_info=True
                    )
                else:
                    init_disp, disp_preds = model(
                        left, right, iters=args.train_iters, 
                        low_memory=args.low_memory
                    )
                    edge_map = None
                    edge_loss = None
                
                disp_pred = disp_preds[-1].squeeze(1)
                valid = (mask > 0) & (disp_gt > 0) & (disp_gt < 200)
                
                loss = F.l1_loss(disp_pred[valid], disp_gt[valid])
                
                if use_earr and edge_map is not None:
                    edge_mask_gt = compute_edge_mask_from_disp(disp_gt.unsqueeze(1))
                    edge_mask_gt = edge_mask_gt.squeeze(1)
                    
                    weight_map = edge_mask_gt * edge_weight + (1 - edge_mask_gt) * non_edge_weight
                    weight_map = weight_map[valid]
                    
                    diff = torch.abs(disp_pred[valid] - disp_gt[valid])
                    edge_weighted_loss = torch.mean(diff * weight_map)
                    loss = loss + edge_weighted_loss
                    
                    if edge_loss is not None:
                        loss = loss + edge_supervision_weight * edge_loss
            
            loss = loss / args.accum_steps
            scaler.scale(loss).backward()
            
            accum_iter += 1
            if accum_iter % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_iter = 0
        else:
            if use_earr:
                init_disp, disp_preds, edge_map, edge_loss = model(
                    left, right, iters=args.train_iters, 
                    low_memory=args.low_memory,
                    return_edge_info=True
                )
            else:
                init_disp, disp_preds = model(
                    left, right, iters=args.train_iters, 
                    low_memory=args.low_memory
                )
                edge_map = None
                edge_loss = None
            
            disp_pred = disp_preds[-1].squeeze(1)
            valid = (mask > 0) & (disp_gt > 0) & (disp_gt < 200)
            
            loss = F.l1_loss(disp_pred[valid], disp_gt[valid])
            
            if use_earr and edge_map is not None:
                edge_mask_gt = compute_edge_mask_from_disp(disp_gt.unsqueeze(1))
                edge_mask_gt = edge_mask_gt.squeeze(1)
                
                weight_map = edge_mask_gt * edge_weight + (1 - edge_mask_gt) * non_edge_weight
                weight_map = weight_map[valid]
                
                diff = torch.abs(disp_pred[valid] - disp_gt[valid])
                edge_weighted_loss = torch.mean(diff * weight_map)
                loss = loss + edge_weighted_loss
                
                if edge_loss is not None:
                    loss = loss + edge_supervision_weight * edge_loss
            
            loss = loss / args.accum_steps
            loss.backward()
            
            accum_iter += 1
            if accum_iter % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accum_iter = 0
        
        metrics = compute_metrics(disp_pred, disp_gt, mask)
        
        total_loss += loss.item() * args.accum_steps * left.shape[0]
        if use_earr and edge_loss is not None:
            total_edge_loss += edge_loss.item() * left.shape[0]
        total_epe += metrics['epe'].item() * left.shape[0]
        total_l1 += metrics['l1'].item() * left.shape[0]
        total_d1_3px += metrics['d1_3px'].item() * left.shape[0]
        total_d1_5pct += metrics['d1_5pct'].item() * left.shape[0]
        total_samples += left.shape[0]
        
        if batch_idx % args.log_interval == 0:
            avg_loss = total_loss / total_samples
            avg_epe = total_epe / total_samples
            avg_l1 = total_l1 / total_samples
            log_str = f"Epoch {epoch}/{args.epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {avg_loss:.4f} - EPE: {avg_epe:.4f} - L1: {avg_l1:.4f}"
            if use_earr:
                avg_edge_loss = total_edge_loss / total_samples if total_samples > 0 else 0.0
                log_str += f" - EdgeLoss: {avg_edge_loss:.4f}"
            logging.info(log_str)
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    avg_epe = total_epe / total_samples
    avg_l1 = total_l1 / total_samples
    avg_d1_3px = total_d1_3px / total_samples
    avg_d1_5pct = total_d1_5pct / total_samples
    
    if HAS_TENSORBOARD:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/epe', avg_epe, epoch)
        writer.add_scalar('train/l1', avg_l1, epoch)
        writer.add_scalar('train/d1_3px', avg_d1_3px, epoch)
        writer.add_scalar('train/d1_5pct', avg_d1_5pct, epoch)
        if use_earr:
            avg_edge_loss = total_edge_loss / total_samples if total_samples > 0 else 0.0
            writer.add_scalar('train/edge_loss', avg_edge_loss, epoch)
    
    return avg_loss, avg_epe, avg_l1, avg_d1_3px, avg_d1_5pct, epoch_time

def validate(model, val_loader, args):
    model.eval()
    total_epe = 0.0
    total_l1 = 0.0
    total_d1_3px = 0.0
    total_d1_5pct = 0.0
    total_valid_ratio = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for left, right, disp_gt, mask in val_loader:
            left = left.cuda().float()
            right = right.cuda().float()
            disp_gt = disp_gt.cuda().float()
            mask = mask.cuda().float()
            
            with torch.cuda.amp.autocast(args.mixed_precision):
                disp_pred = model(left, right, iters=args.valid_iters, test_mode=True, low_memory=args.low_memory)
            disp_pred = disp_pred.squeeze(1)
            
            metrics = compute_metrics(disp_pred, disp_gt, mask)
            
            total_epe += metrics['epe'].item() * left.shape[0]
            total_l1 += metrics['l1'].item() * left.shape[0]
            total_d1_3px += metrics['d1_3px'].item() * left.shape[0]
            total_d1_5pct += metrics['d1_5pct'].item() * left.shape[0]
            total_valid_ratio += metrics['valid_ratio'].item() * left.shape[0]
            total_samples += left.shape[0]
    
    avg_epe = total_epe / total_samples
    avg_l1 = total_l1 / total_samples
    avg_d1_3px = total_d1_3px / total_samples
    avg_d1_5pct = total_d1_5pct / total_samples
    avg_valid_ratio = total_valid_ratio / total_samples
    
    return {
        'epe': avg_epe,
        'l1': avg_l1,
        'd1_3px': avg_d1_3px,
        'd1_5pct': avg_d1_5pct,
        'valid_ratio': avg_valid_ratio
    }

def plot_metrics(epochs, train_metrics, val_metrics, out_dir):
    """Plot training and validation metrics"""
    if not HAS_MATPLOTLIB:
        logging.warning("matplotlib not installed, skipping plotting")
        return
    
    metrics_to_plot = ['epe', 'd1_5pct', 'd1_3px']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_metrics[metric], label=f'Train {metric}', color='blue')
        plt.plot(epochs, val_metrics[metric], label=f'Val {metric}', color='red')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f'{metric}_plot.png'))
        plt.close()
    
    plt.figure(figsize=(10, 6))
    train_total = [epe + l1 + d1_3px + d1_5pct for epe, l1, d1_3px, d1_5pct in 
                   zip(train_metrics['epe'], train_metrics['l1'], train_metrics['d1_3px'], train_metrics['d1_5pct'])]
    val_total = [epe + l1 + d1_3px + d1_5pct for epe, l1, d1_3px, d1_5pct in 
                 zip(val_metrics['epe'], val_metrics['l1'], val_metrics['d1_3px'], val_metrics['d1_5pct'])]
    plt.plot(epochs, train_total, label='Train Total', color='blue')
    plt.plot(epochs, val_total, label='Val Total', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Total')
    plt.title('Total Metric vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'total_plot.png'))
    plt.close()
    
    logging.info(f"Plots saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train FoundationStereo on Monkaa Lite Dataset')
    
    # Dataset settings
    parser.add_argument('--dataset_dir', default=f'{code_dir}/../../FoundationCStereo/data/monkaa_lite', type=str)
    parser.add_argument('--img_scale', default=0.5, type=float, help='Scale factor for input images (reduces memory usage)')
    
    # Model settings
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str)
    parser.add_argument('--no_pretrained', action='store_true', default=False, help='Do not load pretrained weights, train from scratch')
    parser.add_argument('--vit_size', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--low_memory', action='store_true', default=True, help='Enable low memory mode')
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Enable mixed precision training')
    parser.add_argument('--use_earr', action='store_true', default=True, help='Enable Edge-Aware Residual Refinement (EARR) module')
    
    # Training settings
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='Per-GPU batch size')
    parser.add_argument('--accum_steps', default=4, type=int, help='Gradient accumulation steps')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--train_iters', default=22, type=int)
    parser.add_argument('--valid_iters', default=32, type=int)
    
    # Logging settings
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--save_interval', default=10, type=int)
    parser.add_argument('--out_dir', default=f'{code_dir}/../train_output_monkaa', type=str)
    
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    args.out_dir = os.path.join(args.out_dir, timestamp)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    log_file = os.path.join(args.out_dir, 'train.log')
    set_logging_format()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(file_handler)
    set_seed(0)
    
    if HAS_TENSORBOARD:
        writer = SummaryWriter(args.out_dir)
    else:
        writer = DummyWriter()
    
    cfg = OmegaConf.load(f'{os.path.dirname(args.ckpt_dir)}/cfg.yaml')
    cfg['vit_size'] = args.vit_size
    cfg['train_iters'] = args.train_iters
    cfg['valid_iters'] = args.valid_iters
    cfg['low_memory'] = args.low_memory
    cfg['mixed_precision'] = args.mixed_precision
    cfg['use_earr'] = args.use_earr
    
    model = FoundationStereo(cfg)
    
    if args.no_pretrained:
        logging.info("Training from scratch without pretrained weights")
    else:
        logging.info(f"Loading pretrained model from {args.ckpt_dir}")
        ckpt = torch.load(args.ckpt_dir, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        logging.info("Loaded pretrained weights (ignoring new window_attn layers)")
    
    model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    
    train_dataset = MonkaaLiteDataset(args.dataset_dir, split='train', img_scale=args.img_scale)
    val_dataset = MonkaaLiteDataset(args.dataset_dir, split='val', img_scale=args.img_scale)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    best_epe = float('inf')
    effective_batch_size = args.batch_size * args.accum_steps
    
    train_metrics = {'epe': [], 'l1': [], 'd1_3px': [], 'd1_5pct': []}
    val_metrics = {'epe': [], 'l1': [], 'd1_3px': [], 'd1_5pct': []}
    epochs_list = []
    
    logging.info(f"Starting training for {args.epochs} epochs...")
    logging.info(f"Effective batch size: {effective_batch_size} (batch_size={args.batch_size}, accum_steps={args.accum_steps})")
    logging.info(f"Image scale: {args.img_scale}")
    logging.info(f"Low memory mode: {args.low_memory}")
    logging.info(f"Mixed precision: {args.mixed_precision}")
    logging.info(f"Edge-Aware Residual Refinement (EARR): {args.use_earr}")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_epe, train_l1, train_d1_3px, train_d1_5pct, epoch_time = train_one_epoch(
            model, train_loader, optimizer, epoch, args, writer, scaler)
        
        if epoch % args.val_interval == 0:
            val_results = validate(model, val_loader, args)
            
            epochs_list.append(epoch)
            train_metrics['epe'].append(train_epe)
            train_metrics['l1'].append(train_l1)
            train_metrics['d1_3px'].append(train_d1_3px)
            train_metrics['d1_5pct'].append(train_d1_5pct)
            val_metrics['epe'].append(val_results['epe'])
            val_metrics['l1'].append(val_results['l1'])
            val_metrics['d1_3px'].append(val_results['d1_3px'])
            val_metrics['d1_5pct'].append(val_results['d1_5pct'])
            
            total = val_results['epe'] + val_results['l1'] + val_results['d1_3px'] + val_results['d1_5pct']
            
            is_best = val_results['epe'] < best_epe
            if is_best:
                best_epe = val_results['epe']
            
            logging.info(f"Epoch {epoch}/{args.epochs} - Time: {epoch_time:.1f}s - Best EPE: {best_epe:.4f}")
            logging.info(f"  d1_3px: {val_results['d1_3px']:.4f}")
            logging.info(f"  d1_5pct: {val_results['d1_5pct']:.4f}")
            logging.info(f"  epe: {val_results['epe']:.4f}")
            logging.info(f"  l1: {val_results['l1']:.4f}")
            logging.info(f"  total: {total:.4f}")
            logging.info(f"  valid_ratio: {val_results['valid_ratio']:.4f}")
            
            if HAS_TENSORBOARD:
                writer.add_scalar('val/epe', val_results['epe'], epoch)
                writer.add_scalar('val/l1', val_results['l1'], epoch)
                writer.add_scalar('val/d1_3px', val_results['d1_3px'], epoch)
                writer.add_scalar('val/d1_5pct', val_results['d1_5pct'], epoch)
                writer.add_scalar('val/valid_ratio', val_results['valid_ratio'], epoch)
                writer.add_scalar('val/total', total, epoch)
            
            if is_best:
                save_path = os.path.join(args.out_dir, 'model_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if args.mixed_precision else None,
                    'best_epe': best_epe,
                    'cfg': cfg
                }, save_path)
                logging.info(f"Saved best model to {save_path}")
            
            plot_metrics(epochs_list, train_metrics, val_metrics, args.out_dir)
    
    writer.close()
    
    final_params = {
        'args': vars(args),
        'cfg': OmegaConf.to_container(cfg),
        'best_epe': float(best_epe),
        'total_epochs': args.epochs,
        'effective_batch_size': effective_batch_size,
        'train_metrics': {k: [float(v) for v in vals] for k, vals in train_metrics.items()},
        'val_metrics': {k: [float(v) for v in vals] for k, vals in val_metrics.items()},
        'epochs': epochs_list
    }
    
    params_file = os.path.join(args.out_dir, 'training_params.json')
    with open(params_file, 'w') as f:
        json.dump(final_params, f, indent=4)
    logging.info(f"Training parameters saved to {params_file}")
    
    logging.info(f"Training completed. Best EPE: {best_epe:.4f}")

if __name__ == '__main__':
    main()
