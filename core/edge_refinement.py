# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDetector(nn.Module):
    """轻量级边缘检测器，使用深度可分离卷积"""
    def __init__(self, in_channels=3, hidden_dim=16):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 第一层深度可分离卷积
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # 第二层深度可分离卷积
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            # 第三层深度可分离卷积
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2, bias=False),
            nn.Conv2d(hidden_dim * 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv_layers(x)


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EdgeAwareRefinement(nn.Module):
    """
    Edge-Aware Residual Refinement (EARR) module
    用于在GRU迭代后对视差图进行边缘感知的残差细化
    
    输入:
        disp_final: B×1×H×W - GRU输出的最终视差图
        feat_left: B×C×H×W - 编码器提取的左图像特征
        img_left: B×3×H×W - 原始左图像（需与特征图尺寸匹配）
    
    输出:
        disp_refined: B×1×H×W - 细化后的视差图
        edge_map: B×1×H×W - 边缘概率图（用于辅助损失）
    """
    
    def __init__(self, feat_dim=128, residual_channels=64):
        super().__init__()
        
        # 边缘检测器
        self.edge_detector = EdgeDetector(in_channels=3, hidden_dim=16)
        
        # 记录配置参数
        self.feat_dim = feat_dim
        self.residual_channels = residual_channels
        
        # 残差预测网络 - 使用普通卷积替代深度可分离卷积以支持动态输入通道
        self.residual_conv1 = nn.Conv2d(feat_dim + 2, residual_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(residual_channels)
        
        self.residual_conv2 = nn.Conv2d(residual_channels, residual_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(residual_channels // 2)
        
        self.residual_conv3 = nn.Conv2d(residual_channels // 2, 1, kernel_size=3, padding=1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def compute_sobel_edge(self, img):
        """计算Sobel梯度作为边缘监督信号"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
        
        # 转换为灰度图
        gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        
        # 计算梯度
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        # 梯度幅度
        edge_sobel = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_sobel = (edge_sobel - edge_sobel.min()) / (edge_sobel.max() - edge_sobel.min() + 1e-6)
        
        return edge_sobel
    
    def forward(self, disp_final, feat_left, img_left, return_edge_loss=False):
        """
        前向传播
        
        Args:
            disp_final: B×1×H×W - GRU输出的最终视差图
            feat_left: B×C×H×W - 编码器特征
            img_left: B×3×H×W - 原始左图像
            return_edge_loss: 是否返回边缘监督损失
        
        Returns:
            disp_refined: B×1×H×W - 细化后的视差图
            edge_map: B×1×H×W - 边缘概率图
            edge_loss: 边缘监督损失（可选）
        """
        B, C, H, W = feat_left.shape
        
        # 确保图像尺寸匹配
        if img_left.shape[-2:] != (H, W):
            img_left = F.interpolate(img_left, size=(H, W), mode='bilinear', align_corners=True)
        
        # 1. 计算边缘图
        edge_map = self.edge_detector(img_left)
        
        # 2. 计算Sobel边缘作为辅助监督
        if return_edge_loss and self.training:
            sobel_edge = self.compute_sobel_edge(img_left)
            edge_loss = F.l1_loss(edge_map, sobel_edge)
        else:
            edge_loss = None
        
        # 3. 拼接输入特征
        # 确保disp_final维度正确
        if disp_final.dim() == 3:
            disp_final = disp_final.unsqueeze(1)
        
        # 拼接特征: disp_final(1) + feat_left(C) + edge_map(1)
        concat_feat = torch.cat([disp_final, feat_left, edge_map], dim=1)
        
        # 4. 预测残差
        x = F.relu(self.bn1(self.residual_conv1(concat_feat)))
        x = F.relu(self.bn2(self.residual_conv2(x)))
        delta_disp = self.residual_conv3(x)
        
        # 5. 边缘门控的残差细化
        disp_refined = disp_final + delta_disp * edge_map
        
        if return_edge_loss:
            return disp_refined, edge_map, edge_loss
        
        return disp_refined, edge_map


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试模块
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模块
    earr = EdgeAwareRefinement(feat_dim=128, residual_channels=64).to(device)
    
    # 计算参数数量
    params = count_parameters(earr)
    print(f"EdgeAwareRefinement 参数数量: {params / 1e6:.2f}M")
    
    # 测试前向传播
    B, C, H, W = 2, 128, 120, 160
    
    disp_final = torch.randn(B, 1, H, W).to(device)
    feat_left = torch.randn(B, C, H, W).to(device)
    img_left = torch.randn(B, 3, H, W).to(device)
    
    disp_refined, edge_map = earr(disp_final, feat_left, img_left)
    print(f"输入视差图形状: {disp_final.shape}")
    print(f"输入特征图形状: {feat_left.shape}")
    print(f"输入图像形状: {img_left.shape}")
    print(f"输出视差图形状: {disp_refined.shape}")
    print(f"边缘图形状: {edge_map.shape}")
    
    # 测试返回边缘损失
    disp_refined, edge_map, edge_loss = earr(disp_final, feat_left, img_left, return_edge_loss=True)
    print(f"边缘损失: {edge_loss.item()}")