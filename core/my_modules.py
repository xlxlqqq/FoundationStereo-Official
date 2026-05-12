"""
轻量级 Window-based Transformer Attention 模块

设计目标:
  - 在浅层特征提取阶段(1/4分辨率)加入局部 Window Attention
  - 增强弱纹理/重复纹理/边缘区域的特征表达能力
  - 保持低计算复杂度: O(window_size^2 * C) per window
  - 输出 tensor shape 与输入完全一致, 可即插即用
  - 使用标准 PyTorch 操作, 兼容 TensorRT 导出

结构:
  Input (B,C,H,W)
      ↓
  LayerNorm → Window Partition → Multi-Head Self-Attention → Window Merge → Residual
      ↓
  LayerNorm → ConvFFN (DWConv + Linear) → Residual
      ↓
  Output (B,C,H,W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowMultiHeadSelfAttention(nn.Module):
    """
    局部窗口多头自注意力 (Window-based MHSA)

    将特征图划分为不重叠的窗口, 在每个窗口内独立计算 self-attention.
    复杂度: O(Ws^2 * C) per window, 其中 Ws = window_size
    相比全局 attention O(H*W*C), 大幅降低显存与计算量.

    使用相对位置偏置 (relative position bias), 参考 Swin Transformer.
    """

    def __init__(self, dim, num_heads=4, window_size=8, qkv_bias=True):
        """
        Args:
            dim: 输入通道数
            num_heads: 注意力头数
            window_size: 窗口大小 (正方形)
            qkv_bias: QKV 投影是否使用 bias
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # 相对位置偏置参数表
        # 窗口内 2*Ws-1 个相对位置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 预计算相对位置索引 (注册为 buffer, 不参与梯度)
        self._compute_relative_position_index()

        # Stereo-aware Learnable Directional Bias
        # 水平方向权重 (强化, 符合 epipolar constraint: y_left = y_right)
        self.horizontal_alpha = nn.Parameter(torch.tensor(1.5))
        # 垂直方向权重 (抑制, 垂直方向不满足立体匹配约束)
        self.vertical_beta = nn.Parameter(torch.tensor(1.0))

        self.scale = self.head_dim ** -0.5

    def _compute_relative_position_index(self):
        """预计算窗口内相对位置索引, 参考 Swin Transformer"""
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Ws, Ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, Ws*Ws)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords_perm = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)

        # 偏移使值为非负
        relative_coords_perm[:, :, 0] += self.window_size - 1
        relative_coords_perm[:, :, 1] += self.window_size - 1
        relative_coords_perm[:, :, 0] *= 2 * self.window_size - 1

        relative_position_index = relative_coords_perm.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        # Stereo-aware: 预计算方向偏移量 (绝对值归一化到 [0, 1])
        # relative_coords[0] = dy (垂直方向偏移), relative_coords[1] = dx (水平方向偏移)
        vertical_offset = relative_coords[0].abs().float() / (self.window_size - 1)   # (N, N)
        horizontal_offset = relative_coords[1].abs().float() / (self.window_size - 1)  # (N, N)
        self.register_buffer("vertical_offset_norm", vertical_offset)
        self.register_buffer("horizontal_offset_norm", horizontal_offset)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 特征图
        Returns:
            (B, C, H, W) 注意力增强后的特征图
        """
        B, C, H, W = x.shape
        Ws = self.window_size

        # 如果 H/W 不是 window_size 的整数倍, 则 pad
        pad_h = (Ws - H % Ws) % Ws
        pad_w = (Ws - W % Ws) % Ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, Hp, Wp = x.shape

        # (B, C, Hp, Wp) -> (B * nH * nW, Ws*Ws, C)
        nH_win = Hp // Ws
        nW_win = Wp // Ws
        x = x.reshape(B, C, nH_win, Ws, nW_win, Ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * nH_win * nW_win, Ws * Ws, C)

        # QKV 投影
        qkv = self.qkv(x).reshape(B * nH_win * nW_win, Ws * Ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nW, nH, Ws*Ws, head_dim)
        q, k, v = qkv.unbind(0)  # 各 (B*nW, nH, Ws*Ws, head_dim)

        # 注意力计算 + 相对位置偏置
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B*nW, nH, Ws*Ws, Ws*Ws)

        # 取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(Ws * Ws, Ws * Ws, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, N, N)

        # Stereo-aware Directional Bias: 方向加权
        # direction_weight = vertical_beta * |dy|_norm + horizontal_alpha * |dx|_norm
        # 水平方向偏移获得更大权重 (horizontal_alpha > vertical_beta), 符合 epipolar constraint
        direction_weight = (
            self.vertical_beta * self.vertical_offset_norm
            + self.horizontal_alpha * self.horizontal_offset_norm
        )  # (N, N), 范围 [0, alpha+beta]
        relative_position_bias = relative_position_bias * direction_weight.unsqueeze(0)  # (nH, N, N)

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)

        # 注意力加权
        out = (attn @ v).transpose(1, 2).reshape(B * nH_win * nW_win, Ws * Ws, C)

        # 输出投影
        out = self.proj(out)

        # 还原特征图形状: (B*nW, N, C) -> (B, C, Hp, Wp)
        out = out.reshape(B, nH_win, nW_win, Ws, Ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)

        # 去除 padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


class MLPFFN(nn.Module):
    """
    轻量级 MLP-based Feed-Forward Network

    纯线性层实现的 FFN，结构简单高效，兼容 TensorRT 导出.
    结构: Linear → GELU → Linear
    """

    def __init__(self, dim, hidden_dim=None, drop=0.0):
        """
        Args:
            dim: 输入/输出通道数
            hidden_dim: 隐藏层维度, 默认 dim*2 (比标准 Swin 的 dim*4 更轻量)
            drop: dropout 比率
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2  # 轻量化: 使用 2x 扩展而非 Swin 标准的 4x

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B*H*W, C) -> fc1 -> GELU -> fc2 -> (B, C, H, W)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x_flat = self.fc1(x_flat)
        x_flat = self.act(x_flat)
        x_flat = self.drop(x_flat)
        x_flat = self.fc2(x_flat)
        x_flat = self.drop(x_flat)
        out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class WindowAttentionBlock(nn.Module):
    """
    CNN + Window Transformer Hybrid Block

    完整结构:
        Input (B,C,H,W)
            ↓
        LayerNorm → Window MHSA → Residual Connection
            ↓
        LayerNorm → MLPFFN → Residual Connection
            ↓
        Output (B,C,H,W)

    设计要点:
      1. Pre-norm (先 norm 再 attention/FFN), 训练更稳定
      2. Window-based MHSA: 复杂度 O(Ws^2*C) per window, 远低于全局 O(H*W*C)
      3. 相对位置偏置: 让注意力感知窗口内空间关系
      4. MLPFFN: 纯线性层实现，结构简单高效，兼容 TensorRT 导出
      5. 输入输出 shape 完全一致, 即插即用
    """

    def __init__(self, dim, num_heads=4, window_size=8, ffn_ratio=2, drop=0.0):
        """
        Args:
            dim: 输入/输出通道数
            num_heads: 注意力头数 (dim 必须能被 num_heads 整除)
            window_size: 窗口大小, 推荐 8
            ffn_ratio: FFN 隐藏层扩展比 (默认 2, 比 Swin 标准 4 更轻量)
            drop: dropout 比率
        """
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowMultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLPFFN(dim=dim, hidden_dim=dim * ffn_ratio, drop=drop)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — shape 与输入完全一致
        """
        B, C, H, W = x.shape

        # Window Attention + Residual
        shortcut = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1))  # (B, H, W, C)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = shortcut + self.attn(x_norm)

        # MLPFFN + Residual
        shortcut = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1))
        x_norm = x_norm.permute(0, 3, 1, 2)
        x = shortcut + self.ffn(x_norm)

        return x