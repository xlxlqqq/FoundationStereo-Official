# 任务目标

你需要基于 FoundationStereo 当前的代码仓库：

https://github.com/NVlabs/FoundationStereo

实现一种：

# Lightweight Residual Dense Aggregation（轻量残差密集聚合，Light-RDA）

用于增强 FoundationStereo 的 Cost Aggregation 阶段。

本次任务的重点是：

# 在尽量控制显存与计算量的前提下，提高 stereo matching 性能。

特别关注：

- disparity 连续性
- 弱纹理区域匹配
- 边缘结构恢复
- 小目标细节
- cost volume 特征复用

---

# 重要设计原则（必须遵守）

本次改进：

# 不是构建全新的 stereo 网络。

而是：

# 对 FoundationStereo 进行轻量增强。

因此必须：

- 尽量少修改原工程
- 保持原始训练流程兼容
- 保持原输入输出接口
- 保持原 disparity prediction pipeline
- 保持原 tensor shape 流程

不要重写整体结构。

---

# 本次改进的核心要求

请实现：

# Lightweight Residual Dense Aggregation（Light-RDA）

而不是传统 DenseNet 风格的大型 dense block。

目标是：

# 用极小的额外开销获得明显的 cost aggregation 提升。

---

# 为什么需要 Light-RDA

传统 aggregation 存在：

- cost 信息传播不足
- 深层特征复用不足
- 小结构信息容易丢失
- disparity 边缘退化
- 弱纹理区域匹配不稳定

Dense aggregation 可以增强：

- feature reuse
- local matching consistency
- cost propagation

但：

# 传统 DenseNet 风格会导致 3D cost volume 显存爆炸。

因此：

必须采用：

# Lightweight Dense Aggregation

---

# 关键要求（非常重要）

## 必须严格控制显存增长

由于 stereo 使用：

```text
[B, C, D, H, W]
```

3D cost volume。

Dense connection 非常容易导致：

- channel 爆炸
- 显存暴涨
- 训练无法进行

因此：

本次实现必须：

# 优先考虑轻量化与工程可训练性。

---

# Light-RDA 模块设计要求

新增模块：

# LightweightResidualDenseAggregationBlock

推荐结构：

```python
class LightweightResidualDenseAggregationBlock(nn.Module):
    ...
```

---

# 内部结构要求

## 1. 小规模 Dense Connection

不要使用深层 DenseNet。

推荐：

# 仅 2~3 层 dense aggregation

例如：

```python
x1 = conv1(x)

x2 = conv2(torch.cat([x, x1], dim=1))
```

最多：

```python
x3 = conv3(torch.cat([x, x1, x2], dim=1))
```

不要继续堆叠。

---

# 2. 必须使用 Channel Compression

Dense 后必须：

# 使用 1×1 Conv 压缩通道

例如：

```python
fusion = nn.Conv3d(total_channels, original_channels, 1)
```

防止 channel 无限增长。

这是强制要求。

---

# 3. 必须使用 Residual Connection

最终输出：

```python
out = fusion + x
```

保证训练稳定性。

---

# 4. 保持输入输出 shape 不变

模块输入输出必须：

```text
same shape
```

不要改变：

- disparity dimension
- spatial resolution
- channel size

---

# Growth Rate 要求（非常重要）

不要使用：

```python
growth_rate = 32
```

推荐：

```python
growth_rate = 4
```

或者：

```python
growth_rate = 8
```

目标：

# 小增长率 + 高效率

而不是大模型。

---

# Aggregation 插入位置（关键）

重点修改：

# Cost Aggregation 阶段

请先分析：

- cost aggregation
- 3D aggregation
- hourglass
- refinement module

再进行插入。

---

# 插入策略（推荐）

## 推荐方案

仅在：

# 中低分辨率 aggregation stage

使用 Light-RDA。

例如：

```text
1/8 scale
1/16 scale
```

不要在 full-resolution volume 上大量使用 dense aggregation。

---

# 不推荐方案（禁止）

不要：

- 所有 stage 全部加 dense
- full DenseNet 结构
- 无限 channel concat
- 高 growth rate
- 超深 dense block

这些会导致：

# 3D stereo 显存爆炸。

---

# Conv 类型要求

必须保持与原始 FS 风格一致。

例如：

- 原 aggregation 使用 Conv3D
→ Light-RDA 继续使用 Conv3D

- 原 normalization
→ 保持一致

- 原 activation
→ 保持一致

不要随意改变框架风格。

---

# 配置化要求（必须支持）

增加配置开关：

```python
USE_LIGHT_RDA = True
```

并支持：

- 原始 aggregation
- 部分 aggregation 使用 Light-RDA
- 全部 aggregation 使用 Light-RDA

方便后续 ablation study。

---

# 工程规范要求

代码必须：

- 模块化
- 可维护
- 易复现
- 避免硬编码
- 添加清晰注释
- 保持原工程风格
- 支持多尺度 feature

---

# 训练兼容性要求

修改后：

- 原训练脚本必须仍然可运行
- 原 dataset pipeline 不允许修改
- 原 inference pipeline 不允许破坏

---

# 需要输出的内容

你必须输出：

---

# 1. 完整修改代码

包括：

- 新增模块
- 修改模块
- 修改文件列表

---

# 2. 修改说明

详细说明：

- Light-RDA 插入位置
- tensor flow
- aggregation 如何变化
- 为什么这样设计

---

# 3. 显存与计算分析

分析：

- 参数量变化
- FLOPs 变化
- 显存变化
- 为什么属于 lightweight design

---

# 4. 训练建议

给出：

- batch size 建议
- mixed precision 建议
- 显存优化建议

---

# 最终目标

在：

# 尽量小的额外开销

下：

提升：

- stereo matching robustness
- disparity continuity
- edge reconstruction
- weak-texture matching
- fine structure recovery

同时：

保持：

- FoundationStereo 原始框架兼容性
- 训练稳定性
- 工程可维护性
- 工业场景可部署性