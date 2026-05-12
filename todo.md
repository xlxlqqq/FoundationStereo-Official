# 任务目标

你需要基于当前的代码仓库，对 FoundationStereo（FS）进行改进。

本次改进的核心目标是：

# 在 Cost Aggregation（代价聚合）阶段引入 Residual Dense Aggregation（RDA，残差密集聚合）

以增强：

- cost volume 特征复用能力
- 弱纹理区域匹配能力
- 细节结构恢复能力
- disparity 连续性
- 边缘区域预测能力

要求：

- 保持 FoundationStereo 原始整体框架
- 尽量少改动原工程
- 保持训练流程兼容
- 保持原有输入输出接口不变
- 自己改进用到的新定义的模块，需要增加到./core/new_module.py中

不要重写整个网络。

---

# 核心思想

目前 FoundationStereo 的 aggregation 仍然主要依赖：

- 3D CNN
- Hourglass
- UNet-like aggregation

其问题在于：

- 深层传播过程中 cost 信息容易丢失
- 特征复用不足
- 小结构与边缘区域容易退化
- 弱纹理区域匹配稳定性不足

因此需要引入：

# Residual Dense Aggregation（残差密集聚合）

参考思想包括：

- DenseNet
- Residual Dense Network（RDN）
- Dense Feature Reuse
- Residual Learning

核心目标：

让 aggregation 阶段能够：

- 充分复用浅层 cost 信息
- 保持多尺度匹配细节
- 增强局部结构表达能力

---

# 重要要求（必须遵守）

## 不允许：

- 重写整个网络
- 大规模破坏原始结构
- 修改数据集读取流程
- 修改训练 pipeline
- 修改输入输出格式
- 修改 disparity prediction 接口
- 引入大量额外依赖

---

# 修改重点

重点修改：

# Cost Aggregation 阶段

请先分析源码中：

- aggregation module
- cost volume aggregation
- hourglass
- 3D CNN aggregation
- refinement module

等相关模块。

RDA 应该插入这些位置。

不要优先修改 backbone feature extractor。

---

# 需要新增的模块

新增模块：

# ResidualDenseAggregationBlock

建议结构：

```python
class ResidualDenseAggregationBlock(nn.Module):
    def __init__(...):
        ...
```

---

# 模块结构要求

模块内部需要包含：

## 1. Dense Connection（密集连接）

每一层都接收之前所有层的特征。

示例：

```python
x1 = conv1(x)

x2 = conv2(torch.cat([x, x1], dim=1))

x3 = conv3(torch.cat([x, x1, x2], dim=1))
```

---

## 2. Local Feature Fusion（局部特征融合）

由于 dense connection 会导致通道数迅速增长：

因此需要：

- 1×1 Conv
- channel compression
- local fusion

例如：

```python
fused = fusion(torch.cat([...], dim=1))
```

---

## 3. Residual Connection（残差连接）

最终输出必须：

```python
out = fused + x
```

保证训练稳定性。

---

# Conv 类型要求

必须保持与 FoundationStereo 原始 aggregation 风格一致。

例如：

- 原 aggregation 使用 Conv3D
→ RDA 继续使用 Conv3D

- 原 aggregation 使用 normalization
→ 保持相同 normalization 风格

- 原 aggregation 使用 activation
→ 保持一致

不要随意更改原始框架规范。

---

# 通道与显存控制（非常重要）

Dense connection 会快速增加显存占用。

因此必须：

- 使用 channel compression
- 避免通道无限增长
- 避免显存爆炸
- 控制参数量

要求：

修改后的模型仍然能够在普通现代 GPU 上训练。

不要设计过于庞大的结构。

---

# 推荐集成方式

优先采用：

# 方案 A（推荐）

替换部分 aggregation Conv Block 为 RDA Block。

例如：

```text
原始：
Conv3D -> Conv3D -> Conv3D

改为：
RDA Block -> RDA Block
```

---

# 可选方案

## 方案 B

在 aggregation stage 后插入：

```text
Aggregation
↓
RDA
↓
Next Stage
```

但优先选择：

# 最小侵入式修改

不要大规模重构网络。

---

# 工程规范要求

代码必须：

- 保持原工程代码风格
- 添加清晰注释
- 不允许硬编码 tensor shape
- 不允许 magic number
- 保持模块化
- 支持多尺度 feature
- 支持后续 ablation study

---

# 配置化要求（重要）

必须支持：

# 开关控制 RDA

例如：

```python
USE_RDA = True
```

或者：

yaml/config 配置。

要求：

关闭 RDA 后：

原始 FoundationStereo 必须仍然能够正常运行。

---

# Ablation Study 支持

代码结构必须支持：

- 原始 aggregation
- 部分 stage 使用 RDA
- 全部 aggregation 使用 RDA

方便后续实验。

---

# 修改流程要求

不要盲目修改。

必须：

1. 先分析源码结构
2. 找出真正 aggregation 位置
3. 分析 tensor flow
4. 再进行修改

不要直接假设文件名。

---

# 输出要求

你必须输出：

## 1. 完整修改代码

包括：

- 新增文件
- 修改文件
- 核心模块

---

## 2. 修改说明

详细说明：

- 修改了哪些文件
- 为什么修改
- RDA 插入位置
- tensor shape 如何变化
- aggregation 流程如何变化

---

## 3. 性能分析

分析：

- 参数量变化
- 显存变化
- FLOPs 变化
- 训练稳定性影响

---

## 4. 改进预期

说明：

RDA 为什么能够提升：

- disparity 连续性
- 边缘区域预测
- 小结构恢复
- 弱纹理区域匹配

---

# 重要原则

不要设计成：

“完全新的 stereo 网络”。

而是：

# FoundationStereo + Residual Dense Aggregation 增强版

保持：

- 工程合理性
- 可训练性
- 易维护性
- 易复现性

---

# 最终目标

最终模型应当：

在尽量少改动 FoundationStereo 的前提下：

增强：

- cost volume 表达能力
- dense feature reuse
- stereo matching 稳定性
- disparity recovery quality

特别是在：

- 边缘区域
- 弱纹理区域
- 细结构区域

具有更好的表现。