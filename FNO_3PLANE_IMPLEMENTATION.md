# FNO 3-Plane Implementation Summary

## Overview

成功实现了基于 Fourier Neural Operator (FNO) 的 3平面湍流预测模型 `flow_FNO_3plane`，作为现有 Swin Transformer 模型的对比方案。

**使用neuralop官方库实现，确保模型的标准性和可靠性。**

## 新增文件清单

### 1. 模型实现
- **src/models/nop/fno_2d_temporal.py**
  - 实现了 `FNO2DTemporalModel` 类
  - 使用 `neuralop.models.FNO2d` 作为核心组件
  - 包含Channel MLP用于跨通道信息混合
  - 参数量：~2.4M（约为Swin的1/40）

### 2. 配置文件
- **configs/model/fno_3plane.yaml**
  - 模型超参数配置
  - hidden_channels: 64
  - num_layers: 4
  - num_modes: [16, 16]
  - lifting/projection_channels: 128
  - 使用neuralop内置的Channel MLP

- **configs/train_flow_fno_3plane.yaml**
  - 训练配置（复用大部分Swin配置）
  - task_name: "flow_fno_3plane"
  - log_project: "turbulence_fno_3plane"

### 3. 训练脚本
- **start_train_flow_fno_3plane_conda.sh**
  - 训练 + 评估一体化脚本
  - 自动使用 `evaluation_3plane.py` 进行评估
  - 结果保存到 `logs/flow_fno_3plane/`

### 4. 测试脚本
- **test_fno_simple.py**
  - 模型功能测试
  - 验证前向传播、梯度流等

## 架构对比

### FNO vs Swin Transformer

| 特性 | FNO | Swin Transformer |
|------|-----|------------------|
| **核心操作** | 频域卷积 (FFT) + Channel MLP | 窗口注意力 (Window Attention) |
| **感受野** | 全局 | 局部 (8×8窗口) |
| **架构** | 扁平单尺度 | U-Net多尺度 |
| **参数量** | ~2.4M | ~100M+ |
| **时序建模** | 通道concat | Temporal Conv + Attention |
| **物理特性** | 频谱保持好 | 空域细节好 |
| **实现** | neuralop官方库 | 自定义实现 |

### FNO 架构流程 (使用neuralop)

```
输入: (B, 5, 12, 128, 128)
  ↓ Reshape
(B, 60, 128, 128)
  ↓ neuralop.FNO2d (all-in-one)
    ├─ Lifting: 60 → 64 (内部处理)
    ├─ FNO Block 1
    │   ├─ SpectralConv2d (FFT → 频域卷积 → iFFT)
    │   ├─ Channel MLP (跨通道混合)
    │   └─ Skip connection + GELU
    ├─ FNO Block 2
    ├─ FNO Block 3
    ├─ FNO Block 4
    └─ Projection: 64 → 12 (内部处理)
  ↓
(B, 12, 128, 128)
```

每个 FNO Block 包含：
1. **SpectralConv2d**:
   - FFT2D: 空域 → 频域
   - 频域卷积: 保留前16×16个modes，学习权重
   - iFFT2D: 频域 → 空域
2. **Channel MLP**: 1D卷积跨通道混合（学习u,v,w,p之间的耦合）
3. **Skip connection**: 软门控残差连接
4. **GELU激活**

## 核心技术特点

### 1. 频域卷积 (Spectral Convolution)
```python
# 伪代码
x_freq = FFT2D(x)  # 转到频域
x_freq_filtered = x_freq[:, :, :modes, :modes]  # 保留低频
y_freq = x_freq_filtered * learnable_weights  # 频域乘法
y = iFFT2D(y_freq)  # 转回空域
```

**优势**：
- 全局感受野（FFT天然覆盖整个空间）
- 参数高效（只学习有限的频域modes）
- 物理约束（频域操作更符合湍流的频谱特性）

### 2. 非周期边界处理
```python
# 使用replicate padding避免周期性假设
x = F.pad(x, (8, 8, 8, 8), mode='replicate')
```

### 3. 残差连接
```python
# 每层包含频域和空域双路径
x1 = spectral_conv(x)  # 频域路径
x2 = spatial_conv(x)   # 空域路径
x = gelu(x1 + x2)      # 残差连接
```

## 预期优势

1. **极致参数效率**: 2.4M vs 100M+ (Swin)，仅约1/40参数量
2. **计算效率**: FFT比多层attention更快，内存占用更少
3. **物理特性**: 频域操作更好保持能量谱
4. **长程依赖**: 全局感受野适合捕捉平面间耦合
5. **跨通道学习**: Channel MLP学习u,v,w,p之间的物理耦合
6. **标准实现**: 使用neuralop官方库，经过验证

## 使用方法

### 训练
```bash
# 1. 确保环境已配置
# 2. 运行训练脚本
./start_train_flow_fno_3plane_conda.sh
```

### 测试模型
```bash
# 运行简单测试
python test_fno_simple.py
```

### 评估
```bash
# 训练脚本会自动调用评估
# 或手动评估：
python evaluation_3plane.py logs/flow_fno_3plane/runs/<run_id>/checkpoints/last.ckpt --num_samples 3 --num_future 20
```

## 复用组件

以下组件无需修改，直接复用：
- ✓ 数据集: `FlowSequence3PlaneDataset`
- ✓ Lightning Module: `FlowSwin2DLitModule`
- ✓ 数据配置: `configs/data/flow_sequence_3plane/`
- ✓ 优化器: `configs/opt/default.yaml`
- ✓ 回调: `configs/callbacks/many_callbacks_flow_swin.yaml`
- ✓ 评估脚本: `evaluation_3plane.py`

## 超参数建议

当前配置（保守）:
```yaml
hidden_channels: 64
num_layers: 4
num_modes: [16, 16]
learning_rate: 0.0005
```

可调整方向：
- `hidden_channels`: 64 → 128（提升容量）
- `num_layers`: 4 → 6（增加深度）
- `num_modes`: [16, 16] → [24, 24]（保留更多频率）

## 注意事项

1. **数据归一化**: FNO对数据尺度敏感，确保使用正确的归一化统计
2. **边界条件**: 非周期边界使用replicate padding
3. **显存占用**: 虽然参数少，但FFT操作可能占用较多显存
4. **训练稳定性**: 初期loss可能震荡，属于正常现象

## 下一步

1. 运行训练对比FNO与Swin性能
2. 分析能量谱保持情况
3. 根据训练结果调整超参数
4. 如需要，可以尝试：
   - 添加频域损失（谱损失）
   - 调整modes数量
   - 实验不同的激活函数

## 参考文献

- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (2021)
- FNO适用于学习PDE算子，湍流预测本质上是学习Navier-Stokes方程的时间演化算子
