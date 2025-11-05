# FNO 1-Plane Implementation Summary

## 概述

已成功将FNO 3平面模型（12通道，128×128）转换为FNO 1平面模型（3通道，256×256），用于学习yslice54的uvw湍流数据演化。

---

## 创建的文件清单

### 1. 数据集实现
**文件**: `src/datasets/flow_sequence_2d/flow_sequence_1plane.py`
- **功能**: 单平面数据集加载器
- **通道**: 3 (u, v, w)
- **分辨率**: 256×256
- **Y平面**: 54
- **数据源**: `u-v-w_scale2-3-1_yslice54_t*.h5`

### 2. 数据配置文件
**目录**: `configs/data/flow_sequence_1plane/`

#### 2.1 主配置
**文件**: `flow_sequence_1plane.yaml`
```yaml
data_folder: /home/sh/CB/icon-thewell-dev/data/preprocessed_flow/
batch_size_per_device: 10  # 可以比3平面更大（3通道 vs 12通道）
enable_normalization: true
norm_stats: "norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json"
```

#### 2.2 训练/验证/测试配置
**文件**:
- `train/flow_1plane_train.yaml`
- `valid/flow_1plane_valid.yaml`
- `test/flow_1plane_test.yaml`

**关键参数**:
```yaml
field_names: ["u", "v", "w"]  # 3个通道
file_pattern: "*u-v-w_scale2-3-1_yslice54_*.h5"
resolution_scale: [2, 3, 1]
y_slice: 54
train_ratio: 0.9
valid_ratio: 0.05
test_ratio: 0.05
```

### 3. 模型配置
**文件**: `configs/model/fno_1plane.yaml`
```yaml
_target_: src.models.nop.fno_2d_temporal.FNO2DTemporalModel
input_shape: [256, 256]  # 全256×256分辨率
num_channels: 3  # u, v, w
num_modes: [32, 32]  # 保持1/8频率比例
hidden_channels: 64
num_layers: 4
```

### 4. 训练配置
**文件**: `configs/train_flow_fno_1plane.yaml`
```yaml
task_name: "flow_fno_1plane"
tags: ["flow", "fno", "1plane", "uvw", "3channels", "spectral", "256x256", "yslice54"]
log_project: "turbulence_fno_1plane"
enable_per_channel_metrics: true  # 3通道监控

k_step_rollout:
  enabled: true
  max_k_steps: 16
  curriculum_k_values: [1, 1, 1, 1, 1]
```

### 5. 评估脚本
**文件**:
- `evaluation_1plane.py` - 标准评估脚本（1701行）
- `evaluation_1plane_new.py` - 增强模块化评估脚本

**功能**:
- Autoregressive预测
- Teacher forcing评估
- 3通道可视化（u, v, w）
- 9个监控点（3×3网格）
- 能量谱分析
- 误差分析（MSE, MAE, RMS-relative）
- WandB集成

### 6. 训练启动脚本
**文件**: `start_train_flow_fno_1plane_conda.sh`
- 执行训练（50000步）
- 自动找到最新checkpoint
- 运行评估
- 整合结果到run目录

---

## 关键参数对比

| 参数 | 3-Plane | 1-Plane | 变化 |
|------|---------|---------|------|
| **通道数** | 12 | 3 | -75% |
| **分辨率** | 128×128 | 256×256 | +4x |
| **Y平面** | [29,54,75] | 54 | 单平面 |
| **物理场** | [u,v,w,p] | [u,v,w] | 无压力 |
| **Fourier模式** | [16,16] | [32,32] | +2x |
| **批次大小** | 6 | 10 | +67% |
| **数据文件** | `*u-v-w-p_scale4-6-1_yslice*.h5` | `*u-v-w_scale2-3-1_yslice54_*.h5` | 专用 |
| **标准化统计** | `norm_stats_12ch_3plane...` | `norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json` | 3通道 |
| **分割比例** | 0.7/0.15/0.15 | 0.9/0.05/0.05 | 更多训练数据 |

---

## 使用方法

### 快速开始

```bash
# 1. 训练模型
bash start_train_flow_fno_1plane_conda.sh

# 或手动训练
python src/train.py --config-name=train_flow_fno_1plane \
    trainer.max_steps=50000 \
    trainer.val_check_interval=300 \
    trainer.limit_val_batches=5

# 2. 评估模型
python evaluation_1plane.py <checkpoint_path> --num_samples 3 --num_future 10

# 或使用增强版
python evaluation_1plane_new.py <checkpoint_path> --num_samples 3 --num_future 100
```

### 自定义训练

```bash
# 修改批次大小
python src/train.py --config-name=train_flow_fno_1plane \
    data.batch_size_per_device=12

# 修改最大K步
python src/train.py --config-name=train_flow_fno_1plane \
    k_step_rollout.max_k_steps=20

# 修改Fourier模式（如果内存不足）
python src/train.py --config-name=train_flow_fno_1plane \
    model.num_modes=[16,16]
```

---

## 数据验证

### 数据文件信息
- **总文件数**: 1,581个时间步
- **时间范围**: t00001 to t01581
- **文件大小**: ~718KB per file
- **数据形状**: (3, 256, 256) per file
- **数据类型**: float32

### 标准化统计
**文件**: `data/preprocessed_flow/norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json`

```json
{
  "per_channel_stats": {
    "u": {"mean": 0.8178983, "std": 0.1126831},
    "v": {"mean": 1.007e-08, "std": 0.05485137},
    "w": {"mean": 0.001271626, "std": 0.07485994}
  },
  "field_names": ["u", "v", "w"],
  "y_slice": 54,
  "resolution_scale": [2, 3, 1]
}
```

### 不连续性处理
- **不连续位置**: timestep 1081
- **自动排除**: 跨越不连续性的序列自动过滤
- **有效样本**: ~1,570 (排除约11个序列)

---

## 模型架构

### FNO2DTemporalModel
```
Input: (B, T=5, C=3, H=256, W=256)
  ↓ Flatten temporal: (B, 15, 256, 256)
  ↓ Lifting: 15 → 128 channels
  ↓ FNO Layers (4层):
      - Spectral convolution in frequency domain
      - Modes: [32, 32] in [H, W]
      - Skip connections with soft-gating
  ↓ Projection: 128 → 3 channels
Output: (B, 3, 256, 256)
```

**特点**:
- 全局感受野（频域卷积）
- 参数量与分辨率无关
- 高效捕捉长程依赖

---

## 训练策略

### K-step Rollout
- **初始K**: 1步
- **最大K**: 16步
- **Curriculum**: 随epoch增加
- **步权重**: [1.0, 1.2, 1.5, 2.0, 2.5]（缓解梯度稀释）

### 验证策略
- **验证间隔**: 每5个epoch
- **验证批次**: 5个batches
- **Early stopping**: 监控验证损失

### 优化器
- 使用默认配置（AdamW）
- Learning rate schedule
- Gradient clipping

---

## 输出结构

### 训练输出
```
logs/flow_fno_1plane/runs/<timestamp>/
├── checkpoints/
│   ├── last.ckpt
│   └── best.ckpt
├── wandb/
│   └── <wandb_logs>
└── evaluation_results/
    ├── plots/
    ├── videos/
    └── metrics/
```

### WandB日志
- **项目名**: turbulence_fno_1plane
- **Tags**: flow, fno, 1plane, uvw, 3channels, 256x256, yslice54
- **Metrics**:
  - 3通道MSE/MAE (u, v, w)
  - K-step rollout loss
  - Energy spectra
  - 监控点时间序列

---

## 性能预期

### 内存使用
- **3平面**: 12通道 × 128×128 = ~786K elements per sample
- **1平面**: 3通道 × 256×256 = ~786K elements per sample
- **结论**: 内存使用相当，但1平面分辨率更高

### 训练时间
- **每个epoch**: ~5-10分钟（取决于硬件）
- **50 epochs**: ~4-8小时
- **50000 steps**: 取决于数据集大小

### Batch Size建议
- **GPU 24GB**: batch_size=10-12
- **GPU 40GB**: batch_size=16-20
- **GPU 80GB**: batch_size=24-32

---

## 故障排查

### 常见问题

#### 1. 数据加载失败
```
ValueError: No files found with pattern...
```
**解决**: 检查数据路径和文件pattern是否正确

#### 2. 标准化统计文件缺失
```
Warning: Could not load normalization stats...
```
**解决**: 确认文件存在于 `data/preprocessed_flow/norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json`

#### 3. OOM (Out of Memory)
```
CUDA out of memory...
```
**解决**:
- 减小batch_size
- 减小num_modes: [32,32] → [16,16]
- 减小hidden_channels: 64 → 48

#### 4. 模型加载失败
```
KeyError or RuntimeError during checkpoint loading
```
**解决**: 检查模型配置与checkpoint是否匹配

---

## 下一步

### 实验建议
1. **基准训练**: 使用默认配置训练50 epochs
2. **超参数调优**:
   - 尝试不同的num_modes
   - 调整learning rate
   - 实验不同的K-step curriculum
3. **长期预测**: 增加num_future_steps到200-500步
4. **模型对比**: 与Swin Transformer等模型对比

### 扩展可能
1. **多尺度融合**: 集成MR-PC策略
2. **物理约束**: 添加物理损失（散度自由等）
3. **数据增强**: 空间旋转、翻转
4. **集成学习**: 多模型ensemble

---

## 参考文件

### 原始3平面实现
- `evaluation_3plane.py` - 参考评估脚本
- `configs/train_flow_fno_3plane.yaml` - 参考训练配置
- `src/datasets/flow_sequence_2d/flow_sequence_3plane.py` - 参考数据集

### 相关文档
- `FUSION_INTERVAL_README.md` - MR-PC融合间隔功能
- `FIRST_FUSION_POINT_README.md` - MR-PC第一个融合点功能
- `VIDEO_GENERATION_README.md` - 视频生成功能

---

## 总结

✅ **完成项目**:
1. 数据集类 (`FlowSequence1PlaneDataset`)
2. 完整配置文件系统
3. 模型配置 (FNO 1-plane)
4. 训练配置
5. 双评估脚本
6. 训练启动脚本

✅ **验证通过**:
- 数据文件存在且格式正确
- 标准化统计文件可用
- 配置参数匹配数据特征

🚀 **可立即使用**:
```bash
bash start_train_flow_fno_1plane_conda.sh
```

---

*Created: 2025-11-05*
*Author: Claude (with user guidance)*
*Project: icon-thewell-dev*
