# 能量谱对比工具使用指南

## 简介

`compare_spectrum_results.py` 是一个用于对比多个模型能量谱（Energy Spectrum）结果的工具。它可以从 `evaluation_results/` 目录下读取不同模型的评估结果，并生成对比图。

## 功能特性

- ✅ 支持对比多个模型（2个或更多）
- ✅ 自动加载 Ground Truth 和模型预测结果
- ✅ 支持 3 种对比模式：
  - **combined**: 组合图（1D kx + 1D kz + 2D 误差）
  - **1d**: 仅 1D 能量谱（kx 和 kz）
  - **2d**: 仅 2D 能量谱
  - **all**: 生成所有类型的图
- ✅ 支持所有 3 个平面（plane 0, 1, 2）
- ✅ 支持所有 4 个字段（u, v, w, p）
- ✅ 自动添加 Kolmogorov k^-5/3 参考线

## 快速开始

### 基本用法

```bash
# 对比两个模型（使用目录名作为模型名）
python compare_spectrum_results.py \
    evaluation_results_2025-10-27_22-56-39-791052 \
    evaluation_results_2025-10-26_12-14-53-336652
```

### 指定模型名称

```bash
# 使用自定义模型名称
python compare_spectrum_results.py \
    evaluation_results_2025-10-27_22-56-39-791052 \
    evaluation_results_2025-10-26_12-14-53-336652 \
    --model-names "LSTM" "Swin Transformer"
```

### 对比三个或更多模型

```bash
python compare_spectrum_results.py \
    dir1 dir2 dir3 \
    --model-names "Model-A" "Model-B" "Model-C"
```

### 指定输出目录

```bash
python compare_spectrum_results.py \
    dir1 dir2 \
    --model-names "LSTM" "Swin" \
    --output-dir my_comparison_results
```

### 选择对比模式

```bash
# 生成组合图（默认，推荐）
python compare_spectrum_results.py dir1 dir2 --mode combined

# 只生成 1D 谱对比
python compare_spectrum_results.py dir1 dir2 --mode 1d

# 只生成 2D 谱对比
python compare_spectrum_results.py dir1 dir2 --mode 2d

# 生成所有类型的图
python compare_spectrum_results.py dir1 dir2 --mode all
```

## 输出说明

### 文件结构

程序会在指定的输出目录（默认 `spectrum_comparison_results/`）生成对比图：

```
spectrum_comparison_results/
├── comparison_combined_plane0_u.png
├── comparison_combined_plane0_v.png
├── comparison_combined_plane0_w.png
├── comparison_combined_plane0_p.png
├── comparison_combined_plane1_u.png
├── ...
├── comparison_1d_plane0_u_kx.png     # --mode 1d 或 all
├── comparison_1d_plane0_u_kz.png     # --mode 1d 或 all
├── comparison_2d_plane0_u.png        # --mode 2d 或 all
└── ...
```

### 图表说明

#### 1. Combined 模式（推荐）

**新版布局（3行结构）**：

每张图包含 3 行子图，完整展示所有模型的对比：

**第 1 行**：基础对比
- **子图 1**: 1D 能量谱（kx 方向）
  - 黑色实线：Ground Truth
  - 彩色虚线：各个模型的预测
- **子图 2**: 1D 能量谱（kz 方向）
  - 同上
- **子图 3**: Ground Truth 的 2D 能量谱
  - 完整的真值 2D 能量分布

**第 2 行**：各模型的 2D 预测
- 并排显示所有模型的 2D 能量谱预测
- **使用 Ground Truth 的色标范围**：与真值使用相同的值域，可以直观看到模型预测是否偏离真值范围
- 使用 log10(Energy) 显示

**第 3 行**：各模型的相对误差
- 并排显示所有模型与 Ground Truth 的相对误差
- **统一色标**：所有模型使用相同的误差范围，方便直接对比哪个模型更准确
- 使用 log10(Relative Error) 显示
- **颜色直观**：浅色（白/黄）= 小误差（好），深色（红/黑）= 大误差（差）

**示例布局（3个模型的情况）**：
```
┌────────────────────────────────────────────────────────────────┐
│ 第1行：基础对比                                                │
│ ┌──────────┬──────────┬──────────┐                            │
│ │ 1D (kx)  │ 1D (kz)  │   GT 2D  │                            │
│ │  ▬ GT    │  ▬ GT    │          │                            │
│ │  ┈ FNO   │  ┈ FNO   │          │                            │
│ │  ┈ LSTM  │  ┈ LSTM  │          │                            │
│ │  ┈ Swin  │  ┈ Swin  │          │                            │
│ └──────────┴──────────┴──────────┘                            │
├────────────────────────────────────────────────────────────────┤
│ 第2行：模型预测 (统一色标)                                    │
│ ┌──────────┬──────────┬──────────┐                            │
│ │ FNO 2D   │ LSTM 2D  │ Swin 2D  │                            │
│ │ Pred.    │ Pred.    │ Pred.    │                            │
│ └──────────┴──────────┴──────────┘                            │
├────────────────────────────────────────────────────────────────┤
│ 第3行：相对误差 (统一色标 - 方便对比)                         │
│ ┌──────────┬──────────┬──────────┐                            │
│ │ FNO      │ LSTM     │ Swin     │                            │
│ │ Rel.Err  │ Rel.Err  │ Rel.Err  │                            │
│ └──────────┴──────────┴──────────┘                            │
└────────────────────────────────────────────────────────────────┘
```

**关键优势**：
- ✅ 一张图看到所有信息：1D谱、2D谱、真值、预测、误差
- ✅ 统一色标让不同模型直接可比
- ✅ 相对误差使用相同范围，一目了然哪个模型更好

#### 2. 1D 模式

- 对比不同模型在 kx 和 kz 方向的 1D 能量谱
- 包含 k^-5/3 Kolmogorov 参考线

#### 3. 2D 模式

- 并排显示 Ground Truth 和各个模型的 2D 能量谱
- 使用 log10 刻度显示

## 完整示例

假设你有以下评估结果：
- `evaluation_results_2025-10-27_22-56-39-791052` (LSTM 模型)
- `evaluation_results_2025-10-26_12-14-53-336652` (Swin 模型)

### 示例 1: 快速对比

```bash
python compare_spectrum_results.py \
    evaluation_results_2025-10-27_22-56-39-791052 \
    evaluation_results_2025-10-26_12-14-53-336652 \
    --model-names "LSTM" "Swin"
```

**输出**:
```
初始化能量谱对比器
  模型数量: 2
  模型名称: LSTM, Swin
  输出目录: spectrum_comparison_results

============================================================
开始能量谱对比 (模式: combined)
============================================================

对比组合谱: Plane0 - u
  ✓ 保存: spectrum_comparison_results/comparison_combined_plane0_u.png

...（共 12 张图：3 planes × 4 fields）

============================================================
✅ 对比完成！
============================================================
  生成图片数量: 12
  输出目录: spectrum_comparison_results

对比的配置:
  - 平面: 3 个 (plane 0, 1, 2)
  - 字段: 4 个 (u, v, w, p)
  - 模型: 2 个 (LSTM, Swin)
```

### 示例 2: 生成所有类型的图

```bash
python compare_spectrum_results.py \
    evaluation_results_2025-10-27_22-56-39-791052 \
    evaluation_results_2025-10-26_12-14-53-336652 \
    --model-names "LSTM" "Swin" \
    --mode all \
    --output-dir detailed_comparison
```

这将生成：
- 12 张 combined 图
- 24 张 1D 图（12 × kx + 12 × kz）
- 12 张 2D 图
- **总计**: 48 张图

## 数据要求

程序需要以下文件存在于评估结果目录中：

```
evaluation_results/<result_dir>/
├── spectrum_ground_truth_plane0_u_kx.npy
├── spectrum_ground_truth_plane0_u_kz.npy
├── spectrum_ground_truth_plane0_u_2d.npy
├── spectrum_prediction_plane0_u_kx.npy
├── spectrum_prediction_plane0_u_kz.npy
├── spectrum_prediction_plane0_u_2d.npy
└── ... (所有 plane × field 的组合)
```

**注意**:
- Ground Truth 只从第一个结果目录读取（假设所有模型使用相同的测试数据）
- 如果某个文件缺失，程序会显示警告并跳过该部分

## 故障排查

### 问题 1: 目录不存在

```
❌ 错误: 结果目录不存在: evaluation_results/xxx
```

**解决**: 确保指定的目录名称正确，且位于 `evaluation_results/` 下

### 问题 2: 文件缺失

```
⚠ 文件不存在: spectrum_prediction_plane0_u_kx.npy
```

**解决**: 这是警告，不是错误。如果缺少关键文件，对应的图会被跳过。

### 问题 3: 模型名称数量不匹配

```
error: 模型名称数量 (2) 必须与目录数量 (3) 匹配
```

**解决**: 确保 `--model-names` 的参数数量与输入目录数量一致

## 技术细节

### 数据格式

- **1D 谱**: shape = (128,)，表示不同波数的能量
- **2D 谱**: shape = (128, 128)，表示 kx-kz 平面的能量分布

### 可视化说明

- 所有能量谱使用 **log-log 刻度**
- 2D 谱使用 **log10(Energy)** 显示
- 颜色映射：viridis（能量谱）、hot（误差）

## 扩展功能（未来）

以下功能可以根据需要添加：

- [ ] 添加量化指标（MSE, MAE, 相关系数）
- [ ] 支持时间序列动画
- [ ] 添加统计置信区间
- [ ] 导出对比数据到 CSV
- [ ] 支持更多能量谱类型

## 相关文件

- `evaluation_3plane.py`: 原始评估脚本
- `evaluation_3plane_new.py`: 新版模块化评估脚本
- `compare_spectrum_results.py`: 本对比工具

## 联系与反馈

如有问题或建议，请通过项目 issue 反馈。
