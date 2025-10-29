# 时间序列对比工具使用指南

## 简介

`compare_timeseries_results.py` 是一个用于对比多个模型在相同采样点的时间序列预测结果的工具。它可以从不同模型的 run 目录下读取评估结果，并生成对比图。

## 功能特性

- ✅ 支持对比多个模型（2个或更多）
- ✅ 自动加载 Ground Truth 和模型预测结果
- ✅ 支持 Teacher Forcing (TF) 和 Autoregressive (AR) 两种模式
- ✅ 支持所有采样点（27个点，分布在3个平面）
- ✅ 支持所有 4 个字段（u, v, w, p）
- ✅ 可选择性对比特定采样点
- ✅ GT 显示为黑色实线，模型预测显示为彩色虚线

## 快速开始

### 基本用法

```bash
# 对比三个模型的所有采样点（TF和AR模式）
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin"
```

### 指定模型名称

```bash
# 使用自定义模型名称
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/xxx \
    logs/flow_lstm_3plane/runs/yyy \
    --model-names "Model-A" "Model-B"
```

### 只对比特定采样点

```bash
# 只对比点 0, 9, 18
python compare_timeseries_results.py \
    run1 run2 run3 \
    --model-names "FNO" "LSTM" "Swin" \
    --points 0 9 18
```

### 指定输出目录

```bash
python compare_timeseries_results.py \
    run1 run2 \
    --model-names "Model-A" "Model-B" \
    --output-dir my_timeseries_comparison
```

### 限制显示的时间步数

```bash
# 只显示前50个时间步
python compare_timeseries_results.py \
    run1 run2 run3 \
    --model-names "FNO" "LSTM" "Swin" \
    --max-timesteps 50

# 只显示前20个时间步，只对比特定点
python compare_timeseries_results.py \
    run1 run2 \
    --model-names "Model-A" "Model-B" \
    --max-timesteps 20 \
    --points 0 9 18
```

### 只对比特定模式

```bash
# 只对比 Teacher Forcing 模式
python compare_timeseries_results.py \
    run1 run2 \
    --model-names "FNO" "LSTM" \
    --modes tf

# 只对比 Autoregressive 模式
python compare_timeseries_results.py \
    run1 run2 \
    --model-names "FNO" "LSTM" \
    --modes ar

# 对比两种模式（默认）
python compare_timeseries_results.py \
    run1 run2 \
    --model-names "FNO" "LSTM" \
    --modes tf ar
```

## 输出说明

### 文件结构

程序会在指定的输出目录（默认 `timeseries_comparison_results/`）生成对比图：

```
timeseries_comparison_results/
├── timeseries_point0_plane0_y29_z40_x40_tf.png
├── timeseries_point0_plane0_y29_z40_x40_ar.png
├── timeseries_point1_plane0_y29_z40_x80_tf.png
├── timeseries_point1_plane0_y29_z40_x80_ar.png
├── ...
├── timeseries_point26_plane2_y75_z104_x104_tf.png
└── timeseries_point26_plane2_y75_z104_x104_ar.png
```

**文件命名规则**:
- `timeseries_point<点号>_plane<平面号>_y<y坐标>_z<z坐标>_x<x坐标>_<模式>.png`
- 模式: `tf` (Teacher Forcing) 或 `ar` (Autoregressive)

### 图表说明

每张图包含 **2×2 = 4 个子图**，对应 4 个流场字段：

```
┌─────────────────────────────────┐
│  u (左上)      │  v (右上)      │
│  ───────────   │  ───────────   │
│  ▬ GT          │  ▬ GT          │
│  ┈ Model-A     │  ┈ Model-A     │
│  ┈ Model-B     │  ┈ Model-B     │
│  ┈ Model-C     │  ┈ Model-C     │
├─────────────────┼─────────────────┤
│  w (左下)      │  p (右下)      │
│  ───────────   │  ───────────   │
│  ▬ GT          │  ▬ GT          │
│  ┈ Model-A     │  ┈ Model-A     │
│  ┈ Model-B     │  ┈ Model-B     │
│  ┈ Model-C     │  ┈ Model-C     │
└─────────────────┴─────────────────┘
```

**视觉设计**:
- **Ground Truth**: 黑色实线 (`'k-'`, linewidth=2, alpha=0.8)
- **模型预测**: 彩色虚线 (`'--'`, linewidth=1.5, alpha=0.7)
- **X轴**: Timestep (时间步)
- **Y轴**: 对应字段的值 (U, V, W, P)
- **网格**: 浅色虚线 (alpha=0.3)
- **图例**: 自动放置在最佳位置

## 完整示例

假设你有以下三个模型的评估结果：
- FNO: `logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052`
- LSTM: `logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652`
- Swin: `logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845`

### 示例 1: 对比所有采样点

```bash
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin"
```

**输出**:
```
============================================================
开始时间序列对比
============================================================

找到 27 个采样点

============================================================
模式: Teacher Forcing
============================================================

对比点 0 (Plane0, y=29, z=40, x=40) - Teacher Forcing
  ✓ 保存: timeseries_comparison_results/timeseries_point0_plane0_y29_z40_x40_tf.png

...（共 27 个点）

============================================================
模式: Autoregressive
============================================================

对比点 0 (Plane0, y=29, z=40, x=40) - Autoregressive
  ✓ 保存: timeseries_comparison_results/timeseries_point0_plane0_y29_z40_x40_ar.png

...（共 27 个点）

============================================================
✅ 对比完成！
============================================================
  生成图片数量: 54
  输出目录: timeseries_comparison_results

对比的配置:
  - 采样点: 27 个
  - 模式: tf, ar
  - 模型: 3 个 (FNO, LSTM, Swin)
  - 时间步: 全部
```

### 示例 2: 只对比特定的点

```bash
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin" \
    --points 0 9 18 \
    --output-dir selected_points
```

**输出**: 只生成 3 个点 × 2 种模式 = 6 张图片

### 示例 3: 只对比 Teacher Forcing 模式

```bash
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin" \
    --modes tf \
    --output-dir tf_only
```

**输出**: 只生成 27 个点 × 1 种模式 = 27 张图片

### 示例 5: 只显示部分时间步

```bash
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin" \
    --max-timesteps 50 \
    --output-dir timeseries_50steps
```

**输出**: 生成 54 张图片，每张图只显示前 50 个时间步

**配置信息**:
```
对比的配置:
  - 采样点: 27 个
  - 模式: tf, ar
  - 模型: 3 个 (FNO, LSTM, Swin)
  - 时间步: 前 50 步
```

**使用场景**:
- 当完整时间序列太长时，可以只关注前面的时间步
- 对比模型在短期预测上的差异
- 生成更清晰、更易读的对比图（时间步太多会导致曲线拥挤）

## 数据要求

程序需要以下文件存在于评估结果目录中：

```
logs/<model_type>/runs/<run_timestamp>/
└── evaluation_results/
    └── time_series_data/
        ├── time_series_test_tf.csv    # Teacher Forcing 模式
        └── time_series_test_ar.csv    # Autoregressive 模式
```

**CSV 文件格式**:

每个 CSV 文件包含以下列：

- `timestep`: 时间步（列索引）
- `<field>_gt_point<p>_plane<n>_y<y>_z<z>_x<x>`: Ground Truth
- `<field>_pred_point<p>_plane<n>_y<y>_z<z>_x<x>`: 模型预测

其中：
- `<field>`: u, v, w, p
- `<p>`: 采样点编号（0-26）
- `<n>`: 平面编号（0, 1, 2）
- `<y>`, `<z>`, `<x>`: 空间坐标

**示例列名**:
- `u_gt_point0_plane0_y29_z40_x40`
- `u_pred_point0_plane0_y29_z40_x40`
- `v_gt_point9_plane1_y54_z64_x64`
- `v_pred_point9_plane1_y54_z64_x64`

**注意**:
- Ground Truth 只从第一个结果目录读取（假设所有模型使用相同的测试数据）
- 如果某个文件缺失，程序会显示警告并跳过该部分

## 采样点说明

评估脚本在 3 个平面上每个平面采样 9 个点，共 27 个点：

| 平面 | Y坐标 | 采样点编号 | Z坐标范围 | X坐标范围 |
|------|-------|------------|-----------|-----------|
| 0    | 29    | 0-8        | 40-104    | 40-104    |
| 1    | 54    | 9-17       | 40-104    | 40-104    |
| 2    | 75    | 18-26      | 40-104    | 40-104    |

每个平面的 9 个点均匀分布在 3×3 网格上（间隔 32 像素）。

## 故障排查

### 问题 1: 目录不存在

```
❌ 错误: Run目录不存在: logs/xxx/runs/yyy
```

**解决**: 确保指定的 run 目录路径正确

### 问题 2: 时间序列数据目录不存在

```
❌ 错误: 时间序列数据目录不存在: logs/xxx/runs/yyy/evaluation_results/time_series_data
```

**解决**: 确保已经运行过评估脚本：
```bash
python evaluation_3plane.py <checkpoint_path>
```

### 问题 3: CSV 文件缺失

```
⚠ 文件不存在: .../time_series_test_tf.csv
```

**解决**: 这是警告，不是错误。如果缺少该模式的数据，对应的图会被跳过。

### 问题 4: 模型名称数量不匹配

```
error: 模型名称数量 (2) 必须与目录数量 (3) 匹配
```

**解决**: 确保 `--model-names` 的参数数量与输入目录数量一致

### 问题 5: 无法获取采样点信息

```
❌ 无法获取采样点信息
```

**解决**: 检查所有输入目录的 CSV 文件是否存在且格式正确

## 技术细节

### 列名解析

程序使用正则表达式解析 CSV 列名：

```python
pattern = r'([uvwp])_(pred|gt)_point(\d+)_plane(\d+)_y(\d+)_z(\d+)_x(\d+)'
```

**示例**:
- 输入: `u_pred_point0_plane0_y29_z40_x40`
- 输出: `{'field': 'u', 'type': 'pred', 'point': 0, 'plane': 0, 'y': 29, 'z': 40, 'x': 40}`

### 对比逻辑

1. 从第一个有效数据源获取所有采样点信息
2. 对每个点，加载所有模型的 TF/AR 数据
3. 创建 2×2 子图布局（4 个字段）
4. 首先绘制 Ground Truth（黑色实线）
5. 然后绘制各个模型的预测（彩色虚线）
6. 保存为 PNG 文件（dpi=150）

### 可视化配置

```python
# Ground Truth
ax.plot(timesteps, gt_data, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)

# 模型预测
colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
ax.plot(timesteps, pred_data, '--', linewidth=1.5, color=colors[i],
        label=model_name, alpha=0.7)

# 网格
ax.grid(True, alpha=0.3, linestyle=':')
```

## Teacher Forcing vs Autoregressive

### Teacher Forcing (TF) 模式

- 使用 Ground Truth 作为输入
- 逐步预测：每次预测使用真实的历史数据
- **优点**: 更稳定，误差不累积
- **缺点**: 不反映模型的真实自回归性能

### Autoregressive (AR) 模式

- 使用模型自身的预测作为输入
- 自回归预测：每次预测使用之前的预测结果
- **优点**: 反映模型的真实长期预测能力
- **缺点**: 误差会累积，可能发散

**建议**:
- 开发阶段：主要关注 TF 模式，快速评估模型性能
- 最终评估：关注 AR 模式，评估长期预测稳定性

## 扩展功能（未来）

以下功能可以根据需要添加：

- [ ] 添加量化指标（MSE, MAE, 相关系数）到图中
- [ ] 支持时间序列动画（GIF/MP4）
- [ ] 添加统计置信区间
- [ ] 导出对比数据到 CSV
- [ ] 支持交互式可视化（plotly）
- [ ] 添加频谱分析对比

## 与能量谱对比工具的区别

| 特性 | 时间序列对比 | 能量谱对比 |
|------|--------------|------------|
| **数据源** | CSV 文件 | NPY 文件 |
| **对比内容** | 时间演化 | 空间频率 |
| **采样点** | 27 个点 | 整个平面 |
| **模式** | TF/AR 分离 | 无模式区分 |
| **图表类型** | 折线图 | 1D谱 + 2D谱 |
| **输出数量** | 27 × 2 = 54 张 | 3 × 4 = 12 张 |

**使用建议**:
- **能量谱**: 宏观评估模型在不同尺度上的表现
- **时间序列**: 微观评估模型在特定位置的时间演化精度

## 相关文件

- [evaluation_3plane.py](evaluation_3plane.py): 原始评估脚本（生成时间序列数据）
- [compare_timeseries_results.py](compare_timeseries_results.py): 本对比工具
- [example_compare_timeseries.sh](example_compare_timeseries.sh): 示例脚本
- [compare_spectrum_results.py](compare_spectrum_results.py): 能量谱对比工具
- [SPECTRUM_COMPARISON_README.md](SPECTRUM_COMPARISON_README.md): 能量谱对比工具文档

## 联系与反馈

如有问题或建议，请通过项目 issue 反馈。
