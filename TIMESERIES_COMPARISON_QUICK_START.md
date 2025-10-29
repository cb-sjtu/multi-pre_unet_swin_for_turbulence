# 时间序列对比工具 - 快速使用指南

## 最常用的命令

### 1. 对比所有采样点（TF + AR）

```bash
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin"
```

### 2. 只显示前50个时间步

```bash
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin" \
    --max-timesteps 50
```

### 3. 只对比特定的点

```bash
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin" \
    --points 0 9 18
```

### 4. 只对比 AR 模式

```bash
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin" \
    --modes ar
```

### 5. 组合使用（推荐）

```bash
# 对比特定点，只看AR模式，只显示前30步
python compare_timeseries_results.py \
    logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052 \
    logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652 \
    logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845 \
    --model-names "FNO" "LSTM" "Swin" \
    --points 0 9 18 \
    --modes ar \
    --max-timesteps 30 \
    --output-dir timeseries_ar_30steps
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `run_dirs` | 模型run目录（必需） | `logs/flow_fno_3plane/runs/xxx` |
| `--model-names` | 模型名称列表 | `--model-names "FNO" "LSTM" "Swin"` |
| `--max-timesteps` | **新增** 最大显示时间步数 | `--max-timesteps 50` |
| `--points` | 指定采样点 | `--points 0 9 18` |
| `--modes` | 指定模式 | `--modes tf` 或 `--modes ar` |
| `--output-dir` | 输出目录 | `--output-dir my_results` |

## 查看帮助

```bash
python compare_timeseries_results.py --help
```

## 常见场景

### 场景1：快速预览（推荐用于首次查看）
```bash
# 只看3个代表性的点，前20步，只看TF模式
python compare_timeseries_results.py \
    run1 run2 run3 \
    --model-names "M1" "M2" "M3" \
    --points 0 13 26 \
    --max-timesteps 20 \
    --modes tf
```
输出：3张图

### 场景2：详细对比AR长期预测能力
```bash
# 所有点，AR模式，全部时间步
python compare_timeseries_results.py \
    run1 run2 run3 \
    --model-names "M1" "M2" "M3" \
    --modes ar
```
输出：27张图

### 场景3：短期预测性能对比
```bash
# 所有点，TF+AR，只看前10步
python compare_timeseries_results.py \
    run1 run2 run3 \
    --model-names "M1" "M2" "M3" \
    --max-timesteps 10
```
输出：54张图（每张只显示前10步）

## 输出说明

- 每个采样点生成1张图（包含4个子图：u, v, w, p）
- 文件命名：`timeseries_point<N>_plane<P>_y<Y>_z<Z>_x<X>_<mode>.png`
- Ground Truth: 黑色实线
- 模型预测: 彩色虚线

## 完整文档

详细文档请参考：[TIMESERIES_COMPARISON_README.md](TIMESERIES_COMPARISON_README.md)
