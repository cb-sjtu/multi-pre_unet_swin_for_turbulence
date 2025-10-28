#!/bin/bash
# 时间序列对比示例脚本

echo "=========================================="
echo "时间序列对比工具 - 使用示例"
echo "=========================================="
echo ""

# 定义三个模型的run目录
FNO_RUN="logs/flow_fno_3plane/runs/2025-10-27_22-56-39-791052"
LSTM_RUN="logs/flow_lstm_3plane/runs/2025-10-26_12-14-53-336652"
SWIN_RUN="logs/flow_swin_3plane/runs/2025-09-22_11-09-35-088845"

# 检查目录是否存在
missing_dirs=0
for dir in "$FNO_RUN" "$LSTM_RUN" "$SWIN_RUN"; do
    if [ ! -d "$dir/evaluation_results/time_series_data" ]; then
        echo "❌ 时间序列数据目录不存在: $dir/evaluation_results/time_series_data"
        missing_dirs=$((missing_dirs + 1))
    fi
done

if [ $missing_dirs -gt 0 ]; then
    echo ""
    echo "请先运行评估脚本生成时间序列数据："
    echo "  python evaluation_3plane.py <checkpoint_path>"
    exit 1
fi

echo "✓ 找到所有模型的时间序列数据"
echo ""

# 示例 1: 对比所有采样点（TF和AR模式）
echo "=========================================="
echo "示例 1: 对比所有采样点（TF + AR 模式）"
echo "=========================================="
python compare_timeseries_results.py \
    "$FNO_RUN" \
    "$LSTM_RUN" \
    "$SWIN_RUN" \
    --model-names "FNO" "LSTM" "Swin" \
    --output-dir timeseries_comparison_all

echo ""
echo "✓ 结果保存在: timeseries_comparison_all/"
echo "  (共 27 个点 × 2 种模式 = 54 张图片)"
echo ""

# 示例 2: 只对比特定的点
echo "=========================================="
echo "示例 2: 只对比特定的点"
echo "=========================================="
python compare_timeseries_results.py \
    "$FNO_RUN" \
    "$LSTM_RUN" \
    "$SWIN_RUN" \
    --model-names "FNO" "LSTM" "Swin" \
    --points 0 9 18 \
    --output-dir timeseries_comparison_selected

echo ""
echo "✓ 结果保存在: timeseries_comparison_selected/"
echo "  (共 3 个点 × 2 种模式 = 6 张图片)"
echo ""

# 示例 3: 只对比 Teacher Forcing 模式
echo "=========================================="
echo "示例 3: 只对比 Teacher Forcing 模式"
echo "=========================================="
python compare_timeseries_results.py \
    "$FNO_RUN" \
    "$LSTM_RUN" \
    "$SWIN_RUN" \
    --model-names "FNO" "LSTM" "Swin" \
    --modes tf \
    --output-dir timeseries_comparison_tf_only

echo ""
echo "✓ 结果保存在: timeseries_comparison_tf_only/"
echo "  (共 27 个点 × 1 种模式 = 27 张图片)"
echo ""

# 示例 4: 只对比 Autoregressive 模式
echo "=========================================="
echo "示例 4: 只对比 Autoregressive 模式"
echo "=========================================="
python compare_timeseries_results.py \
    "$FNO_RUN" \
    "$LSTM_RUN" \
    "$SWIN_RUN" \
    --model-names "FNO" "LSTM" "Swin" \
    --modes ar \
    --output-dir timeseries_comparison_ar_only

echo ""
echo "✓ 结果保存在: timeseries_comparison_ar_only/"
echo "  (共 27 个点 × 1 种模式 = 27 张图片)"
echo ""

echo "=========================================="
echo "✅ 所有示例完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  ls -lh timeseries_comparison_*/"
echo ""
echo "或打开图片查看器:"
echo "  eog timeseries_comparison_all/*.png &"
