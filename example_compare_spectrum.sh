#!/bin/bash
# 能量谱对比示例脚本

echo "=========================================="
echo "能量谱对比工具 - 使用示例"
echo "=========================================="
echo ""

# 检查是否有足够的评估结果
result_dirs=(evaluation_results/evaluation_results_*)
num_dirs=${#result_dirs[@]}

if [ $num_dirs -lt 2 ]; then
    echo "❌ 错误: 需要至少 2 个评估结果目录"
    echo "   当前找到: $num_dirs 个"
    echo ""
    echo "请先运行评估脚本生成结果："
    echo "  python evaluation_3plane.py <checkpoint_path>"
    exit 1
fi

echo "✓ 找到 $num_dirs 个评估结果目录"
echo ""

# 获取最近的两个目录
dir1=$(basename "${result_dirs[-1]}")
dir2=$(basename "${result_dirs[-2]}")

echo "将对比以下两个结果:"
echo "  1. $dir1"
echo "  2. $dir2"
echo ""

# 示例 1: 基本对比（使用默认模型名）
echo "=========================================="
echo "示例 1: 基本对比（combined 模式）"
echo "=========================================="
python compare_spectrum_results.py \
    "$dir1" \
    "$dir2" \
    --output-dir spectrum_comparison_basic

echo ""
echo "✓ 结果保存在: spectrum_comparison_basic/"
echo ""

# 示例 2: 使用自定义模型名
echo "=========================================="
echo "示例 2: 自定义模型名"
echo "=========================================="
python compare_spectrum_results.py \
    "$dir1" \
    "$dir2" \
    --model-names "Model-A" "Model-B" \
    --output-dir spectrum_comparison_named

echo ""
echo "✓ 结果保存在: spectrum_comparison_named/"
echo ""

# 示例 3: 生成所有类型的图（如果需要详细对比）
read -p "是否生成所有类型的对比图？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "=========================================="
    echo "示例 3: 生成所有类型的图"
    echo "=========================================="
    python compare_spectrum_results.py \
        "$dir1" \
        "$dir2" \
        --model-names "Model-A" "Model-B" \
        --mode all \
        --output-dir spectrum_comparison_all

    echo ""
    echo "✓ 结果保存在: spectrum_comparison_all/"
    echo "  (包含 combined, 1d, 2d 所有类型的图)"
else
    echo "跳过示例 3"
fi

echo ""
echo "=========================================="
echo "✅ 对比完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  ls -lh spectrum_comparison_*/"
echo ""
echo "或打开图片查看器:"
echo "  eog spectrum_comparison_basic/*.png &"
