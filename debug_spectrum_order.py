#!/usr/bin/env python3
"""
诊断脚本：检查能量谱对比程序的数据加载顺序
"""

import sys
from pathlib import Path

import numpy as np


def check_order(result_dirs, model_names):
    """检查目录和模型名的对应关系"""

    print("=" * 60)
    print("数据加载顺序诊断")
    print("=" * 60)

    base_dir = Path("evaluation_results")

    print("\n输入信息:")
    print(f"  目录数量: {len(result_dirs)}")
    print(f"  模型名数量: {len(model_names)}")
    print()

    # 检查每个目录
    for i, (result_dir, model_name) in enumerate(zip(result_dirs, model_names, strict=False)):
        print(f"\n位置 {i}:")
        print(f"  目录名: {result_dir}")
        print(f"  模型名: {model_name}")
        print(f"  颜色索引: {i}")

        # 检查目录是否存在
        full_path = base_dir / result_dir
        if not full_path.exists():
            print("  ❌ 目录不存在!")
            continue

        # 尝试读取一个样本文件
        test_file = full_path / "spectrum_prediction_plane0_u_kx.npy"
        if test_file.exists():
            data = np.load(test_file)
            print(f"  ✓ 数据存在, 形状: {data.shape}")
            print(f"  样本值范围: [{data.min():.6e}, {data.max():.6e}]")
        else:
            print(f"  ⚠ 测试文件不存在: {test_file.name}")

    print("\n" + "=" * 60)
    print("顺序检查:")
    print("=" * 60)
    print("\n在图中:")
    print("  - 1D 图例从上到下的顺序 = Ground Truth, " + ", ".join(model_names))
    print("  - 2D 预测从左到右的顺序 = " + ", ".join(model_names))
    print("  - 2D 误差从左到右的顺序 = " + ", ".join(model_names))
    print()
    print("如果顺序不对，请调整命令行参数的顺序!")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法:")
        print("  python debug_spectrum_order.py dir1 dir2 dir3 --model-names Model1 Model2 Model3")
        sys.exit(1)

    # 解析参数
    args = sys.argv[1:]

    if "--model-names" in args:
        idx = args.index("--model-names")
        result_dirs = args[:idx]
        model_names = args[idx + 1 :]
    else:
        result_dirs = args
        model_names = [f"Model-{i + 1}" for i in range(len(result_dirs))]

    check_order(result_dirs, model_names)
