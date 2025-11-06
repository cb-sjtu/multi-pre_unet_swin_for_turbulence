#!/usr/bin/env python3
"""
多模型时间序列对比程序 - 1平面3通道版本

用于对比不同模型在相同采样点的时间序列预测结果（1-plane, 3-channel: u, v, w）。
支持 Teacher Forcing (TF) 和 Autoregressive (AR) 两种模式。

Usage:
    python compare_timeseries_results.py \
        logs/flow_fno_1plane/runs/2025-10-27_22-56-39-791052 \
        logs/flow_lstm_1plane/runs/2025-10-26_12-14-53-336652 \
        logs/flow_swin_1plane/runs/2025-09-22_11-09-35-088845 \
        --model-names "FNO" "LSTM" "Swin" \
        --output-dir timeseries_comparison_results
"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class TimeSeriesComparator:
    """时间序列对比器 - 用于对比多个模型的时间序列预测"""

    def __init__(self, run_dirs, model_names=None, output_dir="timeseries_comparison_results", max_timesteps=None):
        """
        初始化对比器

        Args:
            run_dirs: 训练run目录列表
            model_names: 模型名称列表（可选）
            output_dir: 输出目录
            max_timesteps: 最大显示的时间步数（None表示显示所有）
        """
        self.run_dirs = [Path(d) for d in run_dirs]
        self.max_timesteps = max_timesteps

        # 验证所有目录存在
        for run_dir in self.run_dirs:
            if not run_dir.exists():
                raise ValueError(f"Run目录不存在: {run_dir}")
            eval_dir = run_dir / "evaluation_results" / "time_series_data"
            if not eval_dir.exists():
                raise ValueError(f"时间序列数据目录不存在: {eval_dir}")

        # 设置模型名称
        if model_names is None:
            self.model_names = [d.name for d in self.run_dirs]
        else:
            if len(model_names) != len(run_dirs):
                raise ValueError(f"模型名称数量 ({len(model_names)}) 与目录数量 ({len(run_dirs)}) 不匹配")
            self.model_names = model_names

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 字段配置 - 1平面3通道：u, v, w (无pressure)
        self.fields = ["u", "v", "w"]
        # 1平面模型没有多个plane，只有单个y切片
        self.y_slice = 54  # 固定的y切片位置

        print("初始化时间序列对比器")
        print(f"  模型数量: {len(self.model_names)}")
        print(f"  模型名称: {', '.join(self.model_names)}")
        print(f"  输出目录: {self.output_dir}")

    def load_timeseries_data(self, run_dir, mode="tf"):
        """
        加载时间序列数据

        Args:
            run_dir: run目录
            mode: 'tf' (teacher forcing) 或 'ar' (autoregressive)

        Returns:
            pandas DataFrame
        """
        data_dir = run_dir / "evaluation_results" / "time_series_data"
        filename = f"time_series_test_{mode}.csv"
        filepath = data_dir / filename

        if not filepath.exists():
            print(f"  ⚠ 文件不存在: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"  ❌ 加载失败 {filename}: {e}")
            return None

    def parse_column_name(self, col_name):
        """
        解析列名，提取字段、点号、位置等信息（1平面格式）

        Example: 'u_pred_point0_y54_z40_x40' ->
            {'field': 'u', 'type': 'pred', 'point': 0, 'y': 54, 'z': 40, 'x': 40}
        """
        pattern = r"([uvw])_(pred|gt)_point(\d+)_y(\d+)_z(\d+)_x(\d+)"
        match = re.match(pattern, col_name)

        if not match:
            return None

        return {
            "field": match.group(1),
            "type": match.group(2),
            "point": int(match.group(3)),
            "y": int(match.group(4)),
            "z": int(match.group(5)),
            "x": int(match.group(6)),
        }

    def get_point_info_list(self, df):
        """
        从DataFrame中提取所有采样点信息（1平面格式）

        Returns:
            list of dict: 每个点的信息
        """
        point_info_list = []
        seen_points = set()

        for col in df.columns:
            if col == "timestep":
                continue

            info = self.parse_column_name(col)
            if info and info["type"] == "pred":  # 只处理预测列
                point_key = (info["point"], info["y"], info["z"], info["x"])
                if point_key not in seen_points:
                    point_info_list.append({"point": info["point"], "y": info["y"], "z": info["z"], "x": info["x"]})
                    seen_points.add(point_key)

        return sorted(point_info_list, key=lambda x: x["point"])

    def compare_point_timeseries(self, point_info, mode="tf"):
        """
        对比指定采样点的所有字段的时间序列（1平面格式）

        Args:
            point_info: 点信息字典
            mode: 'tf' 或 'ar'
        """
        point = point_info["point"]
        y = point_info["y"]
        z = point_info["z"]
        x = point_info["x"]

        mode_name = "Teacher Forcing" if mode == "tf" else "Autoregressive"
        print(f"\n对比点 {point} (y={y}, z={z}, x={x}) - {mode_name}")

        # 加载所有模型的数据
        dfs = []
        for run_dir in self.run_dirs:
            df = self.load_timeseries_data(run_dir, mode)
            dfs.append(df)

        # 检查是否有有效数据
        if all(df is None for df in dfs):
            print("  ⚠ 无有效数据，跳过")
            return

        # 创建子图：3个字段 (u, v, w) - 1行3列布局
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.flatten()

        # 为每个模型分配颜色，Swin使用红色
        colors = []
        default_colors = plt.cm.tab10(np.linspace(0, 1, len(self.model_names)))
        for i, model_name in enumerate(self.model_names):
            if model_name.lower() == "swin":
                colors.append("red")
            else:
                colors.append(default_colors[i])

        for field_idx, field in enumerate(self.fields):
            ax = axes[field_idx]

            # 构造列名 (1平面格式: field_pred/gt_point#_y#_z#_x#)
            gt_col = f"{field}_gt_point{point}_y{y}_z{z}_x{x}"

            # 首先绘制 Ground Truth（从第一个有效数据中读取）
            gt_data = None
            timesteps = None
            for df in dfs:
                if df is not None and gt_col in df.columns:
                    gt_data = df[gt_col].values
                    timesteps = df["timestep"].values
                    break

            if gt_data is not None:
                # 如果指定了最大时间步数，则截取数据
                if self.max_timesteps is not None and len(timesteps) > self.max_timesteps:
                    timesteps_plot = timesteps[: self.max_timesteps]
                    gt_data_plot = gt_data[: self.max_timesteps]
                else:
                    timesteps_plot = timesteps
                    gt_data_plot = gt_data

                ax.plot(timesteps_plot, gt_data_plot, "k-", linewidth=2, label="Ground Truth", alpha=0.8)

            # 绘制每个模型的预测
            for i, (df, model_name) in enumerate(zip(dfs, self.model_names, strict=False)):
                if df is None:
                    continue

                pred_col = f"{field}_pred_point{point}_y{y}_z{z}_x{x}"

                if pred_col in df.columns:
                    pred_data = df[pred_col].values
                    timesteps = df["timestep"].values

                    # 如果指定了最大时间步数，则截取数据
                    if self.max_timesteps is not None and len(timesteps) > self.max_timesteps:
                        timesteps_plot = timesteps[: self.max_timesteps]
                        pred_data_plot = pred_data[: self.max_timesteps]
                    else:
                        timesteps_plot = timesteps
                        pred_data_plot = pred_data

                    ax.plot(
                        timesteps_plot,
                        pred_data_plot,
                        "--",
                        linewidth=1.5,
                        color=colors[i],
                        label=model_name,
                        alpha=0.7,
                    )

            ax.set_xlabel("Timestep", fontsize=11)
            ax.set_ylabel(f"{field.upper()}", fontsize=11)
            ax.set_title(f"Field: {field.upper()}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3, linestyle=":")

        plt.suptitle(
            f"Time Series Comparison - Point{point} (y={y}, z={z}, x={x})\nMode: {mode_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # 保存图片 (1平面格式: point#_y#_z#_x#)
        output_file = self.output_dir / f"timeseries_point{point}_y{y}_z{z}_x{x}_{mode}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 保存: {output_file}")

    def run_all_comparisons(self, modes=None, point_indices=None):
        """
        运行所有对比

        Args:
            modes: 对比模式列表，可以是 ['tf'], ['ar'], 或 ['tf', 'ar']。默认为 ['tf', 'ar']
            point_indices: 要对比的点索引列表，None表示对比所有点
        """
        if modes is None:
            modes = ["tf", "ar"]
        print(f"\n{'=' * 60}")
        print("开始时间序列对比")
        print(f"{'=' * 60}")

        # 从第一个有效数据中获取所有点的信息
        point_info_list = None
        for run_dir in self.run_dirs:
            df = self.load_timeseries_data(run_dir, modes[0])
            if df is not None:
                point_info_list = self.get_point_info_list(df)
                break

        if point_info_list is None:
            print("❌ 无法获取采样点信息")
            return

        print(f"\n找到 {len(point_info_list)} 个采样点")

        # 如果指定了点索引，只处理这些点
        if point_indices is not None:
            point_info_list = [p for p in point_info_list if p["point"] in point_indices]
            print(f"仅对比指定的 {len(point_info_list)} 个点")

        total_plots = 0

        for mode in modes:
            mode_name = "Teacher Forcing" if mode == "tf" else "Autoregressive"
            print(f"\n{'=' * 60}")
            print(f"模式: {mode_name}")
            print(f"{'=' * 60}")

            for point_info in point_info_list:
                self.compare_point_timeseries(point_info, mode)
                total_plots += 1

        print(f"\n{'=' * 60}")
        print("✅ 对比完成！")
        print(f"{'=' * 60}")
        print(f"  生成图片数量: {total_plots}")
        print(f"  输出目录: {self.output_dir}")
        print("\n对比的配置:")
        print(f"  - 采样点: {len(point_info_list)} 个")
        print(f"  - 模式: {', '.join(modes)}")
        print(f"  - 模型: {len(self.model_names)} 个 ({', '.join(self.model_names)})")
        if self.max_timesteps is not None:
            print(f"  - 时间步: 前 {self.max_timesteps} 步")
        else:
            print("  - 时间步: 全部")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模型时间序列对比工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 对比三个1平面模型的时间序列（TF和AR模式）
  python compare_timeseries_results.py \\
      logs/flow_fno_1plane/runs/2025-10-27_22-56-39-791052 \\
      logs/flow_lstm_1plane/runs/2025-10-26_12-14-53-336652 \\
      logs/flow_swin_1plane/runs/2025-09-22_11-09-35-088845 \\
      --model-names "FNO" "LSTM" "Swin"

  # 只对比特定的点
  python compare_timeseries_results.py \\
      logs/flow_fno_1plane/runs/xxx \\
      logs/flow_lstm_1plane/runs/yyy \\
      --model-names "FNO" "LSTM" \\
      --points 0 4 8

  # 只对比 Teacher Forcing 模式
  python compare_timeseries_results.py \\
      run1 run2 run3 \\
      --model-names "M1" "M2" "M3" \\
      --modes tf

  # 只显示前50个时间步
  python compare_timeseries_results.py \\
      run1 run2 \\
      --model-names "M1" "M2" \\
      --max-timesteps 50
        """,
    )

    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="训练run目录路径列表",
    )

    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="模型名称列表（用于图例）。如果不指定，使用目录名",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="timeseries_comparison_results",
        help="输出目录（默认: timeseries_comparison_results）",
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["tf", "ar"],
        default=["tf", "ar"],
        help="对比模式: tf (Teacher Forcing), ar (Autoregressive)。默认: tf ar",
    )

    parser.add_argument(
        "--points",
        nargs="+",
        type=int,
        default=None,
        help="要对比的点索引列表。如果不指定，对比所有点",
    )

    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="最大显示的时间步数。如果不指定，显示所有时间步（例如：--max-timesteps 50 只显示前50步）",
    )

    args = parser.parse_args()

    # 验证参数
    if args.model_names is not None and len(args.model_names) != len(args.run_dirs):
        parser.error(f"模型名称数量 ({len(args.model_names)}) 必须与目录数量 ({len(args.run_dirs)}) 匹配")

    # 创建对比器
    try:
        comparator = TimeSeriesComparator(
            run_dirs=args.run_dirs,
            model_names=args.model_names,
            output_dir=args.output_dir,
            max_timesteps=args.max_timesteps,
        )

        # 运行对比
        comparator.run_all_comparisons(modes=args.modes, point_indices=args.points)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
