#!/usr/bin/env python3
"""
多模型能量谱对比程序

用于对比不同模型在 evaluation_results 文件夹下的能量谱结果。
可以同时对比多个模型的预测结果与真值（Ground Truth）。

Usage:
    python compare_spectrum_results.py \
        evaluation_results_2025-10-27_22-56-39-791052 \
        evaluation_results_2025-10-26_12-14-53-336652 \
        --model-names "LSTM" "Swin" \
        --output-dir comparison_results
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


class SpectrumComparator:
    """能量谱对比器 - 用于对比多个模型的能量谱结果"""

    def __init__(self, result_dirs, model_names=None, output_dir="comparison_results"):
        """
        初始化对比器

        Args:
            result_dirs: 评估结果目录列表
            model_names: 模型名称列表（可选，默认使用目录名）
            output_dir: 输出目录
        """
        self.result_dirs = [Path(d) for d in result_dirs]
        self.base_dir = Path("evaluation_results")

        # 验证所有目录存在
        for result_dir in self.result_dirs:
            full_path = self.base_dir / result_dir
            if not full_path.exists():
                raise ValueError(f"结果目录不存在: {full_path}")

        # 设置模型名称
        if model_names is None:
            self.model_names = [d.name.replace("evaluation_results_", "") for d in self.result_dirs]
        else:
            if len(model_names) != len(result_dirs):
                raise ValueError(f"模型名称数量 ({len(model_names)}) 与目录数量 ({len(result_dirs)}) 不匹配")
            self.model_names = model_names

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 3-plane 配置
        self.planes = [0, 1, 2]
        self.fields = ["u", "v", "w", "p"]
        self.spectrum_types = ["kx", "kz", "2d"]  # 1D kx, 1D kz, 2D spectrum

        print("初始化能量谱对比器")
        print(f"  模型数量: {len(self.model_names)}")
        print(f"  模型名称: {', '.join(self.model_names)}")
        print(f"  输出目录: {self.output_dir}")

    def load_spectrum_data(self, result_dir, plane, field, spectrum_type, data_type="prediction"):
        """
        加载能量谱数据

        Args:
            result_dir: 结果目录
            plane: 平面索引 (0, 1, 2)
            field: 字段名 ('u', 'v', 'w', 'p')
            spectrum_type: 谱类型 ('kx', 'kz', '2d')
            data_type: 数据类型 ('prediction' or 'ground_truth')

        Returns:
            numpy array 或 None（如果文件不存在）
        """
        full_path = self.base_dir / result_dir
        filename = f"spectrum_{data_type}_plane{plane}_{field}_{spectrum_type}.npy"
        filepath = full_path / filename

        if not filepath.exists():
            print(f"  ⚠ 文件不存在: {filename}")
            return None

        try:
            data = np.load(filepath)
            return data
        except Exception as e:
            print(f"  ❌ 加载失败 {filename}: {e}")
            return None

    def compare_1d_spectrum(self, plane, field, spectrum_type="kx"):
        """
        对比 1D 能量谱（kx 或 kz）

        Args:
            plane: 平面索引
            field: 字段名
            spectrum_type: 'kx' 或 'kz'
        """
        print(f"\n对比 1D 谱: Plane{plane} - {field} - {spectrum_type}")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 加载并绘制 Ground Truth（只需要从第一个结果目录加载）
        gt_data = self.load_spectrum_data(self.result_dirs[0], plane, field, spectrum_type, "ground_truth")

        if gt_data is not None:
            k = np.arange(len(gt_data))
            ax.loglog(k, gt_data, "k-", linewidth=2, label="Ground Truth", alpha=0.8)

        # 加载并绘制每个模型的预测结果
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.model_names)))

        for i, (result_dir, model_name) in enumerate(zip(self.result_dirs, self.model_names, strict=False)):
            pred_data = self.load_spectrum_data(result_dir, plane, field, spectrum_type, "prediction")

            if pred_data is not None:
                k = np.arange(len(pred_data))
                ax.loglog(k, pred_data, "--", linewidth=1.5, color=colors[i], label=model_name, alpha=0.7)

        # 添加参考线 (k^-5/3 for Kolmogorov spectrum)
        if gt_data is not None:
            k_ref = np.logspace(0, np.log10(len(gt_data) // 2), 50)
            # 调整参考线使其与数据对齐
            ref_scale = gt_data[10] / (k_ref[10] ** (-5 / 3))
            ax.loglog(k_ref, ref_scale * k_ref ** (-5 / 3), "r:", linewidth=1, label=r"$k^{-5/3}$", alpha=0.5)

        ax.set_xlabel(f"Wavenumber {spectrum_type}", fontsize=12)
        ax.set_ylabel("Energy", fontsize=12)
        ax.set_title(f"Energy Spectrum Comparison - Plane{plane} ({field}) - {spectrum_type.upper()}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which="both", linestyle=":")

        # 保存图片
        output_file = self.output_dir / f"comparison_1d_plane{plane}_{field}_{spectrum_type}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 保存: {output_file}")

    def compare_2d_spectrum(self, plane, field):
        """
        对比 2D 能量谱

        Args:
            plane: 平面索引
            field: 字段名
        """
        print(f"\n对比 2D 谱: Plane{plane} - {field}")

        # 加载数据
        gt_data = self.load_spectrum_data(self.result_dirs[0], plane, field, "2d", "ground_truth")

        pred_data_list = []
        for result_dir in self.result_dirs:
            pred_data = self.load_spectrum_data(result_dir, plane, field, "2d", "prediction")
            pred_data_list.append(pred_data)

        # 检查数据有效性
        valid_data = [d for d in [gt_data] + pred_data_list if d is not None]
        if not valid_data:
            print("  ⚠ 无有效数据，跳过")
            return

        # 创建子图：GT + 每个模型 + 差异图
        n_models = len(self.model_names)
        n_cols = min(4, 2 + n_models)  # GT, models, 最多4列
        n_rows = int(np.ceil((1 + n_models) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes_flat = axes.flatten()

        # 确定统一的颜色范围
        vmin = min(d.min() for d in valid_data if d is not None)
        vmax = max(d.max() for d in valid_data if d is not None)

        plot_idx = 0

        # 绘制 Ground Truth
        if gt_data is not None:
            im = axes_flat[plot_idx].imshow(
                np.log10(gt_data + 1e-20),
                cmap="viridis",
                aspect="auto",
                vmin=np.log10(vmin + 1e-20),
                vmax=np.log10(vmax + 1e-20),
            )
            axes_flat[plot_idx].set_title("Ground Truth", fontsize=11)
            axes_flat[plot_idx].set_xlabel("kx")
            axes_flat[plot_idx].set_ylabel("kz")
            plt.colorbar(im, ax=axes_flat[plot_idx], label="log10(Energy)")
            plot_idx += 1

        # 绘制每个模型的预测
        for model_name, pred_data in zip(self.model_names, pred_data_list, strict=False):
            if pred_data is not None:
                im = axes_flat[plot_idx].imshow(
                    np.log10(pred_data + 1e-20),
                    cmap="viridis",
                    aspect="auto",
                    vmin=np.log10(vmin + 1e-20),
                    vmax=np.log10(vmax + 1e-20),
                )
                axes_flat[plot_idx].set_title(model_name, fontsize=11)
                axes_flat[plot_idx].set_xlabel("kx")
                axes_flat[plot_idx].set_ylabel("kz")
                plt.colorbar(im, ax=axes_flat[plot_idx], label="log10(Energy)")
                plot_idx += 1

        # 隐藏多余的子图
        for idx in range(plot_idx, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.suptitle(f"2D Energy Spectrum Comparison - Plane{plane} ({field})", fontsize=14, y=1.02)
        plt.tight_layout()

        # 保存图片
        output_file = self.output_dir / f"comparison_2d_plane{plane}_{field}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 保存: {output_file}")

    def compare_combined_spectrum(self, plane, field):
        """
        对比组合能量谱（1D kx + kz + 2D，包含真值、所有模型预测和相对误差）

        Args:
            plane: 平面索引
            field: 字段名
        """
        print(f"\n对比组合谱: Plane{plane} - {field}")

        # 加载所有数据
        gt_kx = self.load_spectrum_data(self.result_dirs[0], plane, field, "kx", "ground_truth")
        gt_kz = self.load_spectrum_data(self.result_dirs[0], plane, field, "kz", "ground_truth")
        gt_2d = self.load_spectrum_data(self.result_dirs[0], plane, field, "2d", "ground_truth")

        pred_kx_list = []
        pred_kz_list = []
        pred_2d_list = []

        for result_dir in self.result_dirs:
            pred_kx_list.append(self.load_spectrum_data(result_dir, plane, field, "kx", "prediction"))
            pred_kz_list.append(self.load_spectrum_data(result_dir, plane, field, "kz", "prediction"))
            pred_2d_list.append(self.load_spectrum_data(result_dir, plane, field, "2d", "prediction"))

        # 计算需要的子图数量：1D kx + 1D kz + GT 2D + N个模型的2D预测 + N个相对误差
        # 布局：第一行 kx, kz, GT；第二行 模型预测；第三行 相对误差
        n_models = len(self.model_names)
        n_cols = max(3, n_models + 1)  # 至少3列，或者 GT + 所有模型

        # 创建图形：3行布局
        # 第1行：1D kx, 1D kz, 2D GT
        # 第2行：各模型的 2D 预测
        # 第3行：各模型的相对误差
        fig = plt.figure(figsize=(4.5 * n_cols, 12))
        gs = fig.add_gridspec(3, n_cols, hspace=0.35, wspace=0.35)

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.model_names)))

        # ========== 第1行：1D 谱 + GT 2D ==========
        # --- 子图 (0,0): kx 谱对比 ---
        ax_kx = fig.add_subplot(gs[0, 0])
        if gt_kx is not None:
            k = np.arange(len(gt_kx))
            ax_kx.loglog(k, gt_kx, "k-", linewidth=2, label="Ground Truth", alpha=0.8)

        for i, (pred_kx, model_name) in enumerate(zip(pred_kx_list, self.model_names, strict=False)):
            if pred_kx is not None:
                k = np.arange(len(pred_kx))
                ax_kx.loglog(k, pred_kx, "--", linewidth=1.5, color=colors[i], label=model_name, alpha=0.7)

        ax_kx.set_xlabel("Wavenumber kx", fontsize=10)
        ax_kx.set_ylabel("Energy", fontsize=10)
        ax_kx.set_title("1D Spectrum (kx)", fontsize=11, fontweight="bold")
        ax_kx.legend(fontsize=8, loc="best")
        ax_kx.grid(True, alpha=0.3, which="both", linestyle=":")

        # --- 子图 (0,1): kz 谱对比 ---
        ax_kz = fig.add_subplot(gs[0, 1])
        if gt_kz is not None:
            k = np.arange(len(gt_kz))
            ax_kz.loglog(k, gt_kz, "k-", linewidth=2, label="Ground Truth", alpha=0.8)

        for i, (pred_kz, model_name) in enumerate(zip(pred_kz_list, self.model_names, strict=False)):
            if pred_kz is not None:
                k = np.arange(len(pred_kz))
                ax_kz.loglog(k, pred_kz, "--", linewidth=1.5, color=colors[i], label=model_name, alpha=0.7)

        ax_kz.set_xlabel("Wavenumber kz", fontsize=10)
        ax_kz.set_ylabel("Energy", fontsize=10)
        ax_kz.set_title("1D Spectrum (kz)", fontsize=11, fontweight="bold")
        ax_kz.legend(fontsize=8, loc="best")
        ax_kz.grid(True, alpha=0.3, which="both", linestyle=":")

        # --- 子图 (0,2): Ground Truth 2D 谱 ---
        ax_gt = fig.add_subplot(gs[0, 2])
        if gt_2d is not None:
            im_gt = ax_gt.imshow(np.log10(gt_2d + 1e-20), cmap="viridis", aspect="auto", origin="lower")
            ax_gt.set_title("Ground Truth\n2D Spectrum", fontsize=11, fontweight="bold")
            ax_gt.set_xlabel("kx", fontsize=10)
            ax_gt.set_ylabel("kz", fontsize=10)
            cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            cbar_gt.set_label("log10(Energy)", fontsize=9)

        # 隐藏第1行多余的子图
        for col_idx in range(3, n_cols):
            ax_empty = fig.add_subplot(gs[0, col_idx])
            ax_empty.axis("off")

        # ========== 第2行：各模型的 2D 预测 ==========
        # 计算统一的颜色范围（仅使用 Ground Truth 的范围）
        if gt_2d is not None:
            vmin_2d = np.log10(gt_2d.min() + 1e-20)
            vmax_2d = np.log10(gt_2d.max() + 1e-20)
        else:
            vmin_2d, vmax_2d = None, None

        for i, (pred_2d, model_name) in enumerate(zip(pred_2d_list, self.model_names, strict=False)):
            ax_pred = fig.add_subplot(gs[1, i])
            if pred_2d is not None:
                im_pred = ax_pred.imshow(
                    np.log10(pred_2d + 1e-20), cmap="viridis", aspect="auto", origin="lower", vmin=vmin_2d, vmax=vmax_2d
                )
                ax_pred.set_title(f"{model_name}\nPrediction", fontsize=11, fontweight="bold")
                ax_pred.set_xlabel("kx", fontsize=10)
                ax_pred.set_ylabel("kz", fontsize=10)
                cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
                cbar_pred.set_label("log10(Energy)", fontsize=9)
            else:
                ax_pred.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
                ax_pred.set_title(f"{model_name}\nPrediction", fontsize=11)
                ax_pred.axis("off")

        # 隐藏第2行多余的子图
        for col_idx in range(n_models, n_cols):
            ax_empty = fig.add_subplot(gs[1, col_idx])
            ax_empty.axis("off")

        # ========== 第3行：各模型的相对误差（统一色标）==========
        # 计算所有模型的相对误差，找到统一的色标范围
        rel_errors = []
        if gt_2d is not None:
            for pred_2d in pred_2d_list:
                if pred_2d is not None:
                    rel_error = np.abs(pred_2d - gt_2d) / (gt_2d + 1e-20)
                    rel_errors.append(rel_error)

        # 确定统一的相对误差色标范围
        if rel_errors:
            # 使用 log10 刻度
            all_errors = np.concatenate([e.flatten() for e in rel_errors])
            vmin_err = np.log10(np.percentile(all_errors, 1) + 1e-10)  # 使用1%分位数避免极值
            vmax_err = np.log10(np.percentile(all_errors, 99) + 1e-10)  # 使用99%分位数

            for i, (pred_2d, model_name, rel_error) in enumerate(
                zip(pred_2d_list, self.model_names, rel_errors, strict=False)
            ):
                ax_err = fig.add_subplot(gs[2, i])
                if pred_2d is not None:
                    im_err = ax_err.imshow(
                        np.log10(rel_error + 1e-10),
                        cmap="hot_r",  # 反转 hot colormap: 浅色=小误差, 深色=大误差
                        aspect="auto",
                        origin="lower",
                        vmin=vmin_err,
                        vmax=vmax_err,
                    )
                    ax_err.set_title(f"{model_name}\nRelative Error", fontsize=11, fontweight="bold")
                    ax_err.set_xlabel("kx", fontsize=10)
                    ax_err.set_ylabel("kz", fontsize=10)
                    cbar_err = plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
                    cbar_err.set_label("log10(Rel. Err)", fontsize=9)
                else:
                    ax_err.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
                    ax_err.set_title(f"{model_name}\nRelative Error", fontsize=11)
                    ax_err.axis("off")

        # 隐藏第3行多余的子图
        for col_idx in range(n_models, n_cols):
            ax_empty = fig.add_subplot(gs[2, col_idx])
            ax_empty.axis("off")

        plt.suptitle(
            f"Energy Spectrum Comparison - Plane{plane} ({field.upper()})\n"
            f"Row 1: 1D Spectra + GT 2D | Row 2: Model Predictions (GT colorbar range) |\
                  Row 3: Relative Errors (unified range)",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )

        # 保存图片
        output_file = self.output_dir / f"comparison_combined_plane{plane}_{field}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 保存: {output_file}")

    def run_all_comparisons(self, spectrum_mode="combined"):
        """
        运行所有对比

        Args:
            spectrum_mode: 对比模式
                - 'combined': 生成组合图（推荐）
                - '1d': 只生成 1D 谱对比
                - '2d': 只生成 2D 谱对比
                - 'all': 生成所有类型的对比图
        """
        print(f"\n{'=' * 60}")
        print(f"开始能量谱对比 (模式: {spectrum_mode})")
        print(f"{'=' * 60}")

        total_plots = 0

        for plane in self.planes:
            for field in self.fields:
                if spectrum_mode in ["combined", "all"]:
                    self.compare_combined_spectrum(plane, field)
                    total_plots += 1

                if spectrum_mode in ["1d", "all"]:
                    self.compare_1d_spectrum(plane, field, "kx")
                    self.compare_1d_spectrum(plane, field, "kz")
                    total_plots += 2

                if spectrum_mode in ["2d", "all"]:
                    self.compare_2d_spectrum(plane, field)
                    total_plots += 1

        print(f"\n{'=' * 60}")
        print("✅ 对比完成！")
        print(f"{'=' * 60}")
        print(f"  生成图片数量: {total_plots}")
        print(f"  输出目录: {self.output_dir}")
        print("\n对比的配置:")
        print(f"  - 平面: {len(self.planes)} 个 (plane 0, 1, 2)")
        print(f"  - 字段: {len(self.fields)} 个 ({', '.join(self.fields)})")
        print(f"  - 模型: {len(self.model_names)} 个 ({', '.join(self.model_names)})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模型能量谱对比工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 对比两个模型（使用默认模型名）
  python compare_spectrum_results.py \\
      evaluation_results_2025-10-27_22-56-39-791052 \\
      evaluation_results_2025-10-26_12-14-53-336652

  # 对比两个模型（指定模型名）
  python compare_spectrum_results.py \\
      evaluation_results_2025-10-27_22-56-39-791052 \\
      evaluation_results_2025-10-26_12-14-53-336652 \\
      --model-names "LSTM" "Swin Transformer"

  # 对比三个模型
  python compare_spectrum_results.py \\
      dir1 dir2 dir3 \\
      --model-names "Model-A" "Model-B" "Model-C" \\
      --output-dir my_comparison

  # 生成所有类型的对比图
  python compare_spectrum_results.py dir1 dir2 --mode all
        """,
    )

    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="评估结果目录名称（在 evaluation_results/ 下）",
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
        default="spectrum_comparison_results",
        help="输出目录（默认: spectrum_comparison_results）",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["combined", "1d", "2d", "all"],
        default="combined",
        help="对比模式: combined(组合图), 1d(仅1D谱), 2d(仅2D谱), all(所有图)",
    )

    args = parser.parse_args()

    # 验证参数
    if args.model_names is not None and len(args.model_names) != len(args.result_dirs):
        parser.error(f"模型名称数量 ({len(args.model_names)}) 必须与目录数量 ({len(args.result_dirs)}) 匹配")

    # 创建对比器
    try:
        comparator = SpectrumComparator(
            result_dirs=args.result_dirs, model_names=args.model_names, output_dir=args.output_dir
        )

        # 运行对比
        comparator.run_all_comparisons(spectrum_mode=args.mode)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
