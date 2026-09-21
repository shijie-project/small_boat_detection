from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


bins = [0, 8**2, 16**2, 32**2, 96**2, 1e9]
labels = ["VeryTiny\n(<8²)", "Tiny\n(<16²)", "Small\n(<32²)", "Medium\n(<96²)", "Large\n(>96²)"]
colors = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]


def _get_bins(values):
    return [values[(areas >= bins[i]) & (areas < bins[i + 1])] for i in range(len(labels))]


def _violin_with_box(ax, data_bins, ylabel, title, color_list, show_zero_line=False):
    valid_data = [(i, d, l, c) for i, (d, l, c) in enumerate(zip(data_bins, labels, color_list)) if len(d) > 30]

    positions = [v[0] for v in valid_data]
    plot_data = [v[1] for v in valid_data]
    plot_labels = [v[2] for v in valid_data]
    plot_colors = [v[3] for v in valid_data]

    # Violin
    parts = ax.violinplot(plot_data, positions=positions, showmedians=False, showextrema=False, widths=0.7)
    for pc, color in zip(parts["bodies"], plot_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
        pc.set_edgecolor("gray")
        pc.set_linewidth(0.5)

    # 内嵌箱线图（不显示离群点）
    bp = ax.boxplot(
        plot_data,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,  # ← 不显示离群点
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.9)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    if show_zero_line:
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="unbiased (0)")
        ax.legend(fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(plot_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_vfl_status():
    # ----------------------------------------------------------------
    # Figure 1: IoU / Score / Residual
    # ----------------------------------------------------------------
    def _plot_1():
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        _violin_with_box(
            axes[0],
            _get_bins(ious),
            ylabel="IoU",
            title="IoU Distribution vs Object Size",
            color_list=colors,
        )
        axes[0].set_ylim(0, 1)

        _violin_with_box(
            axes[1],
            _get_bins(scores),
            ylabel="Predicted Score",
            title="Predicted Score Distribution vs Object Size",
            color_list=colors,
        )
        axes[1].set_ylim(0, 1)

        _violin_with_box(
            axes[2],
            _get_bins(residuals),
            ylabel="score − IoU",
            title="Score Residual vs Object Size",
            color_list=colors,
            show_zero_line=True,
        )

        plt.tight_layout()
        plt.savefig(output_dir / "vfl_diagnosis_violin.png", dpi=200, bbox_inches="tight")
        plt.show()

    _plot_1()

    # ----------------------------------------------------------------
    # Figure 2: Score vs IoU hexbin
    # ----------------------------------------------------------------
    n_size_bins = len(labels)
    n_cols = 2
    n_hex_rows = n_size_bins // n_cols + 1
    n_rows = n_hex_rows + 1  # summary errorbar

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes_flat = axes.flatten()

    valid_bins = []

    for ax_idx, (lo, hi, label, color) in enumerate(zip(bins[:-1], bins[1:], labels, colors)):
        ax = axes_flat[ax_idx]
        mask = (areas >= lo) & (areas < hi)

        if mask.sum() < 30:
            ax.set_visible(False)
            continue

        hb = ax.hexbin(
            ious[mask],
            scores[mask],
            gridsize=50,
            cmap="YlOrRd",
            mincnt=1,
            extent=[0, 1, 0, 1],
        )
        ax.plot([0, 1], [0, 1], "b--", linewidth=1.5, label="Ideal")
        plt.colorbar(hb, ax=ax, label="Count")
        ax.set_xlabel("IoU (soft label)", fontsize=11)
        ax.set_ylabel("Predicted Score", fontsize=11)
        ax.set_title(
            f"Score vs IoU — {label.replace(chr(10), ' ')}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        valid_bins.append((label, color, mask))

    for j in range(n_size_bins, n_hex_rows * n_cols):
        axes_flat[j].set_visible(False)

    ax_summary = axes[-1, 0]

    for label, color, mask in valid_bins:
        mean_iou = ious[mask].mean()
        mean_score = scores[mask].mean()
        std_score = scores[mask].std()
        ax_summary.errorbar(
            mean_iou,
            mean_score,
            yerr=std_score,
            fmt="o",
            color=color,
            markersize=10,
            capsize=4,
            label=label.replace("\n", " "),
            linewidth=1.5,
        )

    ax_summary.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal (score = IoU)")
    ax_summary.set_xlabel("Mean IoU per Size Bin", fontsize=11)
    ax_summary.set_ylabel("Mean Predicted Score ± std", fontsize=11)
    ax_summary.set_title("Mean Score vs Mean IoU by Size", fontsize=12, fontweight="bold")
    ax_summary.legend(fontsize=9)
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.spines["top"].set_visible(False)
    ax_summary.spines["right"].set_visible(False)

    # 隐藏最后一行其余空格
    for j in range(1, n_cols):
        axes[-1, j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "vfl_diagnosis_scatter.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------
    # Figure 3: Gradient 分析（保留结构，改善视觉）
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _violin_with_box(
        axes[0],
        _get_bins(gradient_weights),
        ylabel="Gradient Weight q",
        title="Effective Gradient Weight (q) vs Object Size\n(lower = weaker training signal)",
        color_list=colors,
    )
    axes[0].set_ylim(0, 1)

    # 样本数 vs 归一化梯度贡献（per-sample，更公平的比较）
    gw_bins = _get_bins(gradient_weights)
    counts = [len(b) for b in gw_bins]
    mean_gw = [b.sum() if len(b) > 0 else 0 for b in gw_bins]

    # per-sample 归一化梯度（消除样本数差异的影响）
    max_gw = max(mean_gw) if max(mean_gw) > 0 else 1
    norm_gw = [g / max_gw for g in mean_gw]

    x = np.arange(len(labels))
    valid_mask = [c > 0 for c in counts]

    ax_left = axes[1]
    ax_right = axes[1].twinx()

    ax_left.bar(
        [i for i, v in enumerate(valid_mask) if v],
        [c for c, v in zip(counts, valid_mask) if v],
        width=0.4,
        color="steelblue",
        alpha=0.7,
        label="Sample Count",
    )
    ax_right.bar(
        [i + 0.4 for i, v in enumerate(valid_mask) if v],
        [g for g, v in zip(norm_gw, valid_mask) if v],
        width=0.4,
        color="tomato",
        alpha=0.7,
        label="Normalized Gradient Weight",
    )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=10)
    ax_left.set_ylabel("Sample Count", color="steelblue", fontsize=11)
    ax_right.set_ylabel("Norm. Mean Gradient Weight", color="tomato", fontsize=11)
    ax_left.set_title("Sample Count vs Per-sample Gradient Weight", fontsize=12, fontweight="bold")
    ax_left.set_ylim(bottom=0)
    ax_right.set_ylim(0, 1.2)

    lines1, lbls1 = ax_left.get_legend_handles_labels()
    lines2, lbls2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, lbls1 + lbls2, fontsize=9)
    ax_left.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "vfl_diagnosis_gradient.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------
    # Figure 4: Calibration curve（不变，已经够好）
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_bins_cal = 10
    bin_edges = np.linspace(0, 1, n_bins_cal + 1)

    mean_score_per_bin, mean_iou_per_bin = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() > 10:
            mean_score_per_bin.append(scores[mask].mean())
            mean_iou_per_bin.append(ious[mask].mean())

    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[0].plot(mean_score_per_bin, mean_iou_per_bin, "o-", color="steelblue", linewidth=2, label="Model")
    axes[0].fill_between(
        mean_score_per_bin,
        np.array(mean_iou_per_bin) - 0.05,
        np.array(mean_iou_per_bin) + 0.05,
        alpha=0.15,
        color="steelblue",
    )
    axes[0].set_xlabel("Mean Predicted Score", fontsize=11)
    axes[0].set_ylabel("Mean IoU", fontsize=11)
    axes[0].set_title("Reliability Diagram (Overall)", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    for lo_s, hi_s, label, color in zip(bins[:-1], bins[1:], labels, colors):
        size_mask = (areas >= lo_s) & (areas < hi_s)
        if size_mask.sum() < 50:
            continue
        s_scores = scores[size_mask]
        s_ious = ious[size_mask]
        bin_c, bin_iou = [], []
        for lo_b, hi_b in zip(bin_edges[:-1], bin_edges[1:]):
            m = (s_scores >= lo_b) & (s_scores < hi_b)
            if m.sum() > 5:
                bin_c.append((lo_b + hi_b) / 2)
                bin_iou.append(s_ious[m].mean())
        if len(bin_c) > 1:
            axes[1].plot(bin_c, bin_iou, "o-", color=color, label=label.replace("\n", " "), linewidth=2)

    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect")
    axes[1].set_xlabel("Mean Predicted Score", fontsize=11)
    axes[1].set_ylabel("Mean IoU", fontsize=11)
    axes[1].set_title("Reliability Diagram by Object Size", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "vfl_diagnosis_calibration.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------
    # Figure 5（新增）: Classification Accuracy vs Object Size
    # ----------------------------------------------------------------
    if "is_correct_class" in data:
        is_correct = data["is_correct_class"].astype(bool)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：分尺度的分类准确率
        acc_means = []
        acc_stds = []
        acc_labels = []
        acc_colors = []
        acc_positions = []

        for i, (lo, hi, label, color) in enumerate(zip(bins[:-1], bins[1:], labels, colors)):
            mask = (areas >= lo) & (areas < hi)
            if mask.sum() < 30:
                continue
            correct_in_bin = is_correct[mask]
            acc_means.append(correct_in_bin.mean())
            # Wilson interval for binomial proportion
            n = mask.sum()
            p = correct_in_bin.mean()
            ci = 1.96 * np.sqrt(p * (1 - p) / n)
            acc_stds.append(ci)
            acc_labels.append(label.replace("\n", " "))
            acc_colors.append(color)
            acc_positions.append(i)

        axes[0].bar(acc_positions, acc_means, color=acc_colors, alpha=0.75, width=0.6, edgecolor="gray")
        axes[0].errorbar(acc_positions, acc_means, yerr=acc_stds, fmt="none", color="black", capsize=4, linewidth=1.5)
        axes[0].set_xticks(acc_positions)
        axes[0].set_xticklabels(acc_labels, fontsize=10)
        axes[0].set_ylabel("Classification Accuracy", fontsize=11)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_title("Classification Accuracy vs Object Size", fontsize=12, fontweight="bold")
        axes[0].axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Perfect (1.0)")
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].legend(fontsize=9)

        # 右图：Score 对比（gt_class_scores vs max_scores）
        # 核心问题：GT class 的 score 是否低于最大 score？
        # 如果两者接近，说明模型知道正确类别但分数低
        if "max_scores" in data:
            max_scores = data["max_scores"]

            gt_score_by_bin = _get_bins(scores)
            max_score_by_bin = _get_bins(max_scores)

            x = np.arange(len(labels))
            width = 0.35
            valid = [
                (i, lo, hi)
                for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]))
                if (areas >= lo).sum() > 30 and (areas < hi).sum() > 0 and len(gt_score_by_bin[i]) > 30
            ]

            for i, lo, hi in valid:
                gt_mean = gt_score_by_bin[i].mean()
                max_mean = max_score_by_bin[i].mean()
                axes[1].bar(
                    i - width / 2,
                    gt_mean,
                    width,
                    color=colors[i],
                    alpha=0.8,
                    label="GT class score" if i == valid[0][0] else "",
                )
                axes[1].bar(
                    i + width / 2,
                    max_mean,
                    width,
                    color=colors[i],
                    alpha=0.4,
                    hatch="//",
                    label="Max class score" if i == valid[0][0] else "",
                )

            axes[1].set_xticks([v[0] for v in valid])
            axes[1].set_xticklabels([labels[v[0]].replace("\n", " ") for v in valid], fontsize=10)
            axes[1].set_ylabel("Mean Score", fontsize=11)
            axes[1].set_ylim(0, 1)
            axes[1].set_title(
                "GT Class Score vs Max Class Score\n(gap = model predicts wrong class)", fontsize=12, fontweight="bold"
            )
            axes[1].legend(fontsize=9)
            axes[1].grid(axis="y", alpha=0.3, linestyle="--")
            axes[1].spines["top"].set_visible(False)
            axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / "vfl_diagnosis_accuracy.png", dpi=200, bbox_inches="tight")
        plt.show()

    # ----------------------------------------------------------------
    # 数值摘要
    # ----------------------------------------------------------------
    print("\n=== Summary Table ===")
    print(f"{'Size':<20} {'N':>8} {'Mean IoU':>12} {'Mean Score':>12} {'Mean Residual':>16} {'Mean Grad W':>14}")
    print("-" * 85)
    for i, label in enumerate(labels):
        mask = (areas >= bins[i]) & (areas < bins[i + 1])
        if mask.sum() == 0:
            continue
        print(
            f"{label.replace(chr(10), ' '):<20} "
            f"{mask.sum():>8} "
            f"{ious[mask].mean():>12.4f} "
            f"{scores[mask].mean():>12.4f} "
            f"{residuals[mask].mean():>16.4f} "
            f"{gradient_weights[mask].mean():>14.4f}"
        )


def analyze_counterfactual_calibration():
    # ── Step 1：统计每个 bin 的 IoU 上界（95th percentile）──────────
    q_max = np.ones(len(bins) - 1)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (areas >= lo) & (areas < hi)
        if mask.sum() > 30:
            q_max[i] = np.percentile(ious[mask], 95)

    print("=== IoU Upper Bound (95th percentile) per Size Bin ===")
    for i, label in enumerate(labels):
        print(f"  {label.replace(chr(10), ' '):<20}: q_max = {q_max[i]:.4f}")

    # ── Step 2：计算校正后的 score ──────────────────────────────────
    scores_calibrated = scores.copy()
    bin_indices = np.zeros(len(areas), dtype=int)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (areas >= lo) & (areas < hi)
        bin_indices[mask] = i
        if q_max[i] > 0:
            scores_calibrated[mask] = np.clip(scores[mask] / q_max[i], 0, 1)

    # ── Step 3：比较校正前后的 score 分布 ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 左图：校正前后的 score 均值对比
    mean_before, mean_after, mean_iou_list = [], [], []
    valid_positions, valid_labels, valid_colors = [], [], []

    for i, (lo, hi, label, color) in enumerate(zip(bins[:-1], bins[1:], labels, colors)):
        mask = (areas >= lo) & (areas < hi)
        if mask.sum() < 30:
            continue
        mean_before.append(scores[mask].mean())
        mean_after.append(scores_calibrated[mask].mean())
        mean_iou_list.append(ious[mask].mean())
        valid_positions.append(i)
        valid_labels.append(label.replace("\n", " "))
        valid_colors.append(color)

    x = np.arange(len(valid_positions))
    width = 0.25

    axes[0].bar(x - width, mean_iou_list, width, label="Mean IoU (target)", color="gray", alpha=0.6)
    axes[0].bar(x, mean_before, width, label="Score (original)", color="steelblue", alpha=0.8)
    axes[0].bar(x + width, mean_after, width, label="Score (calibrated)", color="tomato", alpha=0.8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(valid_labels, fontsize=10)
    axes[0].set_ylabel("Score / IoU", fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Score Before/After Calibration vs IoU", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # 중간图：校正前后 residual（score - IoU）
    residual_before = scores - ious
    residual_after = scores_calibrated - ious

    residual_before_bins = [residual_before[(areas >= bins[i]) & (areas < bins[i + 1])] for i in range(len(labels))]
    residual_after_bins = [residual_after[(areas >= bins[i]) & (areas < bins[i + 1])] for i in range(len(labels))]

    # violin plot 对比
    valid = [(i, lo, hi) for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])) if len(residual_before_bins[i]) > 30]

    pos_before = [v[0] * 2 - 0.3 for v in valid]
    pos_after = [v[0] * 2 + 0.3 for v in valid]

    parts_b = axes[1].violinplot(
        [residual_before_bins[v[0]] for v in valid],
        positions=pos_before,
        widths=0.5,
        showmedians=True,
        showextrema=False,
    )
    parts_a = axes[1].violinplot(
        [residual_after_bins[v[0]] for v in valid],
        positions=pos_after,
        widths=0.5,
        showmedians=True,
        showextrema=False,
    )

    for pc in parts_b["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.5)
    for pc in parts_a["bodies"]:
        pc.set_facecolor("tomato")
        pc.set_alpha(0.5)

    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1.5, label="Unbiased (0)")
    axes[1].set_xticks([v[0] * 2 for v in valid])
    axes[1].set_xticklabels([labels[v[0]].replace("\n", " ") for v in valid], fontsize=10)
    axes[1].set_ylabel("score − IoU", fontsize=11)
    axes[1].set_title("Residual Before (blue) vs After (red) Calibration", fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch

    axes[1].legend(
        handles=[
            Patch(facecolor="steelblue", alpha=0.5, label="Before"),
            Patch(facecolor="tomato", alpha=0.5, label="After"),
        ],
        fontsize=9,
    )
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # 오른쪽图：ECE（Expected Calibration Error）校正前后对比
    def compute_ece(scores_arr, ious_arr, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (scores_arr >= lo) & (scores_arr < hi)
            if mask.sum() == 0:
                continue
            avg_score = scores_arr[mask].mean()
            avg_iou = ious_arr[mask].mean()
            ece += (mask.sum() / len(scores_arr)) * abs(avg_score - avg_iou)
        return ece

    ece_results = {"Size": [], "ECE Before": [], "ECE After": []}
    for i, (lo, hi, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
        mask = (areas >= lo) & (areas < hi)
        if mask.sum() < 30:
            continue
        ece_b = compute_ece(scores[mask], ious[mask])
        ece_a = compute_ece(scores_calibrated[mask], ious[mask])
        ece_results["Size"].append(label.replace("\n", " "))
        ece_results["ECE Before"].append(ece_b)
        ece_results["ECE After"].append(ece_a)

    x2 = np.arange(len(ece_results["Size"]))
    width = 0.35
    axes[2].bar(x2 - width / 2, ece_results["ECE Before"], width, label="ECE Before", color="steelblue", alpha=0.8)
    axes[2].bar(x2 + width / 2, ece_results["ECE After"], width, label="ECE After", color="tomato", alpha=0.8)
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(ece_results["Size"], fontsize=10)
    axes[2].set_ylabel("ECE (lower = better)", fontsize=11)
    axes[2].set_title("Expected Calibration Error\nBefore vs After", fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=9)
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "vfl_counterfactual_calibration.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ── 数值摘要 ────────────────────────────────────────────────────
    print("\n=== Counterfactual Calibration Summary ===")
    print(f"{'Size':<20} {'N':>8} {'ECE Before':>12} {'ECE After':>12} {'ECE Reduction':>15}")
    print("-" * 70)
    for size, ece_b, ece_a in zip(ece_results["Size"], ece_results["ECE Before"], ece_results["ECE After"]):
        reduction = (ece_b - ece_a) / ece_b * 100
        print(f"{size:<20} {ece_b:>12.4f} {ece_a:>12.4f} {reduction:>14.1f}%")


if __name__ == "__main__":
    model_size = "m"  # m, l
    dataset = "aitod"  # aitod, visdrone
    split = "train"  # val, train

    output_dir = Path(f"dome_{model_size}_{dataset}/{split}")
    status_file = output_dir / "vfl_stats.npz"

    data = np.load(status_file)

    ious = data["ious"]
    scores = data["scores"]
    areas = data["areas"]
    gradient_weights = data["gradient_weights"]
    residuals = data["residuals"]

    plot_vfl_status()
    analyze_counterfactual_calibration()
