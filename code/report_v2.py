"""Render the audited v2 results with fixed comparisons and explicit label budgets."""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS = {
    ("binary", "nonhs_as_nnhs"): ("基线：非热点全判 NNHS", "Non-HS to NNHS"),
    ("binary", "fixed_band_02_08"): ("方案二：固定区间 0.2–0.8", "Fixed score band"),
    ("binary", "two_threshold"): ("方案二：双阈值（仅 TUNE 拟合）", "Tuned score band"),
    ("proto", "zero_label_prototype_rejection"): ("方案一：零 NNHS 标签的原型距离拒识", "Zero-label rejection"),
    ("binary", "prototype"): ("普通 CNN 特征 + 原型", "CNN + prototypes"),
    ("proto", "prototype"): ("元学习特征 + 原型", "Meta + prototypes"),
    ("binary", "knn15"): ("普通 CNN 特征 + kNN(15)", "CNN + kNN"),
    ("proto", "knn15"): ("元学习特征 + kNN(15)", "Meta + kNN"),
    ("sup3", "supervised"): ("全监督三分类参考", "Supervised reference"),
    ("binary", "direct_no_target_labels"): ("普通 CNN 直接迁移（目标标签为零）", "CNN, no target labels"),
}
PRIMARY = [("binary", "nonhs_as_nnhs", None), ("binary", "fixed_band_02_08", None),
           ("binary", "two_threshold", None), ("proto", "zero_label_prototype_rejection", None),
           ("binary", "prototype", 5), ("proto", "prototype", 5), ("sup3", "supervised", None)]
CONFUSION_METHODS = [("binary", "two_threshold", None), ("binary", "prototype", 5),
                     ("proto", "prototype", 5), ("sup3", "supervised", None)]


def label(row, english=False):
    name = METHODS.get((row["mode"], row["method"]), (row["method"], row["method"]))[int(english)]
    if row.get("requested_k") is not None:
        name += f" (K≤{row['requested_k']})"
    return name


def val(v, percent=False, digits=3):
    return "—" if v is None else f"{100 * v:.1f}%" if percent else f"{v:.{digits}f}"


def stat(row, key, percent=False, digits=3):
    mean, sd = row["mean"][key], row["training_seed_std"][key]
    if len(row["training_seeds"]) == 1:
        return val(mean, percent, digits) + "（单种子）"
    if mean is None:
        return "—"
    return (f"{100 * mean:.1f}% ± {100 * sd:.1f} pp" if percent
            else f"{mean:.{digits}f} ± {sd:.{digits}f}")


def main(args):
    out = Path(args.out)
    metrics = json.loads((out / "metrics.json").read_text())
    ag = metrics["aggregates"]
    lines = []
    write = lines.append
    lookup = {(x["task"], x["representation"], x["mode"], x["method"], x.get("requested_k")): x for x in ag}
    mainrows = [lookup[("three_class", "densm", *k)] for k in PRIMARY if ("three_class", "densm", *k) in lookup]
    write("# 光刻热点检测：修正数据协议后的复现实验（v2）\n")
    write("本报告由冻结的方法清单自动生成。旧实验保留在上一级目录，两个版本的测试集不同，分数不应直接相减来衡量算法提升。")
    write("这份数据在旧项目中已经被探索过；本次是固定协议后的重新训练与审计评估，不能称为从未接触过的外部测试集。\n")
    write(f"Manifest SHA-256：`{metrics['protocol_sha256']}`。完整指标与每次支持集抽样结果见 [metrics.json](metrics.json)，逐样本预测位于 `predictions/`，支持样本编号位于 `support_plans/`。\n")
    write("## 1. 数据边界和评估规则\n")
    write("| 分区 | NHS | HS | NNHS |\n|---|---:|---:|---:|")
    for part in ("train", "val", "support", "tune", "test", "excluded"):
        row = metrics["counts"].get(part, {})
        write(f"| {part} | {row.get('NHS', 0):,} | {row.get('HS', 0):,} | {row.get('NNHS', 0):,} |")
    write("\n划分按邻域和重复关系隔离，并检查统一裁剪、D4 变换后的图像等价关系。官方训练数据关联的测试组件被排除，因而这是受限的去重基准，不代表原始全部测试样本的性能。具体排除规则及数量以 `experiments/v2/audit.json` 为准。")
    source_n = metrics.get("source_test_count", 0)
    excluded_n = metrics.get("excluded_source_test_count", 0)
    if source_n:
        write(f"\n原官方测试来源共 {source_n:,} 张，其中排除 {excluded_n:,} 张（{excluded_n / source_n:.1%}）；保留 {source_n - excluded_n:,} 张再分为 SUPPORT、TUNE、TEST。")
    structure = metrics.get("test_structure")
    if structure:
        write(f"\n最终 TEST 有 {structure['rows']:,} 张图、{structure['neighbourhoods']:,} 个邻域、{structure['components']:,} 个独立划分组件；全黑图 {structure['blank_rows_n_edges_zero']:,} 张。角色分布如下：\n")
        write("| 角色 | 样本数 |\n|---|---:|")
        for role, count in structure["counts_by_role"].items():
            write(f"| {role} | {count:,} |")
        conflict = structure["raw_hash_conflicts"]
        write(f"\nTEST 内仍有 {conflict['groups']:,} 组原始图像冲突标签，涉及 {conflict['rows']:,} 张图。对仅以图像为输入的确定性分类器，至少 {conflict['minimum_unavoidable_errors']:,} 个标签无法同时满足；这不是 Macro-F1 上限，也不意味着所有相关 NNHS 都必错。")
    write("\n- 二分类和元训练只用 TRAIN；全监督三分类用 TRAIN + SUPPORT。")
    write("- 双阈值只在 TUNE 上选定；零标签拒识阈值只在 VAL 上选定；最终 TEST 不用于选方法、K 或训练轮次。")
    write("- 原型主结果固定 K=5；K=1、10、20 是预先声明的敏感性分析。kNN 的 K=20、100 是额外标签预算实验，不将其包装成少量标签的主结果。")
    write("- 每个设计每类最多 K 个支持样本；类别不足时使用实际数量，下面单独列出。CNN 与元学习方法复用完全相同的支持样本编号。")
    write("- 各模型独立使用自己的 TUNE 阈值；以下汇总单模型结果，没有混入集成模型。\n")
    write("均值 ± 标准差表示各训练种子结果的均值和标准差（ddof=0）。每个种子的少样本结果先对 30 次支持集抽样取平均；支持集抽样方差另列，不当成 30 个独立训练实验。单种子表示方式对照不宣称稳健性。\n")
    write("## 2. 三分类主结果（固定 densm 表示）\n")
    write("| 方法 | 目标域标签数 | Macro-F1 | HS 召回 | NHS 召回 | NNHS 召回 | 训练种子 |\n|---|---:|---|---|---|---|---|")
    for row in mainrows:
        pc = row["per_class_mean"]
        write(f"| {label(row)} | {row['target_label_budget']:,} | {stat(row, 'macro_f1')} | {val(pc['HS']['recall'], True)} | {val(pc['NHS']['recall'], True)} | {val(pc['NNHS']['recall'], True)} | {', '.join(map(str, row['training_seeds']))} |")
    write("\n标签预算分别指实际提供的支持标签、TUNE 拟合标签或全监督 SUPPORT 标签；这些预算不同的方法并非等监督条件对照。公平的元学习对照是表中的两个同 K 原型方法。\n")
    cnn_main = lookup.get(("three_class", "densm", "binary", "prototype", 5))
    meta_main = lookup.get(("three_class", "densm", "proto", "prototype", 5))
    if cnn_main and meta_main and cnn_main["target_label_budget"] == meta_main["target_label_budget"]:
        conclusion = ("本次三分类主结果没有显示元学习特征相对普通 CNN 特征的优势。"
                      if meta_main["mean"]["macro_f1"] <= cnn_main["mean"]["macro_f1"] else
                      "本次三分类主结果中，元学习方法的平均 Macro-F1 更高；这仍是两种完整方法的比较。")
        write(f"在预先固定的 densm、K=5 比较中，两种方法使用相同的 {cnn_main['target_label_budget']} 个目标域标签："
              f"普通 CNN 特征 + 原型的 Macro-F1 为 {cnn_main['mean']['macro_f1']:.3f}，"
              f"元学习特征 + 原型为 {meta_main['mean']['macro_f1']:.3f}。{conclusion}\n")
    write("| 方法 | NHS precision / recall / F1 | HS precision / recall / F1 | NNHS precision / recall / F1 |\n|---|---|---|---|")
    for row in mainrows:
        cells = [" / ".join(val(row["per_class_mean"][c][k]) for k in ("precision", "recall", "f1")) for c in ("NHS", "HS", "NNHS")]
        write(f"| {label(row)} | " + " | ".join(cells) + " |")
    write("\n![三分类主结果](figures/main_three_class.png)\n")
    write("以下四个面板固定为双阈值、普通 CNN + 5-shot 原型、元学习 + 5-shot 原型、全监督参考，均使用 densm。先对每个训练种子的支持集抽样混淆矩阵取平均，再对训练种子等权平均，最后逐行归一化；每行表示同一真实类别被分到三个预测类别的比例。双阈值和全监督方法每个种子只有一次预测，不虚构支持抽样。\n")
    write("![四方法混淆矩阵：跨训练种子和支持抽样的逐行归一化均值](figures/confusion_four_methods.png)\n")
    write("## 3. 同标签预算的支持集实验\n")
    write("| 表示 | 方法 | 实际总标签数 | Macro-F1（训练种子均值 ± SD） | 平均支持抽样 SD |\n|---|---|---:|---|---:|")
    for row in ag:
        if row["task"] == "three_class" and row["method"] in ("prototype", "knn15"):
            write(f"| {row['representation']} | {label(row)} | {row['target_label_budget']} | {stat(row, 'macro_f1')} | {row['support_sampling_std_mean']['macro_f1']:.3f} |")
    budgets = {}
    for row in ag:
        if row.get("actual_support_per_design_class"):
            budgets[(row["task"], row["requested_k"])] = row["actual_support_per_design_class"]
    write("\n实际每设计每类支持样本数：\n")
    write("| 任务 | 请求 K | 实际计数（设计/类别） |\n|---|---:|---|")
    for (task, K), counts in sorted(budgets.items()):
        write(f"| {task} | {K} | " + "; ".join(f"{k}: {v}" for k, v in counts.items()) + " |")
    write("\n同一设计、同一类别内部按 dup_key 去重采样，避免将重复文件计为独立支持样本。不同类别可能有相同图像但冲突标签；这部分不被悄悄删除，完整 trial 的唯一图像数和冲突 key 数均保存在 metrics.json。\n")
    write("| 任务 | 请求 K | 每次标签数 | 每次唯一输入数范围 | 每次冲突输入数范围 |\n|---|---:|---:|---|---|")
    seen = set()
    for row in ag:
        key = (row["task"], row.get("requested_k"))
        if row.get("unique_support_inputs_per_trial") is None or key in seen:
            continue
        seen.add(key)
        uniq = row["unique_support_inputs_per_trial"]
        conflicts = row["conflicting_support_keys_per_trial"]
        write(f"| {key[0]} | {key[1]} | {row['target_label_budget']} | {min(uniq)}–{max(uniq)} | {min(conflicts)}–{max(conflicts)} |")
    write("\n## 4. 跨节点二分类迁移\n")
    write("28nm 数据训练后，在 32nm 目标设计上评估。无目标标签的直接迁移与有支持标签的方法单独标明；不会把两者的差值全归因于元学习。这里只测试一个目标节点/设计，不能推断普遍的跨节点优势。\n")
    write("跨节点目标单独由 manifest.cross_target 定义：该设计所有不关联官方训练图的合格中心窗，包含主任务 SUPPORT、TUNE、TEST 分区中的样本。对于跨节点模型，这些目标图及标签都没有参与训练或阈值校准；支持样本来自 manifest.cross_support，即该设计的官方训练数据中保留在 TRAIN 的部分。\n")
    cross = [r for r in ag if r["task"].startswith("cross_") and (r["method"] == "direct_no_target_labels" or r.get("requested_k") == 5)]
    write("| 表示 | 方法 | 目标标签数 | HS 召回 | FA | FPR | ROC-AUC | 目标 HS / NHS |\n|---|---|---:|---|---|---|---|---|")
    for row in cross:
        write(f"| {row['representation']} | {label(row)} | {row['target_label_budget']} | {stat(row, 'hs_recall', True)} | {stat(row, 'false_alarms', digits=1)} | {stat(row, 'false_positive_rate', True)} | {stat(row, 'roc_auc')} | {row['n_per_class'].get('HS', 0)} / {row['n_per_class'].get('NHS', 0)} |")
    cnn_cross = lookup.get(("cross_iccad1", "densm", "binary", "prototype", 5))
    meta_cross = lookup.get(("cross_iccad1", "densm", "proto", "prototype", 5))
    if cnn_cross and meta_cross and cnn_cross["target_label_budget"] == meta_cross["target_label_budget"]:
        components = meta_cross.get("component_bootstrap_ci95", {}).get("components")
        component_text = f"、{components} 个组件" if components is not None else ""
        write(f"\n在相同的 {meta_cross['target_label_budget']} 个目标节点标签下，元学习方法的 HS 召回为 "
              f"{val(meta_cross['mean']['hs_recall'], True)}，普通 CNN 特征方法为 {val(cnn_cross['mean']['hs_recall'], True)}；"
              f"FPR 分别为 {val(meta_cross['mean']['false_positive_rate'], True)} 和 {val(cnn_cross['mean']['false_positive_rate'], True)}。"
              f"该观察仅限 iccad1 这一个目标设计的 {meta_cross['n']} 张查询图{component_text}，不能推广为所有工艺节点上的优势。\n")
    write("\n这里 HS 召回就是此前报告所谓的 Accuracy，不能与整体分类准确率混用。FA 是错误报为热点的非热点数量；FPR = FA / 非热点数量。\n")
    write("对跨节点主结果，另按 component_id 整组重采样 1,000 次，给出 95% percentile bootstrap 区间。每张查询图的热点预测指示先跨训练种子和支持抽样取平均；区间反映固定模型和支持抽样下的目标样本变动，不包含重新训练的全部不确定性，也不代表跨更多工艺节点的置信区间。\n")
    write("| 方法 | 目标组件数 | HS 召回 95% 区间 | FPR 95% 区间 |\n|---|---:|---|---|")
    for row in cross:
        ci = row.get("component_bootstrap_ci95")
        if ci:
            cells = [f"{100 * ci[k]['low']:.1f}%–{100 * ci[k]['high']:.1f}%" for k in ("hs_recall", "false_positive_rate")]
            write(f"| {label(row)} | {ci['components']} | {cells[0]} | {cells[1]} |")
    if cross:
        write("![跨节点公平对照](figures/cross_node.png)\n")
    write("## 5. 二分类中心窗口与表示对照\n")
    write("| 表示 | 阈值 | HS 召回 | FA | FPR | AUC | HS / NHS |\n|---|---|---|---|---|---|---|")
    for row in ag:
        if row["task"] == "binary_centers":
            write(f"| {row['representation']} | 固定 0.5 | {stat(row, 'hs_recall', True)} | {stat(row, 'false_alarms', digits=1)} | {stat(row, 'false_positive_rate', True)} | {stat(row, 'roc_auc')} | {row['n_per_class']['HS']} / {row['n_per_class']['NHS']} |")
    write("\n此处不与论文原表的 FA 直接排名：图像格式、数据划分、样本数和训练协议不一致。\n")
    write("| 表示 | 方法（预先固定） | Macro-F1 | 训练种子 |\n|---|---|---|---|")
    for row in ag:
        if row["task"] == "three_class" and (row["method"] in ("two_threshold", "supervised") or (row["mode"] == "proto" and row["method"] == "prototype" and row.get("requested_k") == 5)):
            write(f"| {row['representation']} | {label(row)} | {stat(row, 'macro_f1')} | {row['training_seeds']} |")
    dens_reference = lookup.get(("three_class", "dens", "sup3", "supervised", None))
    if dens_reference and len(dens_reference["training_seeds"]) == 1:
        write(f"\ndens 全监督参考得到 Macro-F1 {dens_reference['mean']['macro_f1']:.3f}，但只运行了 "
              f"{len(dens_reference['training_seeds'])} 个训练种子，不能据此宣称它稳健优于 densm。"
              "主比较仍采用预先固定的 densm 多种子结果，不因表示对照的分数而更换主结果。\n")
    write("\n## 6. 结果解释的边界\n")
    write("- 相同图像存在冲突标签，意味着无法同时正确满足所有冲突标签；不意味着所有相关 NNHS 都必错。")
    write("- 全监督三分类是经验参考，不是数学上限；二维投影也不能证明高维特征完全不可分。")
    write("- CNN 特征为 256 维、元学习嵌入为 128 维，训练目标和训练预算也不同。相同支持集的结果是两种完整方法的比较，不能将差值严格因果归因于元学习损失一项。")
    write("- 概率区间实验使用模型分数，没有声称 softmax 是经过校准的真实物理风险概率。")
    write("- 统一裁剪/D4 等价和邻域检查只能排除已建模的重复关系，不能证明不存在任意平移重叠或同源版图片段；当前缺少原始版图坐标。")
    write("- 按设计及中心/邻域角色的全部分项指标、混淆矩阵和每次支持抽样指标保存在 metrics.json；不能只根据一个总分忽略 NHS 识别失败。")
    (out / "RESULTS.md").write_text("\n".join(lines))
    make_figures(mainrows, cross, metrics["entries"], out)
    print("saved", out / "RESULTS.md")


def make_figures(rows, cross, entries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    confusion_figure(entries, figdir, plt)
    if rows:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        positions = np.arange(len(rows))[::-1]
        colors = ["#777777" if r["method"] == "nonhs_as_nnhs" else "#a85915" if r["mode"] == "binary" else "#26639c" if r["mode"] == "proto" else "#555555" for r in rows]
        ax.barh(positions, [r["mean"]["macro_f1"] for r in rows], height=0.64, color=colors,
                xerr=[r["training_seed_std"]["macro_f1"] for r in rows], capsize=3)
        for pos, row in zip(positions, rows):
            ax.text(row["mean"]["macro_f1"] + row["training_seed_std"]["macro_f1"] + 0.01, pos,
                    f"{row['mean']['macro_f1']:.3f}", va="center")
        ax.set_yticks(positions, [label(r, True) for r in rows])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Test macro-F1 (error bars: training-seed SD)")
        ax.set_title("Audited three-class comparison | fixed densm representation", loc="left")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(figdir / "main_three_class.png", dpi=180)
        plt.close(fig)
    chosen = [r for r in cross if r["representation"] == "densm"]
    if chosen:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
        for ax, metric, title in zip(axes, ["hs_recall", "false_positive_rate"], ["Hotspot recall", "False-positive rate"]):
            vals = np.array([r["mean"][metric] for r in chosen]) * 100
            std = np.array([r["training_seed_std"][metric] for r in chosen]) * 100
            ax.bar(np.arange(len(chosen)), vals, yerr=std, capsize=4, color=["#777777" if r["method"] == "direct_no_target_labels" else "#a85915" if r["mode"] == "binary" else "#26639c" for r in chosen])
            ax.set_xticks(np.arange(len(chosen)), [label(r, True).replace(" + ", "\n+ ") for r in chosen], fontsize=8)
            ax.set_ylim(0, 110)
            ax.set_ylabel(title + " (%)")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            for i, v in enumerate(vals):
                ax.text(i, v + std[i] + 2, f"{v:.1f}%", ha="center")
        fig.suptitle("28nm to 32nm | identical five-shot supports for both feature models\n"
                     "Error bars: training-seed SD", fontsize=12)
        fig.tight_layout()
        fig.savefig(figdir / "cross_node.png", dpi=180)
        plt.close(fig)


def confusion_figure(entries, figdir, plt):
    """Fixed panels; average draws within seed and then give seeds equal weight."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), layout="constrained")
    classes = ["NHS", "HS", "NNHS"]
    image = None
    for ax, (mode, method, k) in zip(axes.flat, CONFUSION_METHODS):
        selected = [row for row in entries if row["task"] == "three_class" and
                    row["representation"] == "densm" and row["mode"] == mode and
                    row["method"] == method and row.get("requested_k") == k]
        title = label(dict(mode=mode, method=method, requested_k=k), english=True)
        if not selected:
            ax.text(.5, .5, "Result not available", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title, loc="left")
            ax.set_axis_off()
            continue
        matrices = np.array([row["summary"]["confusion_mean"] for row in selected], float)
        if matrices.shape[1:] != (3, 3) or not np.isfinite(matrices).all():
            raise ValueError(f"Invalid three-class confusion matrix: {mode}/{method}")
        # Every seed must evaluate the same target population before averaging.
        if not np.allclose(matrices.sum(axis=2), matrices[0].sum(axis=1)):
            raise ValueError(f"Different target class counts across seeds: {mode}/{method}")
        mean_cm = matrices.mean(axis=0)
        row_total = mean_cm.sum(axis=1, keepdims=True)
        percentages = np.divide(100 * mean_cm, row_total, out=np.zeros_like(mean_cm), where=row_total > 0)
        image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        for true in range(3):
            for predicted in range(3):
                value = percentages[true, predicted]
                ax.text(predicted, true, f"{value:.1f}%", ha="center", va="center", fontsize=13,
                        color="white" if value > 55 else "#17212b")
        trials = sorted({row["summary"]["trials"] for row in selected})
        trial_text = "/".join(map(str, trials))
        seeds = len(selected)
        n = int(round(mean_cm.sum()))
        ax.set_title(title, fontsize=11, loc="left", pad=27)
        ax.text(0, 1.015, f"{seeds} seeds | {trial_text} predictions/seed | n={n:,}",
                transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom")
        ax.set_xticks(range(3), classes)
        ax.set_yticks(range(3), classes)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.grid(False)
        ax.tick_params(length=0)
    fig.suptitle("Three-class confusion matrices | fixed densm comparisons\n"
                 "Row-normalized means over training seeds and support draws", fontsize=13)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=.82, fraction=.035, pad=.03)
        colorbar.set_label("Mean row share (%)")
    fig.savefig(figdir / "confusion_four_methods.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results/v2"))
    main(parser.parse_args())
