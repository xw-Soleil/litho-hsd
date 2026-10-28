"""metrics.json -> results/RESULTS.md (Chinese summary with tables and figure links)."""
import json

from common import ROOT

R = json.loads((ROOT / "results" / "metrics.json").read_text())
L = []
w = L.append
pct = lambda v: f"{100 * v:.2f}%"


ROLE_N = {}


def role_table(br, title):
    w(f"\n{title}\n")
    for r, v in br.items():
        if "n" in v:
            ROLE_N[r] = v["n"]
    w("| 窗口类型 | 数量 | 判为 NHS | 判为 HS | 判为 NNHS |")
    w("|---|---:|---:|---:|---:|")
    for r, v in br.items():
        n = f"{v.get('n', ROLE_N.get(r, 0)):,}"
        w(f"| {r} | {n} | {pct(v['NHS'])} | {pct(v['HS'])} | {pct(v['NNHS'])} |")


def mc_row(name, rep, note=""):
    r = rep["recall"]
    w(f"| {name} | **{rep['macro_f1']:.3f}** | {rep['bal_acc']:.3f} | {pct(r['HS'])} | {pct(r['NHS'])} | {pct(r['NNHS'])} | {note} |")


w("# 光刻热点检测：方案一（Meta Learning）与方案二（概率阈值）实验结果\n")
w("> 本文件由 `code/make_report.py` 根据 `results/metrics.json` 自动生成。图在 `results/figures/`。\n")

# ------------------------------------------------------------------ data split
w("## 1. 数据划分（报告里要写清楚）\n")
w("| 部分 | 来源 | HS | NHS | NNHS | 用途 |\n|---|---|---:|---:|---:|---|")
w("| TRAIN | 官方 train 的 90% | 1,172 | 15,693 | 0 | 训练 |")
w("| VAL | 官方 train 的 10%（按设计×类别分层） | 131 | 1,743 | 0 | 监控训练、0-shot 阈值 |")
w("| CALIB | 官方 test 的 20%（**按邻域组**划分） | 550 | 7,100 | 21,600 | 方案二调阈值 / 方案一取 K 个 NNHS 样本 / 全监督参考的训练 |")
w("| EVAL | 官方 test 的 80% | 2,200 | 28,403 | 86,424 | **所有三分类指标都在这里报告** |")
w("\n一个“邻域组”= 1 个中心窗口 + 它的 8 个相邻窗口，同组样本不会同时出现在 CALIB 和 EVAL。\n")

# ------------------------------------------------------------------ binary
A = R["A_binary"]
w("## 2. 二分类热点检测（与参考论文对比）\n")
w("测试集 = ICCAD12 标准测试集（iccad2–5 合并，2,524 HS / 13,184 NHS），与 Geng et al. ICCAD'20 Table 2 的样本数一致。")
w("Accuracy = 热点检出率，FA = 被误报为热点的非热点个数。@0.5 列是固定阈值的真实结果；"
  "最后一列把阈值调到和论文相同的检出率再数误报，属于工作点对比（相当于在 ROC 曲线上取同一点），阈值用到了测试标签。\n")
w("| 方法 | AUC | Accuracy@0.5 | FA@0.5 | 在论文同等 Accuracy 下的 FA |\n|---|---:|---:|---:|---:|")
w(f"| Geng et al. ICCAD'20（参考论文） | – | 98.42% | 2481 | 2481 |")
for k, v in A.items():
    if k.startswith("binary_"):
        continue
    w(f"| 方案二 CNN {k} | {v['auc']:.4f} | {pct(v['at_default']['accuracy'])} | {v['at_default']['fa']} | "
      f"{v['fa_at_paper_accuracy']['fa']}（Acc {pct(v['fa_at_paper_accuracy']['accuracy'])}） |")
for k, v in R["C_scheme1"].items():
    b = v["binary"]
    w(f"| 方案一 原型网络 {k} | {b['auc']:.4f} | {pct(b['at_default']['accuracy'])} | {b['at_default']['fa']} | "
      f"{b['fa_at_paper_accuracy']['fa']}（Acc {pct(b['fa_at_paper_accuracy']['accuracy'])}） |")
w("\n单模型明细（表示方式消融：`dens`=120×120 面积平均密度图，`squish`=64×64 无损 squish 编码）：\n")
w("| 模型 | AUC | Acc@0.5 | FA@0.5 | 同等 Acc 下 FA |\n|---|---:|---:|---:|---:|")
for k, v in A.items():
    if k.startswith("binary_"):
        w(f"| {k} | {v['auc']:.4f} | {pct(v['at_default']['accuracy'])} | {v['at_default']['fa']} | {v['fa_at_paper_accuracy']['fa']} |")
best = max((k for k in A if not k.startswith("binary_")), key=lambda k: A[k]["auc"])
w(f"\n按 benchmark（{best}，阈值 0.5）：\n")
w("| benchmark | HS | NHS | Accuracy | FA |\n|---|---:|---:|---:|---:|")
for b, v in A[best]["per_bench"].items():
    w(f"| {b} | {v['n_hs']} | {v['n_non']} | {pct(v['accuracy'])} | {v['fa']} |")
w("\n注意：论文的 ICCAD12 测试图是 3.6×3.6 µm 的大图，本课程数据是 1.2×1.2 µm 的 clip，所以这是“同一基准、同一样本数、不同 clip 格式”的对比，"
  "汇报时要说明，不要直接宣称超过论文。\n")
w("![ROC](figures/binary_roc.png)\n")

# ------------------------------------------------------------------ three-class comparison
w("## 3. 三分类总表（EVAL，macro-F1 为主指标）\n")
w("| 方法 | macro-F1 | 平衡准确率 | HS 召回 | NHS 召回 | NNHS 召回 | 用到的 NNHS 标签 |\n|---|---:|---:|---:|---:|---:|---|")
for rep, v in R["B_scheme2"].items():
    mc_row(f"方案二（{rep}）概率双阈值", v["tuned_on_calib"]["eval"], f"CALIB 调阈值 p∈[{v['tuned_on_calib']['lo_p']:.3f}, {v['tuned_on_calib']['hi_p']:.3f}] 判 NNHS")
    mc_row(f"方案二（{rep}）固定区间 0.2–0.8", v["naive_band_02_08"], "无")
    if "uncertainty_aware" in v:
        mc_row(f"方案二（{rep}）+ 集成不确定度", v["uncertainty_aware"]["eval"], "CALIB 调阈值")
    mc_row(f"平凡基线（{rep}）：二分类 + 非热点全判 NNHS", v["trivial_all_nonHS_as_NNHS"], "无")
for k, v in R["C_scheme1"].items():
    mc_row(f"方案一 0-shot 开集（{k}）", v["zero_shot_openset"]["eval"], "**无**（阈值只用 VAL）")
    mc_row(f"方案一 0-shot 开集，阈值用 CALIB 调", v["zero_shot_openset_tau_tuned_on_calib"], "CALIB 调阈值")
    for K, ks in v["k_shot_all_classes_from_test_domain"].items():
        r = ks["recall_mean"]
        w(f"| 方案一 {K}-shot 原型（三类都取 K 个） | **{ks['macro_f1_mean']:.3f}** ± {ks['macro_f1_std']:.3f} | – | {pct(r['HS'])} | {pct(r['NHS'])} | {pct(r['NNHS'])} | 每个设计每类 {K} 个 |")
    for K, ks in v["k_shot_knn15"].items():
        r = ks["recall_mean"]
        w(f"| 方案一 {K}-shot kNN(15)（三类都取 K 个） | **{ks['macro_f1_mean']:.3f}** ± {ks['macro_f1_std']:.3f} | – | {pct(r['HS'])} | {pct(r['NHS'])} | {pct(r['NNHS'])} | 每个设计每类 {K} 个 |")
    for K, ks in v["k_shot"].items():
        if K not in ("10", "100"):
            continue
        r = ks["recall_mean"]
        w(f"| 方案一 {K}-shot（只取 NNHS，HS/NHS 原型用训练集） | {ks['macro_f1_mean']:.3f} ± {ks['macro_f1_std']:.3f} | – | {pct(r['HS'])} | {pct(r['NHS'])} | {pct(r['NNHS'])} | 每个设计 {K} 个 NNHS |")
for k, v in R["D_optionA"].items():
    mc_row(f"参考：全监督三分类（Option A）", v, "CALIB 全部 21,600 个")
if "G_context_analysis" in R:
    mc_row("分析：二分类 + 邻域上下文（**非纯图像**）", R["G_context_analysis"], "无，但用了文件名里的邻域结构")

# ------------------------------------------------------------------ scheme 2 detail
w("\n## 4. 方案二细节：NNHS 并不在概率的“中间”\n")
w("![prob](figures/s2_prob_hist.png)\n")
for rep, v in R["B_scheme2"].items():
    role_table(v["tuned_on_calib"]["by_role"], f"方案二（{rep}）各类窗口的去向：")
w("\n![grid](figures/s2_threshold_grid.png)\n")

# ------------------------------------------------------------------ scheme 1 detail
w("## 5. 方案一细节：原型网络（Prototypical Network）\n")
for k, v in R["C_scheme1"].items():
    z = v["zero_shot_openset"]
    w(f"- 0-shot：“离所有训练样本都远 → 没见过 → NNHS”，NNHS 与已知类的可分性 AUROC = {z['auroc_nnhs_vs_known']:.3f}")
    ksn = v["k_shot_knn15"]
    Kb = max(ksn, key=lambda K: ksn[K]["macro_f1_mean"])
    role_table(ksn[Kb]["by_role_mean"], f"方案一 {Kb}-shot kNN 各类窗口的去向（5 次平均）：")
w("\n![kshot](figures/s1_kshot_curve.png)\n\n![pca](figures/s1_embedding_pca.png)\n")

F = R.get("F_cross_node", {})
if "protonet_k_shot_per_class" in F:
    w("### 5.1 跨工艺节点：只用 28nm（iccad2–5）训练，迁移到 32nm（iccad1）\n")
    w(f"目标集：iccad1 测试中心窗 {F['n_target']['hs']} HS / {F['n_target']['nhs']} NHS。\n")
    w("| 方法 | 用到的 32nm 标签 | Accuracy | FA | AUC |\n|---|---|---:|---:|---:|")
    if "binary_trained_on_28nm_only" in F:
        b = F["binary_trained_on_28nm_only"]
        w(f"| 普通二分类 CNN（28nm 训练，直接用） | 0 | {pct(b['accuracy'])} | {b['fa']} | {b['auc']:.3f} |")
    for K, v in F["protonet_k_shot_per_class"].items():
        w(f"| 方案一 原型网络 {K}-shot | 每类 {K} 个 | {pct(v['accuracy_mean'])} ± {100*v['accuracy_std']:.1f} | {v['fa_mean']:.1f} ± {v['fa_std']:.1f} | {v['auc_mean']:.3f} |")
    for K, v in F.get("cnn_features_k_shot_per_class", {}).items():
        w(f"| 消融：普通 CNN 特征 + 同样的 {K}-shot 原型（无 meta 训练） | 每类 {K} 个 | {pct(v['accuracy_mean'])} ± {100*v['accuracy_std']:.1f} | {v['fa_mean']:.1f} ± {v['fa_std']:.1f} | {v['auc_mean']:.3f} |")
    if "reference_in_domain_binary" in F:
        b = F["reference_in_domain_binary"]
        w(f"| 参考：二分类 CNN 用全部 32nm 训练数据（99 HS + 340 NHS） | 439 | {pct(b['accuracy'])} | {b['fa']} | – |")

FS = R.get("F_cross_node_seeds")
if FS:
    w(f"\n训练随机种子稳健性（28nm → 32nm，{len(FS['seeds'])} 个种子，每个种子各自重新训练 CNN 和原型网络；5-shot）：\n")
    w("| 指标 | 均值 ± 标准差 | 各种子 |\n|---|---:|---|")
    names = {"cnn_direct_accuracy": "普通 CNN 直接用：Accuracy", "cnn_direct_auc": "普通 CNN 直接用：AUC",
             "proto_5shot_accuracy": "方案一 5-shot：Accuracy", "proto_5shot_fa": "方案一 5-shot：FA",
             "proto_5shot_auc": "方案一 5-shot：AUC", "cnnfeat_5shot_accuracy": "消融 CNN 特征 5-shot：Accuracy",
             "cnnfeat_5shot_fa": "消融 CNN 特征 5-shot：FA", "cnnfeat_5shot_auc": "消融 CNN 特征 5-shot：AUC"}
    for k, lab in names.items():
        v = FS[k]
        if k.endswith("_fa"):
            mean, sd, each = f"{v['mean']:.1f}", f"{v['std']:.1f}", [f"{x:.1f}" for x in v["per_seed"]]
        elif k.endswith("_auc"):
            mean, sd, each = f"{v['mean']:.3f}", f"{v['std']:.3f}", [f"{x:.3f}" for x in v["per_seed"]]
        else:
            mean, sd, each = f"{100 * v['mean']:.1f}%", f"{100 * v['std']:.1f} 个百分点", [f"{100 * x:.1f}%" for x in v["per_seed"]]
        w(f"| {lab} | {mean} ± {sd} | {', '.join(each)} |")

LO = R.get("F_lodo", {})
if LO:
    w("\n### 5.2 留一设计（leave-one-design-out）：每次去掉一个 28nm 设计训练，再用 5 个样本/类适配\n")
    w("| 留出的设计 | 目标集 HS / NHS | 普通 CNN 直接用 Acc / FA | 方案一 5-shot Acc / FA | 消融：CNN 特征 5-shot Acc / FA | 参考：用全部数据训练 Acc / FA |")
    w("|---|---|---|---|---|---|")
    rows = [("iccad1（32nm）", F)] + [(b, v) for b, v in LO.items()]
    for b, v in rows:
        n = v["n_target"]
        bt = v.get("binary_trained_on_28nm_only") or v.get("binary_trained_without_it")
        cell = lambda d: f"{pct(d['accuracy'])} / {d['fa']}" if d else "–"
        ks = lambda key: (lambda d: f"{pct(d['accuracy_mean'])} / {d['fa_mean']:.0f}" if d else "–")(v.get(key, {}).get("5") or v.get(key, {}).get(5))
        w(f"| {b} | {n['hs']} / {n['nhs']} | {cell(bt)} | {ks('protonet_k_shot_per_class')} | {ks('cnn_features_k_shot_per_class')} | {cell(v.get('reference_in_domain_binary'))} |")
    w("\n结论：在 28nm 设计之间互相迁移时，普通 CNN 本来就泛化得很好，meta learning 没有优势（误报还更多）；"
      "它的优势只出现在跨工艺节点（28nm → 32nm）、普通 CNN 完全失效的情况。")

# ------------------------------------------------------------------ option A + context
if R["D_optionA"]:
    w("\n## 6. 参考：全监督三分类（Option A）\n")
    for k, v in R["D_optionA"].items():
        role_table(v["by_role"], "全监督三分类各类窗口的去向：")
if "G_context_analysis" in R:
    role_table(R["G_context_analysis"]["by_role"], "\n二分类 + 邻域上下文各类窗口的去向（非纯图像分类，只用于说明标签含义）：")
w("\n![cm](figures/confusion.png)\n")

E = R["E_ceiling"]
w("## 7. 数据审计\n")
w(f"- EVAL 中 {E['eval_nnhs']:,} 个 NNHS 里，{E['identical_to_an_NHS_image']:,} 个（{pct(E['identical_to_an_NHS_image']/E['eval_nnhs'])}）"
  f"与某个 NHS 图像**逐像素完全相同**，{E['blank_clip']:,} 个是全黑空白图。仅凭图像，这些样本不可能被分对。")
w("- train 与 test 没有完全相同的图像（无泄漏）。")
w("- 训练图内容铺满 1200×1200；测试图内容只在 [88, 1112] 内（四周 88 px 黑边），线宽相同，说明测试图是裁剪而非缩放。")
w("- squish 64×64 编码对 100% 的图像无损（最多 57 条边）。")
w("\n## 8. 复现（conda 环境 torchTrain，Apple M4 Pro GPU；一次只跑一个训练任务，每个约 10–15 分钟）\n")
w("```bash\ncd code\npython preprocess.py ../data/iccad-official ../cache   # 1 分钟：密度图 + squish + 哈希\n"
  "python audit.py ../cache ../results                     # 数据审计\n"
  "python train_binary.py --rep dens --seed 0              # 方案二基模型（再跑 --seed 1 组成集成）\n"
  "python train_binary.py --rep squish --seed 0            # 表示方式消融\n"
  "python train_proto.py --rep dens --seed 0               # 方案一 原型网络\n"
  "python train_sup3.py --rep dens --seed 0                # 参考：全监督三分类\n"
  "python train_binary.py --rep dens --seed 0 --holdout iccad1   # 跨节点：普通 CNN\n"
  "python train_proto.py --rep dens --seed 0 --holdout iccad1    # 跨节点：原型网络\n"
  "python fig_explain.py && python evaluate.py && python make_report.py\n```")
w("\n\n在 xlab01 集群上（NVIDIA GB10，Slurm，student QOS：同时只能跑 1 个作业、单作业最长 4 小时）：\n")
w("```bash\n# 本地上传代码与 sbatch 脚本\nrsync -az --exclude __pycache__ code/ xlab01:~/work/dtco/code/ && rsync -az slurm/ xlab01:~/work/dtco/slurm/\n"
  "# 集群上：预处理 -> 自检 -> 全部训练（已完成的会自动跳过，可重复提交续跑）\n"
  "ssh xlab01 'cd ~/work/dtco && sbatch slurm/prep.sbatch'\n"
  "ssh xlab01 'cd ~/work/dtco && sbatch slurm/train_all.sbatch && sbatch slurm/seeds.sbatch'\n"
  "# 拉回结果后在本地评估\nslurm/sync_back.sh --models && cd code && python evaluate.py && python make_report.py\n```")
(ROOT / "results" / "RESULTS.md").write_text("\n".join(L))
print("written", ROOT / "results" / "RESULTS.md")
