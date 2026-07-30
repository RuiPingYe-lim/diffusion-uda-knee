# DOSC：诊断正交的目标风格条件化

DOSC（Diagnostic-Orthogonal Style Conditioning）把 UNSB 原来的随机风格噪声 \(z\) 改为由无标签目标参考图编码得到的条件码，并显式抑制其中可由源域标签识别的诊断信息。该模块用于 **BUSI（有标签源域）→ BrEaST（无标签目标域）** 的源图目标风格增强。

## 1. 设计目标

普通 UNSB 在每个残差块中使用随机 \(z\) 调制特征。已有 Gate 0 表明，BrEaST 个体间在去除全局均值和方差后仍保留明显的浅层纹理差异，而随机 \(z\) 无法对应这些真实目标风格。因此 DOSC 使用目标参考图 \(x_j^t\) 构造条件：

\[
s_j = E_s(x_j^t), \qquad
\widetilde{s}_j = P_{\perp\mathcal D}(s_j), \qquad
z_j = A(\widetilde{s}_j).
\]

其中：

- \(E_s\) 从多尺度浅层特征的逐通道均值与标准差提取散斑、增益和设备纹理；
- \(\mathcal D\) 是由有标签源样本风格码的 EMA 类中心构造的诊断子空间；
- \(P_{\perp\mathcal D}\) 将目标风格码投影到诊断子空间的正交补；
- \(A\) 将风格码映射到 UNSB 生成器要求的 \(4\times\text{ngf}\) 维条件向量。

二分类时，\(\mathcal D\) 就是两个源域类别风格中心之差所张成的一维子空间。多分类时最多删除 \(C-1\) 个方向。

## 2. 为什么不只使用梯度反转

乳腺超声中的采集纹理和病灶纹理高度耦合，仅靠一个类别对抗器容易出现“对抗器暂时失效，但风格码仍泄漏诊断信息”的情况。DOSC 同时使用：

1. **显式线性投影**：直接删除 EMA 类中心张成的诊断方向；
2. **诊断对抗头**：通过 GRL 抑制投影后残留的非线性诊断信息；
3. **域分类头**：要求投影后的编码仍能区分源域和目标域，避免把域信息一并剥除；
4. **参考实例对比**：目标图与其水平翻转视图为正对，历史目标参考码为负样本，防止所有目标风格坍缩为一个常量。

## 3. 完整目标函数

\[
\mathcal L =
\mathcal L_{\mathrm{UNSB}}
+\lambda_{\mathrm{diag}}\mathcal L_{\mathrm{diag\text{-}adv}}
+\lambda_{\mathrm{dom}}\mathcal L_{\mathrm{domain}}
+\lambda_{\mathrm{ins}}\mathcal L_{\mathrm{instance}}
+\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{style\text{-}rec}}
+\lambda_{\mathrm{safe}}\mathcal L_{\mathrm{safe}}.
\]

诊断安全项使用冻结的源域教师分类器：

\[
\mathcal L_{\mathrm{safe}}
=
\left[
m_y(x^s)-m_y(G(x^s,\widetilde{s}^t))-\delta
\right]_+ .
\]

它不是强制翻译图与原图预测完全一致，而是允许 \(\delta\) 范围内的 margin 变化，惩罚超过容忍度的真实类别 margin 下降。该损失默认在 1000 个优化步内线性升温，避免随机初始化的生成器被教师过早锁死。

## 4. 严格 UDA 边界

- 训练只读取源域标签；
- 目标 CSV 即使含有标签列，数据构建器也不会把目标标签写入 UNSB 数据集；
- 诊断子空间、诊断对抗头和安全 margin 均只使用源域标签；
- 目标图只提供无标签风格参考和域标识；
- 源域验证集用于分类器与超参数选择，目标测试标签只用于最终一次评估。

## 5. 安装到官方 UNSB

当前 overlay 面向官方 `cyclomon/UNSB` 的 `main` API（核对版本：`d1f644f`）。它不会修改上游 `sb_model.py`，而是在 `models/` 和 `data/` 下新增独立类型。

```bash
python scripts/install_dosc_unsb_overlay.py \
  --unsb_root /root/autodl-tmp/UNSB
```

安装后新增：

- `models/dosc_modules.py`
- `models/dosc_sb_model.py`
- `data/dosc_unaligned_dataset.py`

## 6. 构建 BUSI→BrEaST 数据

```bash
python scripts/build_breast_dosc_unsb_dataset.py \
  --source_train_csv /path/to/busi_train.csv \
  --source_val_csv /path/to/busi_valid.csv \
  --target_train_csv /path/to/breast_train.csv \
  --out_root /root/autodl-tmp/UNSB/datasets/busi_to_breast_dosc
```

目录语义固定为：

| 目录 | 内容 | 标签用途 |
|---|---|---|
| `trainA` | BUSI source train | 读取真实源标签 |
| `trainB` | BrEaST target train | 不读取标签 |
| `testA` | BUSI source train + valid | 生成目标风格训练候选 |
| `testB` | BrEaST target-train references | 只提供风格条件 |

`trainA_manifest.csv` 是训练时唯一标签入口。
目标验证集和目标测试集不会进入 `trainB` 或 `testB`，避免用最终评价病例生成分类器训练候选。

## 7. 导出诊断教师

```bash
python scripts/export_diagnostic_teacher.py \
  --weights /path/to/busi_source_classifier/best_checkpoint.pt \
  --backbone custom_resnet50_space \
  --out /root/autodl-tmp/busi_teacher.ts
```

导出的 TorchScript 模型直接接收 UNSB 的 `[-1,1]` 图像张量，并在内部调整到分类器输入尺寸。

## 8. 训练与生成

```bash
export UNSB_ROOT=/root/autodl-tmp/UNSB
export DOSC_DATA_ROOT=/root/autodl-tmp/UNSB/datasets/busi_to_breast_dosc
export DOSC_TEACHER=/root/autodl-tmp/busi_teacher.ts
bash scripts/run_unsb_dosc_breast.sh
```

推荐先保持 `K=2–3` 个目标参考风格，不要直接扩到 8 个。当前脚本以确定性参考条件生成一个候选；多候选实验应改变 `testB` 的参考排列或参考簇，而不是把 `fake_1…fake_5` 解释为五种独立风格。

## 9. 必做消融

| 版本 | 命令设置 | 回答的问题 |
|---|---|---|
| 原始 UNSB | `--model sb` | 随机 \(z\) 基线 |
| 目标范例条件化 | `--dosc_disable_projection --lambda_DOSC_diag 0` | 真实目标风格是否优于随机 \(z\) |
| 仅显式正交投影 | `--lambda_DOSC_diag 0` | 线性诊断方向删除是否有效 |
| 投影 + 对抗 | 默认投影与 `lambda_DOSC_diag>0` | 非线性泄漏抑制是否必要 |
| 完整 DOSC | 默认配置 + 教师 | 是否兼顾目标风格、个体差异与诊断安全 |

多候选方法必须保证每个源样本的总训练权重一致，避免把“更多训练步数”误判成选择或条件化收益。

## 10. 训练日志判据

- `DOSC_domain_acc` 应明显高于随机，证明域信息没有被投影剥光；
- `DOSC_diag_acc` 只能作为在线对抗状态，最终必须另训冻结探针验证风格码诊断 AUC；
- `DOSC_removed` 应非零但不应长期接近 1，否则删除范围过大；
- `DOSC_margin_drop` 应逐步压到容忍度附近；
- 最终判定仍以同一患者划分下的目标 AUC、配对 bootstrap CI 和三个随机种子为准。
