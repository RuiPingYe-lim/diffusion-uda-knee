# TRSC：目标参考风格条件化与输出安全审计

TRSC（Target-Reference Style Conditioning）把 UNSB 的随机风格噪声 \(z\) 改为由无标签目标参考图编码得到的条件码，用于 **BUSI（有标签源域）→ BrEaST（无标签目标域）** 的源图目标风格增强。该模块最初使用 DOSC（Diagnostic-Orthogonal Style Conditioning）命名；因冻结探针不支持“诊断正交已经实现”，该名称停止用于方法主张，仅作为旧命令、检查点和日志前缀保留。

## 0. 当前证据边界

最终因果交换审计同时测量潜变量可解码性和生成输出：

| 设置 | 诊断 AUC（raw / 投影 / 条件） | 域 AUC | U1 参考引起的分数 std | U1 病例级斜率 | 参考置换 \(p\) |
|---|---:|---:|---:|---:|---:|
| `safe0` | 0.651 / 0.652 / 0.633 | 0.933 | 0.0029 | +0.0001 [−0.0003, +0.0004] | 0.86 |
| `safe05` | 0.748 / 0.763 / 0.731 | 0.930 | 0.0048 | −0.0003 [−0.0004, −0.0001] | 0.62 |

随机 AUC 基准为 0.5。域信息得到保留，诊断信息仍可线性解码；投影后 AUC 没有下降。另一方面，同一源图更换 12 个目标参考时，U1 教师分数波动只有 0.0029–0.0048，而独立路径噪声在 U5 产生约 0.39–0.43 的波动；参考分数斜率的置换检验均不显著。当前严格结论是：

- **目标参考风格条件化有效**：风格码保留强域信息，并显著提高类条件域覆盖；
- **潜变量诊断正交性不成立**：投影与 GRL 未优于无保护描述子；
- **未发现生成器因果使用泄漏改变诊断**：潜变量可解码没有传导为参考驱动的输出诊断变化；
- **CIDP 没有稳定下游收益**：它可作为输出层安全消融，但不再进入默认模型；
- **不加入风格交换一致性损失**：当前没有需要修复的参考驱动诊断波动，额外约束只会增加忽略条件的风险。

因此，默认方法只保留 **TRSC core**；CIDP、投影和 GRL 均关闭并仅作为消融。新命令使用 `scripts/eval_trsc_style_swap.py`；旧文件名仅为既有服务器命令兼容保留。

## 1. 设计目标

普通 UNSB 在每个残差块中使用随机 \(z\) 调制特征。已有 Gate 0 表明，BrEaST 个体间在去除全局均值和方差后仍保留明显的浅层纹理差异，而随机 \(z\) 无法对应这些真实目标风格。因此 TRSC 使用目标参考图 \(x_j^t\) 构造默认条件：

\[
s_j = E_s(x_j^t), \qquad
z_j = A(s_j).
\]

其中：

- \(E_s\) 从多尺度浅层特征的逐通道均值与标准差提取散斑、增益和设备纹理；
- \(A\) 将风格码映射到 UNSB 生成器要求的 \(4\times\text{ngf}\) 维条件向量。

旧投影臂额外计算 \(\widetilde{s}_j=P_{\perp\mathcal D}(s_j)\)。二分类时它只删除两个源域类别风格中心之差张成的一维方向；实验表明该操作没有降低冻结探针泄漏，因此不再进入默认路径。

## 2. 投影与梯度反转仅作消融

历史 DOSC 实现包含：

1. **显式线性投影**：直接删除 EMA 类中心张成的诊断方向；
2. **诊断对抗头**：通过 GRL 抑制投影后残留的非线性诊断信息；
3. **域分类头**：要求投影后的编码仍能区分源域和目标域，避免把域信息一并剥除；
4. **参考实例对比**：目标图与其水平翻转视图为正对，历史目标参考码为负样本，防止所有目标风格坍缩为一个常量。

前两项已经没有通过冻结探针闸门，默认配置设为 `projection=off`、`\(\lambda_{\mathrm{diag}}=0\)`。域分类、参考实例对比和风格重构继续用于学习非坍缩的目标参考条件。

## 3. 完整目标函数

\[
\mathcal L_{\mathrm{TRSC\text{-}core}} =
\mathcal L_{\mathrm{UNSB}}
+\lambda_{\mathrm{dom}}\mathcal L_{\mathrm{domain}}
+\lambda_{\mathrm{ins}}\mathcal L_{\mathrm{instance}}
+\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{style\text{-}rec}}.
\]

CIDP、\(\lambda_{\mathrm{diag}}\mathcal L_{\mathrm{diag\text{-}adv}}\) 和线性投影仅在旧控制消融中加入。

### 3.1 为什么废弃绝对 margin

令冻结教师的二分类分数为

\[
d(x)=z_1(x)-z_0(x),\qquad m_y(x)=(2y-1)d(x).
\]

旧安全项直接计算

\[
\left[m_y(x^s)-m_y(G(x^s))-\delta\right]_+.
\]

独立 BUSI 测试表明，U1 同时存在校准漂移和局部排序损失。旧教师在 U1 上对良性和恶性的平均安全罚分别为 0.101 和 6.399，形成 63 倍类别不对称；更换渲染鲁棒教师后仍为 0.509 和 1.276。原因不是 256 像素渲染，而是绝对 margin 对诊断分数的整体平移和温度变化敏感。因此历史 `safe05` 臂把安全项改为 CIDP（Calibration-Invariant Diagnostic Preservation，校准不变诊断保持）；完成下游审计后该项已降为消融。

### 3.2 停止梯度的正仿射校准

记源图与翻译图的教师分数为 \(d_i^s\) 和 \(d_i^u\)。CIDP 在类别平衡 FIFO 队列上拟合

\[
(a^\star,b^\star)
=
\arg\min_{a>0,b}
\sum_i w_{y_i}
\left(a d_i^u+b-d_i^s\right)^2,
\]

其中两类具有相同总权重。\(a^\star,b^\star\) 只由 `detach()` 后的分数估计并限制 \(a^\star>0\)，因此校准器不能与生成器共同降低损失。校准后的分数为

\[
\widetilde d_i^u=a^\star d_i^u+b^\star.
\]

默认队列保存 128 个样本，每类容量相同；两类各累计至少 8 个样本前，CIDP 不产生训练梯度。

### 3.3 校准 margin 与成对 rank 非劣约束

校准 margin 项为

\[
\mathcal L_{\mathrm{margin}}
=
\frac{1}{B}\sum_i
\left[
m_{y_i}^s-\widetilde m_{y_i}^u-\delta_m
\right]_+.
\]

对源教师排序正确的正负样本对 \(\mathcal R=\{(p,n):d_p^s>d_n^s\}\)，rank 项为

\[
\mathcal L_{\mathrm{rank}}
=
\frac{1}{|\mathcal R|}
\sum_{(p,n)\in\mathcal R}
\left[
(d_p^s-d_n^s)
-
(\widetilde d_p^u-\widetilde d_n^u)
-\delta_r
\right]_+.
\]

最终

\[
\mathcal L_{\mathrm{CIDP}}
=
\mathcal L_{\mathrm{margin}}
+\lambda_r\mathcal L_{\mathrm{rank}}.
\]

正仿射校准吸收阈值和温度漂移；rank 项处理任何阈值移动都无法恢复的局部可分性损失。排序项同时使用当前 batch 与历史类别平衡队列，因此 batch size 为 4 时仍可获得跨类样本对。该损失默认在 1000 个优化步内线性升温。旧绝对 margin 仅通过 `--dosc_safe_mode legacy_margin` 保留为消融，不再作为默认方案。

## 4. 严格 UDA 边界

- 训练只读取源域标签；
- 目标 CSV 即使含有标签列，数据构建器也不会把目标标签写入 UNSB 数据集；
- 可选诊断子空间、可选诊断对抗头和 CIDP 均只使用源域标签；
- 目标图只提供无标签风格参考和域标识；
- 源域验证集用于分类器与超参数选择，目标测试标签只用于最终一次评估。

## 5. 安装到官方 UNSB

当前 overlay 面向官方 `cyclomon/UNSB` 的 `main` API（核对版本：`d1f644f`）。它不会修改上游 `sb_model.py`，而是在 `models/` 和 `data/` 下新增独立类型。

```bash
python scripts/install_trsc_unsb_overlay.py \
  --unsb_root /root/autodl-tmp/UNSB
```

安装后新增：

- `models/dosc_modules.py`
- `models/dosc_sb_model.py`
- `models/trsc_sb_model.py`（新实验的规范入口）
- `data/dosc_unaligned_dataset.py`
- `data/trsc_unaligned_dataset.py`（新实验的规范入口）

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

## 7. 训练、校验并导出诊断教师

教师训练必须读取 `da_manifest.csv` 中显式且病例级互斥的 `src_train` 与 `src_valid`。默认混合 `src_path,raw,U1,U5`，并按源验证集的最差渲染 AUC 选模：

```bash
python scripts/train_render_robust_teacher.py \
  --manifest /path/to/da_manifest.csv \
  --train_split src_train \
  --val_split src_valid \
  --views src_path,raw,U1,U5 \
  --init_weights /path/to/raw_source_teacher.pt \
  --baseline_weights /path/to/raw_source_teacher.pt \
  --epochs 30 \
  --out /path/to/render_robust_teacher.pt
```

独立 BUSI test 只作一次验收，不参与训练、选模、校准拟合或阈值选择：

```bash
python scripts/v10_heldout.py \
  --old_weights /path/to/raw_source_teacher.pt \
  --new_weights /path/to/render_robust_teacher.pt \
  --manifest /path/to/u2b_rev_srctest_manifest.csv \
  --images_root /path/to/results_u2b_srctest/u2b_rev_SB/test_latest/images
```

校准检验在 `src_valid` 拟合，在独立 `src_test` 评价：

```bash
python scripts/v9_calibration.py \
  --weights /path/to/render_robust_teacher.pt \
  --fit_manifest /path/to/da_manifest.csv \
  --fit_split src_valid \
  --test_manifest /path/to/u2b_rev_srctest_manifest.csv \
  --test_split src_test \
  --test_images_root /path/to/results_u2b_srctest/u2b_rev_SB/test_latest/images
```

验收后导出 TorchScript：

```bash
python scripts/export_diagnostic_teacher.py \
  --weights /path/to/render_robust_teacher.pt \
  --backbone custom_resnet50_space \
  --out /root/autodl-tmp/busi_teacher.ts
```

导出的 TorchScript 模型直接接收 UNSB 的 `[-1,1]` 图像张量，并在内部调整到分类器输入尺寸。

## 8. 训练与生成

```bash
export UNSB_ROOT=/root/autodl-tmp/UNSB
export TRSC_DATA_ROOT=/root/autodl-tmp/UNSB/datasets/busi_to_breast_dosc
bash scripts/run_unsb_trsc_breast.sh
```

该脚本用于预训练单参考 TRSC 翻译器。后续多候选联合训练使用同一源图对应的不同 `testB` 目标参考；不能把 `fake_1…fake_5` 当成五种独立风格，因为它们是同一桥路径的不同翻译深度。
复现 CIDP 消融时才设置 `TRSC_TEACHER=/path/to/busi_teacher.ts LAMBDA_SAFE=0.5`。

## 9. 必做消融

| 版本 | 命令设置 | 回答的问题 |
|---|---|---|
| 原始 UNSB | `--model sb` | 随机 \(z\) 基线 |
| 目标参考条件化（core） | 默认关闭投影，`lambda_DOSC_diag=0`、`lambda_DOSC_safe=0` | 真实目标风格条件化的净作用 |
| TRSC + CIDP | 默认关闭投影，`lambda_DOSC_diag=0`、`lambda_DOSC_safe=0.5` | 复现 CIDP 无稳定收益的消融 |
| 旧泄漏控制 | `--dosc_enable_projection --lambda_DOSC_diag 0.1` | 复现投影 + GRL 的负结果 |
| TRSC + 旧 margin | `--dosc_safe_mode legacy_margin` | 验证类别偏斜的旧约束是否阻碍域迁移 |

多候选方法必须保证每个源样本的总训练权重一致，避免把“更多训练步数”误判成选择或条件化收益。

## 10. 训练日志判据

- `DOSC_*` 是为旧检查点兼容保留的日志前缀，不再表示诊断正交；
- `DOSC_domain_acc` 用于监测域信息；
- `DOSC_diag_acc` 仅在旧 GRL 消融中有解释意义；
- `DOSC_removed` 在默认配置中应为 0，仅在投影消融中非零；
- `DOSC_cidp_ready` 在两类队列就绪后应从 0 变为 1；
- `DOSC_affine_scale` 与 `DOSC_affine_bias` 记录已吸收的校准漂移；
- `DOSC_raw_margin_drop` 仅作旧测量诊断，不能解释为病灶损伤；
- `DOSC_margin_drop` 是仿射校准后的局部 margin 变化；
- `DOSC_safe_rank` 反映阈值移动无法修复的成对排序损失；
- `DOSC_safe_c0` 与 `DOSC_safe_c1` 不应再出现旧约束的数量级不对称；
- 最终判定仍以同一患者划分下的目标 AUC、配对 bootstrap CI 和三个随机种子为准。

## 11. 风格码交换因果检验

冻结探针只能证明风格码中的诊断信息“可解码”，不能证明生成器使用了这部分信息。交换检验对每个固定源病例使用 \(K\) 个无标签目标训练参考，并在不同参考之间复用完全相同的 UNSB 桥噪声：

\[
u_{ij}=T\!\left(G(x_i^s,z_j^t;\epsilon_i)\right).
\]

其中 \(x_i^s\) 与路径噪声 \(\epsilon_i\) 固定，仅改变目标参考条件 \(z_j^t\)。另设“固定参考 + 独立路径噪声”对照，避免把随机采样波动误判为参考风格效应。

```bash
python scripts/eval_trsc_style_swap.py \
  --unsb_root /root/autodl-tmp/UNSB \
  --checkpoints_dir /root/autodl-tmp/UNSB/checkpoints \
  --experiment_name busi_to_breast_dosc \
  --epoch latest \
  --teacher /root/autodl-tmp/busi_teacher.ts \
  --probe_manifest /path/to/da_manifest.csv \
  --probe_splits src_train \
  --probe_path_col src_path \
  --source_manifest /path/to/u2b_rev_srctest_manifest.csv \
  --source_split src_test \
  --source_images_root /path/to/results_u2b_srctest/u2b_rev_SB/test_latest/images \
  --source_images_subdir real \
  --target_manifest /path/to/breast_target_train.csv \
  --target_path_col image_path \
  --references 12 \
  --steps 1,5 \
  --out_dir /path/to/style_swap_audit
```

新 TRSC 检查点默认不启用投影。复现历史 DOSC 检查点的投影后结果时必须显式加入 `--dosc_enable_projection`，避免把同一检查点在不同前向路径下的结果混在一起。

输出包括：

- `style_swap_per_pair.csv`：病例—参考—深度级诊断分数和风格一致性；
- `fixed_reference_noise_per_pair.csv`：固定参考的路径噪声对照；
- `selected_target_references.csv`：参考选择与源标签探针给出的 malignancy-like 分数；
- `style_swap_summary.json`：源测试诊断探针 AUC、独立域探针 AUC、病例内方差、翻转率、病例级斜率 bootstrap CI 和参考置换检验。

脚本只用源训练标签拟合诊断探针；目标清单即使包含标签列也不会读取。真实目标参考从目标训练集抽取，目标验证/测试图不能进入该实验。

判定顺序为：

1. 当前结果满足病例级参考置换不显著，且参考引起的诊断波动远低于路径噪声，因此只能说明“潜变量可解码”，不能认定存在有害因果泄漏；
2. 若 malignancy-like 参考分数稳定推动同一源病例的输出诊断分数，且效应超过路径噪声，则再实现风格交换一致性损失；
3. 若输出诊断稳定但不同参考的输出风格也不变，则生成器可能忽略条件，不能把安全性归功于泄漏控制；
4. 投影与 GRL 只有在独立探针、因果安全和下游效用三个闸门同时改善时才保留为主方法，否则降为消融或删除。

此前 130 例 BUSI 测试集已经用于教师验收并影响 CIDP 设计，因此后续交换结果属于机制开发证据，不再是完全未触碰的最终确认集。正式论文若声称因果安全，仍需新的独立源安全集或外部数据验证。

## 12. 定位下游 AUC 退化

因果交换审计排除了“参考诊断倾向直接推动输出诊断分数”这一解释，但没有证明翻译伪影、频率损伤或训练分布重加权无害。下一步只比较三个最小翻译器臂：

1. `core`：目标参考条件化；
2. `cidp`：`core + CIDP`；
3. `legacy_controls`：`cidp + 投影 + GRL`。

先生成单种子，再在方向稳定后确认三随机种子：

```bash
SEEDS="7" bash scripts/run_trsc_factorial.sh
```

将每个翻译器随机种子的三组结果分别构造成病例完全对齐的分类器清单。构建器会验证同一随机种子内每个臂的 `real/` 像素完全一致：

```bash
python scripts/build_trsc_downstream_manifest.py \
  --source_manifest "$TRSC_DATA_ROOT/testA_manifest.csv" \
  --variant core=/path/to/core_s7/test_latest/images \
  --variant cidp=/path/to/cidp_s7/test_latest/images \
  --variant legacy_controls=/path/to/legacy_s7/test_latest/images \
  --out /path/to/trsc_downstream_s7.csv
```

对 7、16、42 分别生成 `trsc_downstream_s{seed}.csv`。匹配的两视图分类器使用对应种子的翻译器结果和同种子分类器，只使用源训练/验证标签，并在训练阶段完全不读取目标标签：

```bash
MANIFEST_TEMPLATE='/path/to/trsc_downstream_s{seed}.csv' \
TARGET_CSV=/path/to/breast_test_inference.csv \
SRC_TEST_CSV=/path/to/busi_test.csv \
SEEDS="7 16 42" \
bash scripts/run_trsc_downstream_factorial.sh
```

完整矩阵结束后，才由独立报告脚本一次性连接目标标签并计算病例级、种子级配对 bootstrap：

```bash
python scripts/eval_trsc_downstream.py \
  --runs_root /path/to/runs \
  --sealed_dir /path/to/sealed \
  --target_labels /path/to/breast_test_labels.csv \
  --conditions raw,core_U1,cidp_U1,legacy_controls_U1,core_U5,cidp_U5,legacy_controls_U5 \
  --contrast cidp_U1:core_U1 \
  --contrast legacy_controls_U1:cidp_U1 \
  --contrast cidp_U5:core_U5 \
  --contrast legacy_controls_U5:cidp_U5 \
  --out_dir /path/to/downstream_report
```

解释规则固定为：

- `core < raw`：伤害来自目标参考条件化或翻译伪影；
- `cidp < core`：CIDP 约束本身阻碍有效迁移；
- `legacy_controls < cidp`：投影/GRL 造成额外损失，应从主方法删除；
- 教师安全指标稳定但目标 AUC 仍下降：教师只覆盖一种诊断表征，不能把它当作下游充分条件。

三随机种子、201 例目标开发集的实测均值为：

| 条件 | AUC 均值 | 相对 raw |
|---|---:|---:|
| raw | 0.7842 | — |
| core U1 | 0.7858 | +0.0015 |
| CIDP U1 | 0.7696 | −0.0147 |
| legacy U1 | 0.7848 | +0.0005 |
| core U5 | 0.7287 | −0.0555 |
| CIDP U5 | 0.7343 | −0.0499 |
| legacy U5 | 0.7479 | −0.0364 |

三个 U1 臂相对 raw 的区间均跨零，因此只能说“未检测到稳定影响”，不能说已经证明等效或无害。三个臂的 U5 均低于 U1，说明真正稳定的风险来自翻译深度。新实验只使用 U1，不再投入 U5。

## 13. K 个不筛选候选 + 双 warm start + 端到端任务梯度

当前待验证的问题不是继续压制风格码，而是“同一病例的多个真实目标参考是否能提供有用的数据多样性”。对源病例 \(x_i^s,y_i^s\) 无放回抽取 \(K\) 个目标训练参考 \(r_{ik}^t\)，生成：

\[
u_{ik}=G\!\left(x_i^s,t=0,E_s(r_{ik}^t)\right),\qquad k=1,\ldots,K.
\]

所有 \(u_{ik}\) 都进入分类器，不按教师分数、风格距离或目标标签筛选。默认任务损失给原图组和候选组各一半总权重：

\[
\mathcal L_{\mathrm{task}}
=\frac12\,\mathrm{CE}(C(x_i^s),y_i^s)
+\frac{1}{2K}\sum_{k=1}^{K}\mathrm{CE}(C(u_{ik}),y_i^s).
\]

因此从 \(K=1\) 增至 \(K=3\) 不会放大一个源病例的总权重。分类器始终收到完整梯度；候选分支传给 \(G\) 与 \(E_s\) 的梯度乘 `lambda_TRSC_task`。默认值 1.0 即完整端到端解冻，设为 0.0 则只切断分类 CE 的翻译器梯度，同时保留 TRSC/UNSB 原生训练损失。

规范入口强制使用两个 warm start：

- `TRSC_INIT_DIR`：已有 TRSC core 的 `G/F/D/E/S` 检查点，之后全部保持可训练；
- `SOURCE_CLASSIFIER_CKPT`：源域 `custom_resnet50_space` 最优检查点，联合分类器使用完全相同的网络结构并严格加载。

先更新官方 UNSB overlay：

```bash
python scripts/install_trsc_unsb_overlay.py \
  --unsb_root /root/autodl-tmp/UNSB \
  --force
```

单次规范实验：

```bash
export UNSB_ROOT=/root/autodl-tmp/UNSB
export TRSC_DATA_ROOT=/root/autodl-tmp/UNSB/datasets/busi_to_breast_dosc
export TRSC_INIT_DIR=/path/to/pretrained_trsc_core
export SOURCE_CLASSIFIER_CKPT=/path/to/source_classifier/best_checkpoint.pt

NUM_REFERENCES=3 \
LAMBDA_TASK=1.0 \
SEED=7 \
bash scripts/run_unsb_trsc_joint_breast.sh
```

最小归因矩阵：

```bash
SEEDS="7 16 42" bash scripts/run_trsc_joint_matrix.sh
```

| 实验臂 | \(K\) | 分类 CE → 翻译器 | 直接回答 |
|---|---:|---:|---|
| `k1_joint` | 1 | 是 | 单参考端到端基线 |
| `k3_no_task_gradient` | 3 | 否 | 三候选本身对分类器是否有用 |
| `k3_joint` | 3 | 是 | 多样性与任务驱动翻译能否合并产生收益 |

所有实验臂使用同一个源分类器和同一个 TRSC 检查点起步。主要比较为 `k3_joint − k1_joint`（多参考多样性）和 `k3_joint − k3_no_task_gradient`（任务梯度）。raw 仍使用原 source-only 检查点。目标训练参考只读取图像路径，不读取标签；目标评价 CSV 先由 `eval_trsc_joint_classifier.py` 生成无标签病例概率，再由独立评估器连接封存标签。
