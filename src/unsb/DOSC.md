# DOSC：目标风格条件化与诊断安全审计

DOSC 最初按 Diagnostic-Orthogonal Style Conditioning 设计：把 UNSB 的随机风格噪声 \(z\) 改为由无标签目标参考图编码得到的条件码，并尝试通过线性投影与梯度反转抑制其中可由源域标签识别的诊断信息。该模块用于 **BUSI（有标签源域）→ BrEaST（无标签目标域）** 的源图目标风格增强。

## 0. 当前证据边界

冻结线性探针的最新结果不支持“诊断正交已经实现”：

| 训练设置 | 编码阶段 | 诊断 AUC | 域 AUC |
|---|---|---:|---:|
| `safe0` | 投影前 | 0.6351 | 0.9188 |
| `safe0` | 投影后 | 0.6502 | 0.9169 |
| `safe05` | 投影前 | 0.6611 | 0.9260 |
| `safe05` | 投影后 | 0.6854 | 0.9290 |

随机 AUC 基准为 0.5。域信息得到保留，但诊断信息仍可线性解码；投影后 AUC 没有下降，单方向投影与现有 GRL 因而不能写成有效的泄漏控制。当前严格结论是：

- **目标参考风格条件化有效**：风格码保留强域信息，并显著提高类条件域覆盖；
- **潜变量诊断正交性不成立**：投影与 GRL 暂未优于无保护描述子；
- **有害因果泄漏尚未确定**：风格码可解码不等于生成器实际利用这部分信息改变诊断；
- **CIDP 仍解决输出安全问题**：它约束校准后 margin 与局部排序，但不证明潜变量无泄漏。

因此，代码保留投影与 GRL 作为可复现实验臂，不再把它们的效果写成既成事实。`scripts/eval_dosc_style_swap.py` 使用共同随机数进行目标参考交换，专门检验潜变量泄漏是否被生成器因果利用。

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

## 2. 候选泄漏控制为何包含投影与梯度反转

乳腺超声中的采集纹理和病灶纹理高度耦合，仅靠一个类别对抗器容易出现“对抗器暂时失效，但风格码仍泄漏诊断信息”的情况。DOSC 同时使用：

1. **显式线性投影**：直接删除 EMA 类中心张成的诊断方向；
2. **诊断对抗头**：通过 GRL 抑制投影后残留的非线性诊断信息；
3. **域分类头**：要求投影后的编码仍能区分源域和目标域，避免把域信息一并剥除；
4. **参考实例对比**：目标图与其水平翻转视图为正对，历史目标参考码为负样本，防止所有目标风格坍缩为一个常量。

前两项是待验证的候选机制。在线对抗头准确率和投影删除能量都不能替代训练后冻结探针；只有当独立探针 AUC 相对无机制基线稳定下降、域 AUC 不降且下游效用保留时，才能称其为有效泄漏控制。

## 3. 完整目标函数

\[
\mathcal L =
\mathcal L_{\mathrm{UNSB}}
+\lambda_{\mathrm{diag}}\mathcal L_{\mathrm{diag\text{-}adv}}
+\lambda_{\mathrm{dom}}\mathcal L_{\mathrm{domain}}
+\lambda_{\mathrm{ins}}\mathcal L_{\mathrm{instance}}
+\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{style\text{-}rec}}
+\lambda_{\mathrm{safe}}\mathcal L_{\mathrm{CIDP}}.
\]

### 3.1 为什么废弃绝对 margin

令冻结教师的二分类分数为

\[
d(x)=z_1(x)-z_0(x),\qquad m_y(x)=(2y-1)d(x).
\]

旧安全项直接计算

\[
\left[m_y(x^s)-m_y(G(x^s))-\delta\right]_+.
\]

独立 BUSI 测试表明，U1 同时存在校准漂移和局部排序损失。旧教师在 U1 上对良性和恶性的平均安全罚分别为 0.101 和 6.399，形成 63 倍类别不对称；更换渲染鲁棒教师后仍为 0.509 和 1.276。原因不是 256 像素渲染，而是绝对 margin 对诊断分数的整体平移和温度变化敏感。因此默认安全项改为 CIDP（Calibration-Invariant Diagnostic Preservation，校准不变诊断保持）。

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
- 诊断子空间、诊断对抗头和 CIDP 均只使用源域标签；
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
| 完整 DOSC + CIDP | 默认配置 + 渲染鲁棒教师 | 是否兼顾目标风格、个体差异与诊断安全 |
| DOSC + 旧 margin | `--dosc_safe_mode legacy_margin` | 验证类别偏斜的旧约束是否阻碍域迁移 |
| DOSC，不用安全项 | `--lambda_DOSC_safe 0` | 确认 CIDP 的净贡献 |

多候选方法必须保证每个源样本的总训练权重一致，避免把“更多训练步数”误判成选择或条件化收益。

## 10. 训练日志判据

- `DOSC_domain_acc` 应明显高于随机，证明域信息没有被投影剥光；
- `DOSC_diag_acc` 只能作为在线对抗状态，最终必须另训冻结探针验证风格码诊断 AUC；
- `DOSC_removed` 应非零但不应长期接近 1，否则删除范围过大；
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
python scripts/eval_dosc_style_swap.py \
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

输出包括：

- `style_swap_per_pair.csv`：病例—参考—深度级诊断分数和风格一致性；
- `fixed_reference_noise_per_pair.csv`：固定参考的路径噪声对照；
- `selected_target_references.csv`：参考选择与源标签探针给出的 malignancy-like 分数；
- `style_swap_summary.json`：源测试诊断探针 AUC、独立域探针 AUC、病例内方差、翻转率、病例级斜率 bootstrap CI 和参考置换检验。

脚本只用源训练标签拟合诊断探针；目标清单即使包含标签列也不会读取。真实目标参考从目标训练集抽取，目标验证/测试图不能进入该实验。

判定顺序为：

1. 若病例级斜率置信区间跨 0、参考置换检验不显著，且参考引起的诊断波动不高于路径噪声对照，则只能说明“潜变量可解码”，不能认定存在有害因果泄漏；
2. 若 malignancy-like 参考分数稳定推动同一源病例的输出诊断分数，且效应超过路径噪声，则再实现风格交换一致性损失；
3. 若输出诊断稳定但不同参考的输出风格也不变，则生成器可能忽略条件，不能把安全性归功于泄漏控制；
4. 投影与 GRL 只有在独立探针、因果安全和下游效用三个闸门同时改善时才保留为主方法，否则降为消融或删除。

此前 130 例 BUSI 测试集已经用于教师验收并影响 CIDP 设计，因此后续交换结果属于机制开发证据，不再是完全未触碰的最终确认集。正式论文若声称因果安全，仍需新的独立源安全集或外部数据验证。
