# DA-BRF：诊断感知的桥残差修复

DA-BRF（Diagnosis-Aware Bridge Residual Repair）接在当前 TRSC 多参考联合训练链中。它不重新生成病灶，也不恢复已经停止的 U3/U5 深层翻译；它只处理每张 U1 候选相对源原图的残差，检验“能否保留目标风格变化，同时减少会伤害源诊断标签的改动”。

## 1. 当前证据边界

已完成的三随机种子结果表明：

- 三个 U1 臂相对 raw 均未检测到稳定增益；
- 三个 U5 臂均低于对应 U1，深层翻译停止；
- CIDP 直接约束生成器没有稳定下游收益；
- 投影和 GRL 不再作为默认模块。

因此，DA-BRF 是待验证的新实验，不是已证实有效的方法。它与旧 CIDP 的关键区别是：CIDP 直接处罚生成器输出，而 DA-BRF 在 U1 已经生成后学习一个局部残差修复器，并让安全约束只更新修复器，避免生成器通过改变输出分布规避冻结教师。

## 2. 修复对象

对源图 $x_i^s$ 和第 $k$ 张 U1 候选 $u_{ik}$，定义原始桥残差：

\[
r_{ik}=u_{ik}-x_i^s.
\]

学习空间与通道门控 $g_\omega(x_i^s,u_{ik})\in[0,1]$，得到：

\[
\widehat r_{ik}=g_\omega(x_i^s,u_{ik})\odot r_{ik},
\qquad
\widetilde u_{ik}=x_i^s+\widehat r_{ik}.
\]

门控只缩放 U1 已有残差，不能直接把目标参考图的像素或病灶写入源病例。最终再把 $\widehat r_{ik}$ 投影到以原残差范数定义的 L2 球内：

\[
\|\widehat r_{ik}\|_2
\le R\|r_{ik}\|_2.
\]

默认 $R=1$，所以修复后的改动不会比原始 U1 更大。

## 3. 三项约束

### 3.1 校准解耦的诊断约束

冻结的渲染鲁棒源教师分别输出源图和修复图的二分类分数。类别平衡 FIFO 队列使用停止梯度的分数拟合正仿射映射：

\[
(a^\star,b^\star)
=\arg\min_{a>0,b}
\sum_i w_{y_i}
\left(a\,\operatorname{sg}(d_i^{\mathrm{repair}})+b
-\operatorname{sg}(d_i^s)\right)^2.
\]

校准吸收全局 bias 和温度漂移；随后用真实源标签计算校准 margin 非劣损失和跨类排序损失。目标标签不参与任何步骤。

### 3.2 目标风格进度约束

固定的多尺度统计描述子提取每个尺度的通道均值、标准差及水平/垂直梯度能量。以到对应无标签 BrEaST 参考图的距离衡量风格接近程度：

\[
\Delta_{ik}^{\mathrm{U1}}
=D(x_i^s,r_{ik}^t)-D(u_{ik},r_{ik}^t),
\]

\[
\Delta_{ik}^{\mathrm{repair}}
=D(x_i^s,r_{ik}^t)-D(\widetilde u_{ik},r_{ik}^t).
\]

只对原 U1 确实缩短参考距离的候选施加：

\[
\mathcal L_{\mathrm{progress}}
=\left[
\eta\Delta_{ik}^{\mathrm{U1}}
-\Delta_{ik}^{\mathrm{repair}}
\right]_+,
\qquad \eta=0.8.
\]

该指标不是目标域 AUC 的替代物，只用于防止修复器把所有候选退化回源原图。

### 3.3 残差半径约束

实现同时包含硬 L2 投影和半径超限日志。硬投影保证数值上界，软损失用于发现实现错误或浮点越界：

\[
\mathcal L_{\mathrm{radius}}
=\left[
\frac{\|\widehat r_{ik}\|_2}{\|r_{ik}\|_2}-R
\right]_+.
\]

完整约束为：

\[
\mathcal L_{\mathrm{DA\text{-}BRF}}
=\lambda_d\mathcal L_{\mathrm{diag}}
+\lambda_p\mathcal L_{\mathrm{progress}}
+\lambda_r\mathcal L_{\mathrm{radius}}.
\]

## 4. 梯度边界

训练中对同一批候选执行两条数值相同、反向边界不同的修复路径：

1. **任务路径**：分类 CE 完整更新分类器与 DA-BRF，并按 `lambda_TRSC_task` 将梯度传给 TRSC 生成器和风格编码器；
2. **约束路径**：输入 U1 在进入 DA-BRF 前 `detach()`，诊断、风格进度和半径损失只更新 DA-BRF，不更新生成器。

这一区分防止生成器和冻结教师/校准器形成共同降低安全损失的捷径。分类器仍使用原图与全部 K 张修复候选，`equal_groups` 保持每个源病例总权重不随 K 改变。

## 5. 代码入口

| 文件 | 作用 |
|---|---|
| `dabrf_modules.py` | 残差门控、硬半径投影、固定多尺度风格进度 |
| `trsc_dabrf_joint_sb_model.py` | TRSC + 分类器 + DA-BRF 联合优化及冻结教师约束 |
| `run_unsb_trsc_dabrf_breast.sh` | 单次 DA-BRF 训练入口 |
| `run_trsc_dabrf_matrix.sh` | 身份、简单缩放、裁剪、任务门控与完整 DA-BRF 对照 |

先安装 overlay：

```bash
python scripts/install_trsc_unsb_overlay.py \
  --unsb_root /path/to/UNSB \
  --force
```

准备与现有 K3 联合实验相同的双 warm start，并提供冻结教师：

```bash
export UNSB_ROOT=/path/to/UNSB
export TRSC_DATA_ROOT=/path/to/busi_to_breast_dosc
export TRSC_INIT_DIR=/path/to/pretrained_trsc_core
export SOURCE_CLASSIFIER_CKPT=/path/to/source_classifier/best_checkpoint.pt
export DABRF_TEACHER=/path/to/render_robust_teacher.ts

SEEDS="7" bash scripts/run_trsc_dabrf_matrix.sh
```

种子 7 的训练、日志和推理协议正常，且 `learned_full` 相对 `identity` 有明确正方向后，再运行：

```bash
SEEDS="7 16 42" bash scripts/run_trsc_dabrf_matrix.sh
```

## 6. 归因矩阵

| 实验臂 | 作用 |
|---|---|
| `identity` | 通过同一新模型复现未修复 K3 joint 基线 |
| `fixed_scale08` | 检验统一把残差缩到 0.8 是否足够 |
| `norm_clip08` | 检验仅限制每例残差范数是否足够 |
| `learned_task_only` | 检验空间门控本身及源 CE 是否足够 |
| `learned_full` | 加入校准诊断、目标风格进度和半径约束 |

只有 `learned_full` 同时超过 `identity`、`fixed_scale08` 和 `norm_clip08`，才能支持“诊断感知修复”而不是“减弱翻译”这一解释。

## 7. 停止标准

DA-BRF 先跑一个预注册配置，不根据目标标签回调阈值。建议继续条件为：

- 相对 `identity` 的目标 AUC 至少约 +0.01；
- 三个种子方向基本一致；
- 优于固定缩放和范数裁剪；
- 诊断排序风险下降，同时目标风格进度保留；
- 目标标签仍只在完整矩阵结束后的独立评价阶段解封。

若完整 DA-BRF 只追平简单缩放，或增益仍在约 ±0.005 内，应删除该模块，不把额外参数量包装成创新。
