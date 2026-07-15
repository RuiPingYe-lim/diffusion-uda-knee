# 交接文档 / Handoff for review

> 目的:把「已实现的代码」「跑出的结果」「未决问题」一次说清,便于第三方(GPT)复核代码与实验逻辑。
> 仓库:https://github.com/RuiPingYe-lim/diffusion-uda-knee  (main 分支,截至 commit `926f181`,已全部推送)

---

## 0. 研究问题

诊断分类的**无监督域适应(UDA)**:用「源域标注数据」训练的分类器,在「无标注目标域」上工作。
思路:用**生成式翻译**把目标域图像翻译成源域风格 → 让源域分类器直接可用。

两套数据(均为 mean-projection / 单图-单病例,二分类):
- **膝关节 MRI**:MRNet(源) ↔ KneeMRI(目标)。方向 m2k = MRNet→KneeMRI。
- **乳腺超声**:BUSI(源) ↔ BrEaST(目标)。

核心疑问:**这个「翻译」到底能不能提升目标域分类 AUC?**

---

## 1. 参照基线(paper-1,同数据)

| 方法 | 正向 AUC | 反向 AUC |
|---|---|---|
| Source Only(仅源训练,直接迁移) | 0.740 | 0.807 |
| NMF(paper-1 特征对齐方法) | 0.805 | 0.854 |
| Fully Supervised(上界) | 0.888 | 0.931 |

膝关节冻结源分类器在 kneemri_test 上 direct AUC = **0.7281**;乳腺冻结源(BUSI)分类器在 BrEaST 诊断集上 direct AUC = **0.791**。这两个 direct 是后面所有实验的对照基线。

---

## 2. 已实现的代码(按角色)

路径都在 `src/bbdm_strict/`(除非注明)。

### 2.1 生成器 / 翻译模型
- `train_strict_bbdm.py` — BBDM(Brownian Bridge)潜空间翻译训练。**含建议④开关 `--lambda_stat_prior`**(源域全局强度统计作条件先验的损失约束)。
- `sample_strict_bbdm.py` — BBDM 反向采样(生成翻译图);已验证「源参考图只用于日志、不进反向计算」,故 label_random 配对**不泄漏标签**。
- `ae_frontend.py` / `models_strict_bbdm.py` / `bridge_scheduler.py` — KL-VAE 前端、桥 UNet、桥调度器。
- **UNSB**(外部代码,在服务器 `/root/autodl-tmp/UNSB`,基于 CUT 的 Neural Schrödinger Bridge):训练脚本见 `scripts/build_breast_unsb_dataset.py`(建数据集)+ UNSB 自带 `train.py`/`test.py`。k2m(膝) 与 b2u(乳腺,BrEaST→BUSI) 均已训 30 epoch。

### 2.2 分类器
- `fusion_classifier.py` — **交叉注意力融合分类器**,是导师 ③④⑤ 的落地处:
  - `--fusion_mode orig_kv`(默认,**建议③**):原图=Key/Value、翻译图=Query + 原图锚定残差(原图主导,对坏翻译鲁棒)。另有旧版 `orig_query`。
  - `--stat_prior`(**建议④**,特征向量形式):FiLM 分支,把每样本 `[before_mean,before_std,fake_mean,fake_std,src_mean,src_std]`(强度统计)→ (γ,β) 调制融合表征;FiLM 末层零初始化 → 开关关时**精确等于基线**(已验证 |Δ|=0)。
  - `--supcon_weight`(**建议⑤**):CE + λ·监督对比损失(Khosla),含投影头;已单测(同类分离→0、随机→高、无正样本→0)。
  - `--self_check`:随机前向验证形状。
- `train_path_invariant_classifier.py` — C2「路径不变」分类器(源图 + 朝目标风格的路径增强 + 一致性/KD);严格 UDA 协议(选模仅用源验证,目标 test 只最终评估一次)。
- `train_eval_meanproj.py` / `eval_existing_classifier_on_csv.py` — mean-proj 源分类器训练/加载(`build_model`,backbone `custom_resnet50_space`)。

### 2.3 诊断探针 / 评估(冻结分类器 + 配对 bootstrap CI)
- `eval_meanproj_oracle.py` — mean-proj 上的固定一二阶矩干预探针(第一轮判停闸门;含 direct 复现自检 + VOID 阻断)。
- `eval_moment_intervention.py` — b′ 探针:direct / moment_global / moment_bank_all,三组共享索引配对 CI;`--self_test` 专验 moment-bank 与 bootstrap 分支;记录 checkpoint/manifest 的 SHA256。
- `eval_bbdm_translation.py` — BBDM 翻译三臂分解(before / ae_recon / translated),隔离 VAE 瓶颈与「桥」的贡献。
- `eval_unsb_translation.py` — UNSB 逐桥步评估(real / fake_1..N)+ direct 自检锚点。

### 2.4 数据 / 流水线
- `precompute_meanproj.py` — 体数据→mean 投影→p1/99 强度归一→PNG(paper-1 对齐)。
- `source_style_stats.py` — 源域全局 mean/std(缓存),矩匹配共用。
- `path_augmentation.py` — C2 的目标风格 bank + 路径视图合成(端点无噪、不读目标标签)。
- `scripts/`:`build_fusion_csvs_breast.py` / `build_fusion_csvs_knee.py`(把 UNSB 翻译输出映射成 `(before, fake_1..5, label, case_id)` 配对 CSV)、`build_breast_unsb_dataset.py`、`breast_caseid_audit.py`(病例台账审计:1 图/病例、三 split 无重叠)、`build_breast_diag_cid.py`、`run_multiseed_2x2.sh`(多种子消融驱动)、以及各 `run_*.sh`。
- `tests/test_path_augmentation.py` — 端点精确/无噪、可复现、不读目标标签(3 测试)。

---

## 3. 导师 5 条建议 —— 实现状态

| 编号 | 内容 | 状态 | 落地 |
|---|---|---|---|
| ① | 换 Flow Matching 生成器 | ❌ **未做** | 用 UNSB 先替代验证方向;Flow Matching 本身未实现 |
| ② | 采样步数在**最优模型**上搜 | ✅ 做了 | 在 UNSB(最优)上逐桥步评估;见 §4.4 |
| ③ | 注意力:原图 K/V、翻译图 Query | ✅ 做了 | `fusion_classifier.py` orig_kv(默认) |
| ④ | 源域统计作条件先验 | ✅ 做了(两处) | BBDM 损失 `--lambda_stat_prior`;融合 FiLM `--stat_prior` |
| ⑤ | 保留监督对比损失 | ✅ 做了 | 融合 `--supcon_weight`(需小权重,见 §4.5) |

---

## 4. 实验结果(全部,单种子除非注明;差异带 ≈ ±0.05 bootstrap 噪声)

### 4.1 膝关节 mean-proj 固定矩干预探针 → **STOP**
direct 0.7281,moment_global 0.715,Δ=**−0.013** CI[−0.046,+0.020](跨 0)。诊断:逐病例 p1/99 归一已大幅抹平外观差 → 该管线上矩干预无外观杠杆。

### 4.2 乳腺 b′ 矩探针(冻结 BUSI 分类器,诊断 n=201;自检 src_val=0.9913 精确复现)
direct 0.791 / moment_global 0.771 / **moment_bank_all(全452源) 0.799**。
- bank−direct = +0.008 CI[−0.034,+0.050](跨 0);
- bank−global = +0.028 CI[+0.005,+0.052](**不跨 0**,逐病例 bank 确实优于全局 pooled);
- global−direct = −0.021(略负)。
判定:很弱(0<Δ<0.01)。外观差本身真实(≈0.88 跨病例 SD)。

### 4.3 乳腺 C2 路径增强(真训练,单种子,诊断目标 AUC)
none(源only) **0.808** / endpoint 0.804 / linear 0.802 —— 两个路径模式**均 ≤ none**,未过 +0.01,STOP。

### 4.4 生成式翻译「翻译-only」(冻结分类器,翻译图单独打分)
- **BBDM**(乳腺,三臂):before(128) 0.791 / ae_recon 0.774 / **translated 0.4946(≈随机)**。translated−ae_recon = −0.279 → **是「桥」摧毁了判别内容,不是 VAE**。
- **UNSB 膝**:direct 0.7281 / real 0.7261 / fake_1..5 = 0.687/0.678/0.679/0.673/**0.665**。
- **UNSB 乳腺**:direct 0.791 / real 0.783 / fake_1..5 = 0.730/0.705/0.700/0.699/**0.692**。
→ 两数据集一致:**翻译保住了信号(远好于 BBDM),但单调侵蚀、每步都低于 direct**。「图更清晰 ≠ 判别内容更多」。
→ **②步数结论**:翻译-only 最优 = 少步(fake_1);见 §4.6 融合里相反。

### 4.5 乳腺融合 2×2 消融(orig_kv + ④/⑤,单种子,目标域;direct=0.791)
初版(⑤ 用 0.5):③ 0.778 / ③+⑤ 0.748 / ③+④ **0.790** / ③+④+⑤ 0.784。
⑤ 权重扫描:0.0→0.778 / **0.1→0.781** / 0.2→0.744 / 0.5→0.748 → **⑤ 之前「有害」是权重 0.5 太高;0.1 时无害**。
公平版(⑤ 用 0.1):③ 0.778 / ③+⑤ 0.781 / ③+④ 0.790 / ③+④+⑤ **0.751**。
→ 唯一提示性正贡献是 ④(把融合拉回 ≈direct);但**最好也只追平 direct**,单种子差异都在噪声内。

### 4.6 ②采样步数(在最优模型 UNSB 上,融合 ③+④)
K=1(fake_1) 0.738 / K=3 0.772 / K=5 0.790 → **融合里多视图更好**(注意力对原图的集成),与翻译-only 相反;两者天花板都是 direct。

### 4.7 膝关节融合 2×2(单种子)—— ⚠️ 不稳定,不可解读
base 0.835 / supcon 0.677 / stat 0.700 / stat_supcon 0.762。但**四格源验证分几乎相同(~0.85),目标分却在 0.68~0.84 乱跳** → 「按源验证选模型」在域偏移下选不出目标好模型,单种子差异是**选模噪声**,非 ④⑤ 真效果。base 的 0.835 是运气好的存档,**不能当作「融合超过 direct」**。

### 4.8 多种子确认(2 数据集 × 4 格 × 3 种子)—— ⏳ **未完成**
`scripts/run_multiseed_2x2.sh` 已写并启动,但服务器 SSH 中途掉线(实例疑似重启换端口),**跑到一半中断**。`results.csv` 里已完成的种子在持久盘保住,需重连后断点续跑。这是把融合 ④⑤ 结论做实的**唯一待补步骤**。

---

## 5. 哪些结论可信 / 哪些是噪声

**可信(冻结分类器,不涉及选模)**:
- BBDM 翻译摧毁判别信号(→0.49);UNSB 保信号但单调侵蚀、跨两数据集一致、都低于 direct。
- b′:逐病例 moment bank 稳定优于全局 pooled(CI 不跨 0)。

**不可信 / 需多种子**:
- 融合 2×2 的**格子级**差异(④/⑤ 谁有用)——单种子被「源验证选模」噪声淹没,膝关节尤其严重(0.16 跨度)。

**一句话总体结论(截至目前)**:不管 BBDM/UNSB、翻译-only/融合、③④⑤ 如何组合、步数如何调,**翻译-融合栈的天花板都是 direct**。瓶颈指向「翻译这一操作不产生增量判别信息」,而非生成器清晰度或融合设计。

---

## 6. 想请复核的点(欢迎 GPT 重点看)

1. **融合评估协议**:UDA 下「按源验证选 checkpoint」导致目标 AUC 高方差(§4.7)。多种子平均是否足够?是否该换更稳的选模/评估(如对目标伪标签、或多 checkpoint 集成)?
2. **④ 的 FiLM 设计**(`fusion_classifier._apply_stat_prior`):把源域统计当条件先验是否合理?per-sample 6 维统计向量是否是导师原意的「特征向量」形式?零初始化消融是否干净?
3. **⑤ 在跨域下的用法**:小权重(0.1)无害但也不涨;是否该改「两段式(先对比预训练→再分类)」?
4. **UNSB srcapply 的 OOD**:训练融合时把源域(BUSI/MRNet)图喂给「目标→源」翻译器属轻微 OOD;orig_kv 的原图主导是否足以消化?是否该训第二个方向的翻译器?
5. **① 是否值得做**:现有证据(天花板=direct)下,换 Flow Matching 是否可能突破?还是问题在「问题设定/数据」层面?
6. **代码正确性**:`fusion_classifier.py`(注意力方向、FiLM、SupCon)、`eval_*translation.py`(冻结分类器管线、配对 bootstrap)、`train_strict_bbdm.py` 的 `--lambda_stat_prior`。

---

## 7. 复现关键命令(服务器路径)

```bash
# 冻结分类器 UNSB 翻译评估(乳腺)
python src/bbdm_strict/eval_unsb_translation.py \
  --classifier <BUSI源分类器>.pt --test_csv breast_diag_cid.csv \
  --unsb_images_dir <UNSB>/results/b2u_SB/test_latest/images \
  --arms real,fake_1,fake_2,fake_3,fake_4,fake_5 --ref_arm real --expect_direct_auc 0.7909

# 融合训练+评估(orig_kv + ④ + ⑤)
python src/bbdm_strict/fusion_classifier.py --mode train \
  --train_csv fusion_train.csv --val_csv fusion_val.csv \
  --before_col before_png --other_cols fake_1,fake_2,fake_3,fake_4,fake_5 --label_col label \
  --fusion_mode orig_kv --stat_prior --supcon_weight 0.1 --out_dir run --epochs 30 --seed 42
python src/bbdm_strict/fusion_classifier.py --mode eval --weights run/best.pt --test_csv fusion_eval.csv ...

# 多种子 2×2(断点续跑)
bash scripts/run_multiseed_2x2.sh
```
