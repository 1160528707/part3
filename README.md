# 模型名v2：Coarsened Latent State Learning v2

> 中文定位：**粗化观测下的潜在临床状态学习框架**  
> 英文暂名：**CLSL-v2: Coarsened Latent State Learning for Sparse and Partially Observed EHR**

本代码工程是 CA-MFAM v1 的方法论升级版本。v1 的重点是“缺失特征自适应模块”；v2 把问题进一步提升为：

> 在特征视图、标签集合、时间轨迹都不完整的情况下，学习一个可迁移、可校准、可边缘推断的潜在状态预测模型。

它不是简单的 `mask + attention + teacher-student + GNN`，而是把真实医院低信息场景抽象成一个更一般的机器学习问题：

```text
完整状态 X*, Y*, T* 无法直接观测，训练/部署时只能看到它们的不同粗化投影：

X^v = Cx^v(X*)          # 特征视图粗化，例如 full / initial_lab / hospital_view
Y^O = Cy^O(Y*)          # 标签部分观测，例如 6 个合并症只观测 4 个
T^τ = Ct^τ(T*)          # 时间轨迹粗化，例如首次入院只有单次快照
```

目标是学习：

```text
p(Y_future* | X^v, Y_current^O, Cx^v, Cy^O, Ct^τ)
```

---

## 1. v2 已实现的核心模块

### 1.1 Evidence-Lattice Encoder：证据格编码器

将不同输入视图建模为证据格：

```text
demo ⊆ diagnosis ⊆ initial_lab ⊆ early_stay ⊆ full
hospital_view ⊆ full
```

代码中支持同一条样本在多个 view 下共享编码器参数前向传播，并计算：

1. **视图后验一致性**：coarse view 的风险分布接近 fine/full view 的风险分布；
2. **不确定性单调性**：信息越少，不确定性不应低于 full view；
3. **风险保留评估**：可计算 sparse/hospital view 相对于 full view 的性能保留率。

这一步替代普通 teacher-student 蒸馏。它不是“学生模仿老师”，而是把不同信息粒度的预测关系建模为粗化观测下的 posterior consistency。

### 1.2 Label-Marginalized Graph Energy Decoder：标签边缘化图能量解码器

对于六类合并症，模型不再使用六个彼此独立的 BCE 作为唯一核心，而是定义联合标签分布：

```text
p(Y | X) ∝ exp(Σ ψ_k(X)Y_k + Σ ψ_kl(X)Y_kY_l)
```

如果某个医院样本只有 4 个标签，另外 2 个标签未观测，训练时最大化边缘似然：

```text
L = -log Σ_{Y_missing} p(Y_observed, Y_missing | X)
```

由于当前合并症数量 K=6，所有标签组合只有 2^6=64 种，代码中进行**精确枚举边缘化**，不需要近似采样。

这比 `masked BCE` 更强：

- `masked BCE` 只是忽略未观测标签；
- 本模块让未观测标签通过联合标签结构、共病边和边缘化似然间接参与训练。

### 1.3 Snapshot-to-Trajectory Latent Transition：单次快照到潜在轨迹转移

针对首次入院患者没有历史轨迹的问题，v2 不再强依赖 longitudinal Transformer，而是学习一个潜在状态转移算子：

```text
z_t = q(z | X_snapshot, Y_current^O)
z_{t+Δ} = T(z_t, Δ)
p(Y_future | z_{t+Δ})
```

这意味着：

- 对有纵向记录的 MIMIC-IV 患者，模型学习群体层面的状态转移；
- 对首次入院患者，模型从单次快照推断其潜在病程位置，再通过转移算子预测未来风险；
- 医院数据即使没有第二次检测，也可以用于当前潜在状态校准，未来转移由公开纵向 EHR 学习。

### 1.4 Observation-Policy Disentanglement：观测策略解耦

缺失不是纯随机缺失，而可能来自：

- 医生检查选择；
- 医院系统字段缺失；
- 患者首诊阶段信息尚未产生；
- 数据源结构性不可用。

v2 显式拆分表示：

```text
z = z_disease ⊕ z_observation
```

其中：

- `z_disease` 用于疾病风险预测；
- `z_observation` 用于解释观测 mask、view/stage、数据可用性；
- 使用 gradient reversal 让 disease representation 尽量不携带 view/system 信息，增强跨医院泛化。

---

## 2. 当前工程结构

```text
模型名v2/
├── README.md
├── CODE_STATUS.md
├── requirements.txt
├── configs/
│   └── clsl_v2.yaml
├── clsl_v2/
│   ├── data/
│   │   ├── schema.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── components.py
│   │   ├── evidence_lattice_encoder.py
│   │   ├── graph_energy_decoder.py
│   │   ├── snapshot_transition.py
│   │   ├── observation_disentangle.py
│   │   └── model_v2.py
│   ├── losses/
│   │   ├── lattice_consistency.py
│   │   ├── label_marginalization.py
│   │   ├── transition_loss.py
│   │   └── disentangle_loss.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── metrics.py
│   │   ├── seed.py
│   │   └── training.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
└── scripts/
    ├── inspect_schema.py
    └── make_toy_data.py
```

---

## 3. 安装

```bash
cd 模型名v2
pip install -r requirements.txt
```

---

## 4. 先生成 toy data 验证流程

```bash
python scripts/make_toy_data.py --output data/toy_clsl_v2.csv --n 800
```

---

## 5. 训练

```bash
python -m clsl_v2.train \
  --config configs/clsl_v2.yaml \
  --data data/toy_clsl_v2.csv \
  --output runs/toy_v2 \
  --epochs 5 \
  --batch-size 64 \
  --device cpu
```

如果你使用自己的 MIMIC-IV 表格数据：

```bash
python -m clsl_v2.train \
  --config configs/clsl_v2.yaml \
  --data "E:/AAApaper/part3/data/data_fin/mimiciv_model_ready.csv" \
  --output runs/mimiciv_v2 \
  --epochs 30 \
  --batch-size 128 \
  --device cuda
```

---

## 6. 评估

```bash
python -m clsl_v2.evaluate \
  --run-dir runs/toy_v2 \
  --data data/toy_clsl_v2.csv \
  --split test \
  --view hospital_view
```

可以切换 view：

```bash
--view full
--view initial_lab
--view diagnosis
--view demo
--view hospital_view
```

---

## 7. 推理

```bash
python -m clsl_v2.infer \
  --run-dir runs/toy_v2 \
  --data data/toy_clsl_v2.csv \
  --view hospital_view \
  --output runs/toy_v2/predictions_hospital_view.csv
```

输出包含：

```text
sample_id
view
risk_af, risk_renal, risk_copd, risk_diabetes, risk_cad, risk_anemia
uncertainty_af, ...
evidence_reliability_af, ...
top_evidence_af, ...
```

---

## 8. 输入数据格式

默认六类标签列名：

```text
af_future, renal_future, copd_future, diabetes_future, cad_future, anemia_future
```

当前状态列名：

```text
af_current, renal_current, copd_current, diabetes_current, cad_current, anemia_current
```

如果医院数据只有部分标签，请将未观测标签留空或设为 NaN，例如：

```text
af_future=1, renal_future=0, copd_future=1, diabetes_future=0, cad_future=NaN, anemia_future=NaN
```

代码会自动生成 label mask，并使用标签边缘化似然训练，而不会把 NaN 当成 0。

如果你已经有显式标签观测 mask，也可以在配置中指定：

```yaml
label_mask_suffix: "_mask"
```

此时例如：

```text
af_future_mask=1
cad_future_mask=0
```

---

## 9. 当前版本适合作为什么

v2 适合作为正式实验的**方法原型和论文级代码基座**：

1. 可以在 MIMIC-IV 上训练 full / sparse / hospital-view；
2. 可以模拟医院只有 4 个标签；
3. 可以评估隐藏标签恢复能力；
4. 可以在首次入院单次快照下训练未来风险预测；
5. 可以输出风险、联合标签边缘概率、不确定性、证据可靠性和 top evidence。

---

## 10. 还没有完全实现的部分

当前 v2 是可运行的完整原型，但仍有几个下一步增强点：

1. **真实 visit-level 序列建模**  
   当前 Snapshot-to-Trajectory 使用潜在转移 MLP；如果你有 visit sequence，可以继续加入 temporal Transformer 或 neural ODE。

2. **更严格的 Evidence-Lattice 条件期望一致性**  
   当前使用 full-view risk 作为 fine posterior proxy；下一版可以用 conditional set aggregation 或 variational bound 近似 `E[p(Y|X_fine)|X_coarse]`。

3. **观测策略因果解耦**  
   当前用 adversarial view loss + observation mask decoder；下一版可以加入 hospital/site domain adversarial、IRM 或 invariant risk minimization。

4. **主动信息获取**  
   当前输出证据可靠性和 top evidence；下一版可加入 expected information gain，推荐“下一步补哪个检查”。

5. **更强的校准**  
   当前支持 Brier/ECE 评估；下一版可以加入 temperature scaling、deep ensemble 或 evidential uncertainty。

---

## 11. 推荐论文方法表述

> We propose Coarsened Latent State Learning (CLSL), a unified framework for clinical state prediction under coarsened observations. Instead of treating missing features, partially observed labels, and absent temporal trajectories as isolated problems, CLSL models them as different projections of an underlying complete latent state. The framework consists of an evidence-lattice encoder for view-consistent sparse evidence learning, a label-marginalized graph energy decoder for exact inference under partially observed multi-label states, and a snapshot-to-trajectory transition operator for future state prediction from single-visit observations.

---

## 12. 注意

本工程不会替你自动完成 MIMIC-IV SQL 抽取。它假设你已经有一个 `model_ready.csv/xlsx` 表格，并在 `configs/clsl_v2.yaml` 中配置好列名。
