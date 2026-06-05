# CODE_STATUS：模型名v2 当前实现状态

## 当前版本

```text
模型名v2 / CLSL-v2
Coarsened Latent State Learning for Sparse and Partially Observed EHR
```

## 已实现

| 模块 | 状态 | 说明 |
|---|---:|---|
| 表格数据加载 csv/xlsx/parquet | ✅ | 自动识别 NaN，生成 feature mask 和 label mask |
| Feature schema 构建 | ✅ | 按配置将特征分到 demographic/current_state/lab/medication/procedure/trajectory 等模态 |
| Evidence views | ✅ | demo/diagnosis/initial_lab/early_stay/full/hospital_view |
| Feature tokenizer | ✅ | value + feature id + modality + mask + missing type + stage embedding |
| Evidence-Lattice Encoder | ✅ | 多 view 共享编码器，支持 lattice consistency loss |
| Modality reliability gate | ✅ | 按模态 coverage 和 stage 计算可靠性门控 |
| Disease-query attention | ✅ | 六个疾病 query 分别从证据 token 中提取疾病相关证据 |
| Latent disease state | ✅ | 输出 z_mu/z_logvar，支持 reparameterization |
| Snapshot-to-Trajectory transition | ✅ | 根据 delta_t_days 从当前 latent state 转移到未来 latent state |
| Graph Energy Decoder | ✅ | unary + pairwise energy，精确枚举 2^6 标签状态 |
| Label marginalization loss | ✅ | 对部分观测标签计算 exact marginal NLL |
| Current state head | ✅ | 可监督当前合并症状态 |
| Future risk head | ✅ | 可监督未来合并症状态 |
| Observation-policy disentanglement | ✅ | disease repr adversarial view classifier + obs mask/view decoder |
| ECE/Brier/AUC/F1 指标 | ✅ | 基础评估支持 |
| 推理输出风险和 top evidence | ✅ | 输出每个疾病风险、uncertainty、reliability、top evidence |
| Toy data 生成 | ✅ | 可直接 smoke test |

## v2 代码中的关键损失

```text
L_total =
  λ_future      * LabelMarginalNLL(Y_future^O)
+ λ_current     * LabelMarginalNLL(Y_current^O)
+ λ_lattice     * EvidenceLatticeConsistency
+ λ_entropy     * EntropyMonotonicity
+ λ_kl          * LatentKL
+ λ_obs_mask    * ObservationMaskReconstruction
+ λ_obs_view    * ObservationViewPrediction
+ λ_adv_view    * AdversarialViewConfusion
+ λ_calibration * OptionalBrier
```

## 与 v1 的关键区别

v1：

```text
mask + stage embedding + attention + GNN + basic partial label mask
```

v2：

```text
coarsened observation formulation
+ evidence-lattice consistency
+ exact label-marginalized graph energy model
+ snapshot-to-trajectory latent transition
+ observation-policy disentanglement
```

## 仍需升级

1. 真实 visit-level temporal module；
2. 更严格的条件期望式 lattice consistency；
3. 信息增益式检查推荐；
4. 跨医院 site/domain adversarial；
5. 与 MIMIC-III/eICU 的字段映射工具；
6. 图边先验从临床先验升级为“先验 + 数据驱动 + 患者自适应”。
