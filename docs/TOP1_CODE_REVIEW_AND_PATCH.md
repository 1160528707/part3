# CLSL-v2 Top-1区代码审阅与升级补丁

## 我看到的当前代码状态

当前仓库已经是一个可运行的 CLSL-v2 原型：它有 Evidence-Lattice Encoder、Graph Energy Decoder、Label Marginalization、Snapshot Transition、Observation-Policy heads，以及 train/evaluate/infer 脚本。

但从计算机类一区 Top 的角度，当前实现更像“强应用场景 + 多模块整合”。最需要升级的是：

1. `losses/lattice_consistency.py` 里 coarse-full 一致性仍是 same-sample MSE proxy，不足以支撑“条件期望式粗化学习”的 claim。
2. `graph_energy_decoder.py` 使用 global learned adjacency，个体化 disease interaction 不够强；而临床共病图本应是 patient-conditioned。
3. `snapshot_transition.py` 是单步 MLP residual，不具备时间半群一致性，审稿人可能认为它只是一个 future head。
4. `train.py` 已有多项 loss，但还没有把上述三个理论增强真正接入训练闭环。

## 补丁包含什么

- `clsl_v2/losses/lattice_consistency.py`：把 MSE 蒸馏升级为条件 lattice posterior consistency。
- `clsl_v2/losses/label_marginalization.py`：保留精确枚举，同时加入 masked-BCE 退化等价检查和 pairwise marginals。
- `clsl_v2/models/graph_energy_decoder.py`：把 fixed/global graph 升级为 patient-adaptive graph energy。
- `clsl_v2/models/snapshot_transition.py`：把单步 MLP 转移升级为 neural ODE-style residual flow，并加入 semigroup loss。
- `clsl_v2/train_top1_patch.py`：给出可复制到 `train.py` 的 `compute_batch_loss_top1`。
- `clsl_v2/analysis/active_information_gain.py`：补主动信息获取得分，可作为临床低信息场景下的附加贡献。

## 推荐集成顺序

1. 先替换 `losses/lattice_consistency.py` 和 `losses/label_marginalization.py`，跑 toy data smoke test。
2. 再替换 `models/snapshot_transition.py`，观察 loss 是否稳定。
3. 再替换 `models/graph_energy_decoder.py`，加入 graph regularization。
4. 最后把 `train_top1_patch.py` 中的 `compute_batch_loss_top1` 合并进原 `train.py`。

## 论文中可以怎么写

把方法标题从 “Coarsened Latent State Learning” 进一步强化为：

> Conditional Coarsened Latent State Learning with Patient-Adaptive Label Energy and Time-Consistent Latent Dynamics

核心贡献可以变成：

1. Conditional evidence-lattice posterior consistency：近似 `E[p(Y|X_fine)|X_coarse]`，而不是同一样本蒸馏。
2. Patient-adaptive graph energy：用 `A(x)` 而非固定共病图建模标签依赖。
3. Time-consistent snapshot-to-trajectory transition：用半群一致性约束支撑 latent disease dynamics。
4. Exact label marginalization under partial labels：证明 pairwise=0 时退化为 masked BCE。

## 必做 ablation

- full model
- w/o conditional lattice，换回 same-sample MSE
- w/o patient-adaptive graph，只用 global graph
- w/o semigroup transition
- w/o exact marginalization，换 masked BCE
- observed labels: 6/5/4/3
- views: demo/diagnosis/initial_lab/hospital_view/full
- external/site/time split
