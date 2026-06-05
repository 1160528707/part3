# CLSL-v2 Top-1区创新补丁包

这个补丁包不是替你“美化代码”，而是把当前 CLSL-v2 从原型提升为更容易被计算机类一区 Top 审稿认可的算法实现。

请把 `clsl_v2/` 下的同名文件复制到你的仓库中覆盖，或逐个 diff 合并。`train_top1_patch.py` 不是自动覆盖文件，而是训练逻辑补丁；你可以把其中的 `compute_batch_loss_top1` 合并到原 `train.py`。

运行测试：

```bash
PYTHONPATH=. pytest tests/test_top1_losses.py
```

最小集成：

```bash
cp clsl_v2/losses/lattice_consistency.py  <your_repo>/clsl_v2/losses/lattice_consistency.py
cp clsl_v2/losses/label_marginalization.py <your_repo>/clsl_v2/losses/label_marginalization.py
cp clsl_v2/models/snapshot_transition.py <your_repo>/clsl_v2/models/snapshot_transition.py
cp clsl_v2/models/graph_energy_decoder.py <your_repo>/clsl_v2/models/graph_energy_decoder.py
```

然后把 `configs/clsl_v2_top1.yaml` 里的 loss/model 新字段合并进你的主配置。
