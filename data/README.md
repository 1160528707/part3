# data 目录

本目录默认不放真实数据。你可以用以下命令生成 toy data：

```bash
python scripts/make_toy_data.py --output data/toy_clsl_v2.csv --n 800
```

真实 MIMIC-IV 或医院数据请不要直接提交到代码仓库；训练时通过 `--data` 指定路径。
