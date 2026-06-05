@echo off
python scripts\make_toy_data.py --output data\toy_clsl_v2.csv --n 400
python -m clsl_v2.train --config configs\clsl_v2.yaml --data data\toy_clsl_v2.csv --output runs\smoke_v2 --epochs 2 --batch-size 64 --device cpu
python -m clsl_v2.infer --run-dir runs\smoke_v2 --data data\toy_clsl_v2.csv --view hospital_view --output runs\smoke_v2\predictions.csv
