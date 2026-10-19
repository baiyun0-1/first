# 模型与数据溯源表

记录「代码 commit → 数据版本 → 模型文件」的对应关系，保证生产模型可复现。

## 1. 生产模型

| 项目 | 值 |
|------|-----|
| 模型文件 | `models/model_v1.pkl` |
| 数据版本 | `data-v1`（`data/imdb_v2.csv`，2000 条） |
| 数据 DVC md5 | `48978ce7a8aa6e804962a2f4af700159` |
| 数据 tag / commit | `data-v1` → `e152ed9192551d27828661eb62fb77616624e1eb` |
| 训练脚本 commit | `e73dd4d3984d00f384252d072357fa0d8ec2065f` |
| 超参配置 | `ml/configs/baseline.yaml`（C=1.0, max_features=10000, ngram=(1,2)） |
| MLflow run | `baseline-v1`（run_id: `81cf73188a7b4013bc5985130a48eb31`） |
| 主要指标 | test_f1 = 0.8328 |

> 说明：MLflow 里有 3 条同名 `baseline-v1`（重复实验），生产模型以 run_id `81cf73188a7b4013bc5985130a48eb31` 为准；该 run 的 `model_v1.pkl` 与仓库中的 `models/model_v1.pkl` md5 一致。

## 2. 其他数据版本

| 版本 | 文件 | DVC md5 | Tag 所在 commit | 用于 |
|------|------|---------|-----------------|------|
| `data-v1` | `data/imdb_v2.csv` | `48978ce7...` | `e152ed9...` | 生产模型、`baseline-v1`、`improved-tuned` |
| 中间态 | `data/imdb_v3.csv` | `c6cdef52...` | 未打 tag | 噪声注入产物，`clean.py` 的输入 |
| `data-v2` | `data/imdb_v4.csv` | `1dbc6282...` | `654c13a...` | `improved-v2-cleaned`（未投产） |

## 3. 如何验证

```bash
# 1. 代码版本
git log --oneline -1

# 2. 数据版本（应与上面 md5 一致）
cat data/imdb_v2.csv.dvc
dvc status

# 3. 模型指标（用登记的数据与配置重跑，应得到 0.8328）
python -m ml.train --config ml/configs/baseline.yaml

# 4. 查看 MLflow 中的 run 详情（生产 run: 81cf73188a7b4013bc5985130a48eb31）
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 4. 备注

- 2026-10-19 之前，`.dvc` 指针在数据扩容后没有重新 `dvc add`，导致 `secretive-shark-163` 这条 run 记录的是旧哈希 `58e00f98...`（指向 1000 行的旧文件）；实际训练用的已经是 2000 行数据。当前指针已刷新为 `48978ce7...`，上表以当前指针为准。
- `models/model_v1.pkl` 与 `data/` 不同：模型只有约 450 KB，且推理镜像必须自带模型，因此模型提交进 Git；数据仍然只提交 `.dvc` 指针。
