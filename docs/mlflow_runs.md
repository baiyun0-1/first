# MLflow Run History

> 本文件由仓库中的 `mlflow.db` 导出，便于直接在 GitHub 上查看实验记录；
> 完整 run history 可在本地运行 `mlflow ui --backend-store-uri sqlite:///mlflow.db` 打开。

- 实验名：`IMDb_Sentiment`
- run 总数：15
- 追踪后端：SQLite（`mlflow.db`，已入库）
- 生产 run：`baseline-v1` / `81cf73188a7b4013bc5985130a48eb31`
- 生产模型：`models/model_v1.pkl`（md5 `4bb82b096fee6ec0aad4806317426cb6`）

## 生产 run 详情

- run_id：`81cf73188a7b4013bc5985130a48eb31`
- run name：`baseline-v1`
- git commit：`e73dd4d3984d00f384252d072357fa0d8ec2065f`
- 数据：`data/imdb_v2.csv`
- DVC md5：`48978ce7a8aa6e804962a2f4af700159`
- 超参：C=1.0, max_features=10000, min_df=1, ngram_range=(1, 2), sublinear_tf=False
- test_f1：0.832787
- artifact：`mlruns/1/81cf73188a7b4013bc5985130a48eb31/artifacts/model_v1.pkl`（`mlruns/` 未入库；同一文件已入库为 `models/model_v1.pkl`）

## 关键实验

| run_id | 数据 | DVC md5 | C | max_features | min_df | sublinear_tf | test_f1 |
|---|---|---|---|---|---|---|---|
| **`81cf7318`（生产）** | data/imdb_v2.csv | `48978ce7a8aa6e804962a2f4af700159` | 1.0 | 10000 | 1 | False | **0.8328** |
| `596a91e6`（improved-tuned） | data/imdb_v2.csv | `48978ce7a8aa6e804962a2f4af700159` | 2.0 | 30000 | 2 | True | 0.8212 |
| `45aa42de`（improved-v2-cleaned） | data/imdb_v4.csv | `1dbc6282ccf17c82dc5eaedb684e45fb` | 2.0 | 20000 | 1 | False | 0.7525 |

> `baseline-v1` 有 3 条同名 run（`19dd7837`、`81cf7318`、`aa7258a1`），生产模型以 `81cf7318` 为准；`improved-tuned` 也有 3 条同名 run，本表取最新一条 `596a91e6`。

## 全部 run

| 开始时间 | run_id | run name | 数据 | DVC md5 前缀 | git commit | test_f1 |
|---|---|---|---|---|---|---|
| 2026-10-18 02:46 | `5b32a87c` | dapper-fox-483 | — | — | — | 0.7483 |
| 2026-10-18 02:47 | `ecdbf5dd` | nosy-mole-266 | — | — | — | 0.7550 |
| 2026-10-18 03:08 | `bc42057d` | secretive-shark-163 | — | 58e00f98 | 2109eac6 | 0.8328 |
| 2026-10-19 09:33 | `7642cd56` | tfidf-lr-20261019-093328 | data/imdb_v2.csv | 48978ce7 | 849c9beb | 0.8328 |
| 2026-10-19 09:34 | `45aa42de` | improved-v2-cleaned | data/imdb_v4.csv | 1dbc6282 | 849c9beb | 0.7525 |
| 2026-10-19 09:36 | `e1a48955` | probe-tuned | data/imdb_v2.csv | 48978ce7 | 849c9beb | 0.8212 |
| 2026-10-19 09:37 | `7d8b3426` | probe-C0.5 | data/imdb_v2.csv | 48978ce7 | 849c9beb | 0.8054 |
| 2026-10-19 09:38 | `d0f5223b` | probe-C3 | data/imdb_v2.csv | 48978ce7 | 849c9beb | 0.8185 |
| 2026-10-19 09:39 | `2a51f127` | probe-C10 | data/imdb_v2.csv | 48978ce7 | 849c9beb | 0.8212 |
| 2026-10-19 09:40 | `19dd7837` | baseline-v1 | data/imdb_v2.csv | 48978ce7 | 849c9beb | 0.8328 |
| 2026-10-19 09:42 | `5f992b92` | improved-tuned | data/imdb_v2.csv | 48978ce7 | 849c9beb | 0.8212 |
| 2026-10-19 09:43 | `aa7258a1` | baseline-v1 | data/imdb_v2.csv | 48978ce7 | 849c9beb | 0.8328 |
| 2026-10-19 09:52 | `81cf7318` | baseline-v1 | data/imdb_v2.csv | 48978ce7 | e73dd4d3 | 0.8328 |
| 2026-10-19 09:54 | `2bf6688a` | improved-tuned | data/imdb_v2.csv | 48978ce7 | edecb38d | 0.8212 |
| 2026-10-19 09:56 | `596a91e6` | improved-tuned | data/imdb_v2.csv | 48978ce7 | 0617ab47 | 0.8212 |

## 如何复现

```bash
pip install -r requirements-dev.txt
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

生产模型复现：

```bash
python -m ml.train --config ml/configs/baseline.yaml
```
