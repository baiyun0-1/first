# 数据版本管理 (DVC)

## 1. 数据来源

- 数据集：Hugging Face `stanfordnlp/imdb`（公开数据集，25k 训练样本）。
- 抽样方式：正负各 1000 条，随机种子 `42`，按 70/15/15 分层切分。
- 生成脚本：[`ml/data_pipeline.py`](data_pipeline.py)

## 2. 版本链

```text
stanfordnlp/imdb
      │  python -m ml.data_pipeline
      ▼
data/imdb_v2.csv  2000 条（1400/300/300）   ← data-v1
      │  python -m ml.inject_noise（注入 5% 标签翻转 + 500 条垃圾样本）
      ▼
data/imdb_v3.csv  2500 条（1900/300/300）   ← 中间态，用于「数据灾难演练」
      │  python -m ml.clean（去空文本 / 去乱码 / 去重 / 截断）
      ▼
data/imdb_v4.csv  2001 条（1401/300/300）   ← data-v2
```

- `inject_noise.py` 注入的内容：200 条纯乱码、100 条空文本、200 条重复水军评论，并对 5% 的样本做标签翻转。
- `clean.py` 只能清掉空文本、乱码和重复样本，**无法修复被翻转的标签**，所以 `data-v2` 的模型效果反而低于 `data-v1`（见 [`README_experiments.md`](README_experiments.md)）。

## 3. 版本 → 文件 → Commit 对应关系

| 数据版本 | 文件 | 行数 | DVC md5 | Git tag | Tag 所在 commit |
|----------|------|------|---------|---------|-----------------|
| `data-v1` | `data/imdb_v2.csv` | 2000 | `48978ce7a8aa6e804962a2f4af700159` | `data-v1` | `e152ed9192551d27828661eb62fb77616624e1eb` |
| —（中间态） | `data/imdb_v3.csv` | 2500 | `c6cdef524357df9e75fd6c8279c2a8a5` | 无 | — |
| `data-v2` | `data/imdb_v4.csv` | 2001 | `1dbc6282ccf17c82dc5eaedb684e45fb` | `data-v2` | `654c13a87be834485934a2f94eefff3596ed508e` |

对应的指针文件是 `data/imdb_v2.csv.dvc`、`data/imdb_v3.csv.dvc`、`data/imdb_v4.csv.dvc`，原始 CSV 不会进入 Git（`.gitignore` 用 `data/*` + `!data/*.dvc` 只放行指针）。

## 4. 远端存储

- 类型：Azure Blob Storage（课程提供），DVC remote 名为 `myazure`。
- 地址：`azure://dvc`，`account_name = wut2026`（见 `.dvc/config`，可提交，不含密钥）。
- 凭证：容器级 SAS token，写在 `.dvc/config.local`，由 `.dvc/.gitignore` 忽略，**不会进入 Git**。
- SAS 有效期至 2026-12-12，过期后需重新生成并覆盖 `.dvc/config.local`。

## 5. 如何复现某个数据版本

```bash
# 拉取当前 commit 需要的数据
dvc pull

# 切换到 data-v1
git checkout data-v1
dvc checkout

# 切换到 data-v2
git checkout data-v2
dvc checkout

# 查看工作区数据与指针是否一致
dvc status
```

## 6. 变更记录

- 2026-10-19：修正 `.gitignore`（`data/` → `data/*` + `!data/*.dvc`），否则 `.dvc` 指针文件被忽略、`dvc add` 会直接报错；三个指针全部按工作区文件重新 `dvc add`，并 `dvc push` 到 Azure。
- 2026-10-17：首次建立 `data-v1` / `data-v2` 两个版本。
