# IMDb 影评情感分类器
### —— 一个带数据版本管理的轻量 NLP 分类服务

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Task](https://img.shields.io/badge/Task-Text%20Classification-green.svg)
![Data](https://img.shields.io/badge/Dataset-IMDb-orange.svg)
![DVC](https://img.shields.io/badge/Data%20Version-DVC-red.svg)
![MLflow](https://img.shields.io/badge/Experiment%20Tracking-MLflow-lightgrey.svg)

---

## 📑 目录

- [1. 任务定义](#1-任务定义)
- [2. 数据方案](#2-数据方案)
- [3. 模型方案](#3-模型方案)
- [4. 数据版本链设计（工程主线）](#4-数据版本链设计工程主线)
- [5. Git 与 Secrets 规范](#5-git-与-secrets-规范)
- [6. 验收标准](#6-验收标准)
- [7. 阶段路线图](#7-阶段路线图)

---

## 1. 任务定义

| 项目 | 内容 |
|------|------|
| **任务类型** | 文本二分类（轻量 NLP） |
| **输入** | 一条英文影评文本（`string`，截断至 1000 字符） |
| **输出** | 正面（`1`）/ 负面（`0`） |
| **主指标** | **F1-score**（正类 = 正面） |
| **辅指标** | 5 折交叉验证的 F1 均值 ± 标准差；test 集固定，不参与调参 |

### 💡 业务解释

> F1 低 → 差评被漏判为好评 → 舆情监控失效。
>
> F1 同时约束「漏放」和「错杀」，比 Accuracy 更贴合客服 / 舆情场景。

---

## 2. 数据方案

| 项目 | 内容 |
|------|------|
| **数据集** | `stanfordnlp/imdb`（Hugging Face，公开数据集） |
| **规模** | 1000 条，正负各 500 |
| **切分** | 70 / 15 / 15 分层：train 700 / val 150 / test 150 |
| **随机种子** | `random_state=42`（生成、抽样、切分、训练全部固定） |
| **数据版本链** | v2 干净 → v3 脏 → v4 清洗后，三者均用 DVC 追踪 |
| **可扩展性** | 脚本暴露 `n_samples` 参数，未来可扩到 5000 / 25000，v3/v4 脚本不变 |

---

## 3. 模型方案

| 层级 | 模型 | 说明 |
|------|------|------|
| **基线（必做）** | TF-IDF（1–2 gram, `max_features=10000`）+ 逻辑回归 | 秒级训练，作为「地板指标」 |
| **可选 fine-tune** | SetFit（few-shot）或 DistilBERT | 只训分类头 + 最后 2 层 |

### ⚠️ 约束

- 700 条训练集偏小，**必须用 val 做 early stopping**；不做全参数微调。
- **禁止**大模型全参数微调、需要 GPU 数小时的方案（超出 2–3 周范围）。

---

## 4. 数据版本链设计（工程主线）

这是本项目的**工程核心**——演示「脏数据毁模型、清洗救回来」的完整证据链。

| 版本 | 来源 | 内容 | DVC 命令 |
|------|------|------|----------|
| **v2 干净** | IMDb 分层抽样 | 1000 条，正负各 500，无缺失 | `dvc add data/imdb_v2.csv` |
| **v3 脏** | 从 v2 注入噪声 | ① 5% 标签翻转（约 50 条）<br>② 空文本 5 条<br>③ 超长文本 5 条（>5000 字符）<br>④ 重复文本 5 条 | `dvc add data/imdb_v3.csv` |
| **v4 清洗** | 从 v3 按规则修复 | 丢弃空文本、截断超长、去重、剔除噪声标签行（或规则修正） | `dvc add data/imdb_v4.csv` |

### 🔑 关键约束

> **v4 必须从 v3 清洗得来，不能复制 v2。**
>
> 这样 `dvc diff` 才能完整演示「脏 → 净」的差异，形成可复现的工程证据。

---

## 5. Git 与 Secrets 规范

| 项目 | 内容 |
|------|------|
| **分支模型** | `feature/*` → `dev` → `staging` → `main` |
| **保护规则** | `main`：PR + 2 approvals + up-to-date；`dev`：PR + 1 approval |
| **Environments** | `staging`、`production`；production 只允许 main 部署 |
| **Secrets 清单** | `STAGING_ENV_FILE`、`PRODUCTION_ENV_FILE`、`AZURE_STORAGE_ACCOUNT`、`AZURE_STORAGE_KEY` |

### `.gitignore` 模板

```gitignore
.env
*.env
data/
!data.dvc
mlruns/
mlartifacts/
__pycache__/
*.pkl
*.pt
*.h5
models/
.venv/
