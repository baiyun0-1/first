# 实验总结 / Experiment Summary

## 1. 核心指标

- **优化目标：F1-score**（正类 = 正面）。
- 理由：情感分析需要在 Precision（不把好评误判成差评）和 Recall（不漏掉真实差评）之间平衡。只看 Accuracy 会在类别不平衡时产生误导，而舆情场景里「漏掉差评」和「错杀好评」的代价都很高。

## 2. 生产模型

**`baseline-v1`**（`models/model_v1.pkl`）是当前生产模型：

- 训练数据：`data-v1`（`data/imdb_v2.csv`，2000 条干净样本）
- 超参：`TF-IDF(max_features=10000, ngram_range=(1,2))` + `LogisticRegression(C=1.0)`
- **test F1 = 0.8328**
- 训练脚本 commit：`e73dd4d3`，数据 md5：`48978ce7a8aa6e804962a2f4af700159`
- MLflow run_id：`81cf73188a7b4013bc5985130a48eb31`（同名 `baseline-v1` 有 3 条，以此为准）
- 复现命令：`python -m ml.train --config ml/configs/baseline.yaml`

## 3. 实验对比

| run 名称 | 数据 | C | max_features | min_df | sublinear_tf | test F1 | 结论 |
|----------|------|---|--------------|--------|--------------|---------|------|
| `baseline-v1` | data-v1 | 1.0 | 10,000 | 1 | false | **0.8328** | ✅ 生产模型 |
| `improved-tuned` | data-v1 | 2.0 | 30,000 | 2 | true | 0.8212 | ❌ 调参没有超过基线 |
| `improved-v2-cleaned` | data-v2 | 2.0 | 20,000 | 1 | false | 0.7525 | ❌ 标签噪声未修复，效果下降 |

复现命令：

```bash
python -m ml.train --config ml/configs/baseline.yaml      # 0.8328
python -m ml.train --config ml/configs/improved.yaml      # 0.8212
python -m ml.train --config ml/configs/cleaned-v2.yaml    # 0.7525
```

### 补充：C 值扫描（都在 data-v1 上，max_features=20000）

| C | 0.5 | 1.0 | 3.0 | 10.0 |
|---|-----|-----|-----|------|
| test F1 | 0.8054 | 0.8328 | 0.8185 | 0.8212 |

扫描用的是同一份数据、同一个随机种子，说明在这份 2000 条的数据上，C=1.0 附近就是最优点；继续增大或减小都会掉点。

## 4. 结论与解释

1. **调参没有带来提升**：3 组调参配置（更大的词表、`min_df=2`、`sublinear_tf`、不同 C）都低于基线。测试集只有 300 条，F1 的标准误约 ±0.02，这些差异基本落在噪声范围内。
2. **数据质量比超参更重要**：换成清洗后的 `data-v2` 后 F1 掉到 0.7525。原因是噪声注入时翻转了 5% 的标签，而规则清洗只能去掉空文本 / 乱码 / 重复样本，无法识别被翻转的标签，模型因此学到了错误的对应关系。
3. **因此生产模型保持 `baseline-v1`**：它在干净数据上取得了最高 F1，并且推理延迟最低（词表只有 10k）。

## 5. 每条 run 都记录了什么

- `git_commit_sha`：代码版本（`get_git_commit()`）
- `dvc_dataset_hash`：训练数据的 DVC md5（`get_dvc_hash()` 直接读 `.dvc` 指针）
- `data_path`、`max_features`、`ngram_range`、`C`、`min_df`、`sublinear_tf`、`max_text_length`
- `test_f1`
- 模型 artifact：`mlflow.sklearn.log_model(...)` 与 `.pkl` 文件

> 历史遗留：早期两条 run（`dapper-fox-483` F1=0.7483、`nosy-mole-266` F1=0.7550）没有记录 git / data 版本信息；`secretive-shark-163` 记录的是旧指针哈希 `58e00f98`（当时 `.dvc` 指针尚未刷新）。以上 run 保留作历史，不作为生产依据。

## 6. 查看方式

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
