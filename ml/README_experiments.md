# Experiment Summary / 实验总结报告

## Production-worthy Run / 具备投产价值的实验
- **EN**: The run trained on `data-v2` (cleaned and scaled data) is considered production-worthy.
- **ZH**: 基于 `data-v2`（经过清洗和扩容的数据集）进行的训练运行，被认为是最具备投产价值的版本。

## Metric Optimization / 核心优化指标
- **EN**: We optimize for **F1-score**. Because sentiment analysis needs to balance both Precision (not falsely flagging good reviews as bad) and Recall (not missing actual bad reviews). Accuracy alone is misleading if the dataset has class imbalance.
- **ZH**: 我们主要优化 **F1-score（F1 分数）**。因为情感分析需要在 Precision（精确率，避免将好评误判为差评）和 Recall（召回率，避免漏判真实的差评）之间取得平衡。如果数据集存在类别不平衡，单看 Accuracy（准确率）会产生误导。

## Experiment History / 实验历史记录
- **Baseline / 基线模型**: Trained on v1 dataset (1000 rows). F1: ~0.78
  - *中文*：在 v1 数据集（1000 条数据）上训练，F1 分数约为 0.78。
- **Data Disaster / 数据灾难演练**: Injected 20% noise. F1 dropped significantly.
  - *中文*：注入了 20% 的标签噪声和无效垃圾文本，F1 分数显著下降。
- **Improved / 改进版模型**: Trained on v2 cleaned dataset (2000 rows). F1 recovered and stabilized.
  - *中文*：在清洗后且扩充至 2000 条的 v2 数据集上重新训练，F1 分数成功恢复并保持稳定。