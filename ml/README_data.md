# 数据版本管理 (DVC)

## 数据来源
Hugging Face `stanfordnlp/imdb` 公开数据集。

## 版本历史与变更
- **data-v1**：初次分层抽样，正负各 500 条，共 1000 条 (`imdb_v2.csv`)。
- **data-v2**：在 v1 注入 5% 标签噪声、空文本、超长文本等异常后，经过清洗去重、截断修复后的数据集 (`imdb_v4.csv`)。

## 版本对应关系
- **Git Tag `data-v1`** -> DVC 追踪文件 `data/imdb_v2.csv.dvc`
  - DVC Hash: `58e00f987dad3ae4593a75602592be43`
- **Git Tag `data-v2`** -> DVC 追踪文件 `data/imdb_v4.csv.dvc`
  - DVC Hash: `31addff6fad6e79839a5892dc4359650`

## 如何切换版本
```bash
# 切换到 v1 数据
git checkout data-v1
dvc checkout

# 切换到 v2 数据
git checkout data-v2
dvc checkout