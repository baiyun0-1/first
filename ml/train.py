import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


def train():
    print("1. 读取数据...")
    df = pd.read_csv("data/imdb_v2.csv")

    # 按 split 划分
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    print("2. 提取特征并训练模型 (TF-IDF + LogisticRegression)...")
    # 限制 max_features 防止内存过大，且必须在 1000 字符内截断（符合我们的定义书）
    train_df["text"] = train_df["text"].astype(str).str.slice(0, 1000)
    test_df["text"] = test_df["text"].astype(str).str.slice(0, 1000)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # 评估
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    print(f"[OK] 训练完成，Test F1-score: {f1:.4f}")

    print("3. 保存模型...")
    # 确保 models 目录存在
    os.makedirs("models", exist_ok=True)
    model_path = "models/model_v1.pkl"

    # 使用 Pipeline 把向量器和模型打包在一起，方便后续预测
    pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
    joblib.dump(pipeline, model_path)
    print(f"[OK] 模型已保存至: {model_path}")

    # 自动更新 registry 的 current_model.md
    md_content = f"""# Current Production Model

- 模型文件：{model_path}
- 训练脚本 commit：待补充
- 数据版本：imdb_v2.csv (1000条)
- 主要指标：test_f1 = {f1:.4f}
- 更新日期：{datetime.now().strftime('%Y-%m-%d')}
"""
    with open("ml/registry/current_model.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("[OK] 已更新 ml/registry/current_model.md")


if __name__ == "__main__":
    train()
