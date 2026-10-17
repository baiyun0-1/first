import sys

# 强制 utf-8 防中文乱码
sys.stdout.reconfigure(encoding="utf-8")

import os
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


def train():
    # 设置 MLflow 实验名称
    mlflow.set_experiment("IMDb_Sentiment")

    with mlflow.start_run():
        print("1. 读取数据...")
        df = pd.read_csv("data/imdb_v2.csv")

        # 按 split 划分
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]

        print("2. 提取特征并训练模型 (TF-IDF + LogisticRegression)...")

        # 【修复点】：加上 .fillna("") 防止 NaN 报错
        train_df["text"] = train_df["text"].fillna("").astype(str).str.slice(0, 1000)
        test_df["text"] = test_df["text"].fillna("").astype(str).str.slice(0, 1000)

        max_features = 10000
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
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

        # 记录到 MLflow
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("test_f1", f1)

        print("3. 保存模型...")
        os.makedirs("models", exist_ok=True)
        model_path = "models/model_v1.pkl"

        pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
        joblib.dump(pipeline, model_path)

        # 把模型存入 MLflow
        mlflow.sklearn.log_model(pipeline, "model")
        print(f"[OK] 模型已保存至: {model_path}")

        md_content = f"""# Current Production Model

- 模型文件：{model_path}
- 训练脚本 commit：待补充
- 数据版本：imdb_v2.csv
- 主要指标：test_f1 = {f1:.4f}
- 更新日期：{datetime.now().strftime('%Y-%m-%d')}
"""
        with open("ml/registry/current_model.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        print("[OK] 已更新 ml/registry/current_model.md")


if __name__ == "__main__":
    train()
