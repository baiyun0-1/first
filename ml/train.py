import sys

# 强制 utf-8 防中文乱码
sys.stdout.reconfigure(encoding="utf-8")

import os
import subprocess
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


# === 新增：自动获取 Git Commit SHA 和 DVC Hash 的工具函数 ===
def get_git_commit():
    """获取当前的 Git Commit SHA"""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def get_dvc_hash(dvc_file_path="data/imdb_v2.csv.dvc"):
    """从 .dvc 文件中提取数据的 MD5 Hash"""
    try:
        with open(dvc_file_path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split("md5:")[1].strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def train():
    # 设置 MLflow 实验名称
    mlflow.set_experiment("IMDb_Sentiment")

    with mlflow.start_run():
        # 获取版本信息
        code_ver = get_git_commit()
        data_ver = get_dvc_hash()

        print("1. 读取数据...")
        df = pd.read_csv("data/imdb_v2.csv")

        # 按 split 划分
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]

        print("2. 提取特征并训练模型 (TF-IDF + LogisticRegression)...")

        # 防止 NaN 报错
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

        # === 记录到 MLflow (包含作业要求的版本信息) ===
        mlflow.log_param("git_commit_sha", code_ver)
        mlflow.log_param("dvc_dataset_hash", data_ver)
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

        # 自动更新 registry 的 current_model.md
        md_content = f"""# Current Production Model

- 模型文件：{model_path}
- 训练脚本 commit：{code_ver[:8]} (短哈希)
- 数据版本：imdb_v2.csv (DVC Hash: {data_ver[:8]})
- 主要指标：test_f1 = {f1:.4f}
- 更新日期：{datetime.now().strftime('%Y-%m-%d')}
"""
        with open("ml/registry/current_model.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        print("[OK] 已更新 ml/registry/current_model.md")


if __name__ == "__main__":
    train()
