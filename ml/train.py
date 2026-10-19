"""训练 IMDb 情感分类模型，并把参数 / 指标 / 模型记录到 MLflow。

用法：
    python -m ml.train
    python -m ml.train --data data/imdb_v2.csv --model models/model_v1.pkl
"""

import argparse
import os
import subprocess
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

DEFAULT_DATA_PATH = "data/imdb_v2.csv"
DEFAULT_MODEL_PATH = "models/model_v1.pkl"
DEFAULT_DVC_FILE = "data/imdb_v2.csv.dvc"
DEFAULT_REGISTRY_PATH = "ml/registry/current_model.md"
DEFAULT_EXPERIMENT = "IMDb_Sentiment"
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

MAX_TEXT_LENGTH = 1000
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
SPLIT_COLUMN = "split"


def get_git_commit():
    """当前 Git Commit SHA；不在 git 仓库中运行时返回 unknown。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:  # noqa: BLE001 - 无 git 时降级，不影响训练
        return "unknown"


def get_dvc_hash(dvc_file_path=DEFAULT_DVC_FILE):
    """从 .dvc 指针文件里读取被跟踪数据的 md5。"""
    try:
        with open(dvc_file_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                # .dvc 里的写法是 "- md5: <hash>"，不能只用 startswith("md5:")
                if "md5:" in stripped:
                    return stripped.split("md5:", 1)[1].strip()
    except OSError:
        return "unknown"
    return "unknown"


def load_split(data_path=DEFAULT_DATA_PATH, split="train"):
    """读取数据文件的某个 split，并做基础校验。"""
    df = pd.read_csv(data_path)
    for col in (TEXT_COLUMN, LABEL_COLUMN, SPLIT_COLUMN):
        if col not in df.columns:
            raise ValueError(f"数据文件 {data_path} 缺少 '{col}' 列")
    subset = df[df[SPLIT_COLUMN] == split]
    if subset.empty:
        raise ValueError(f"数据文件 {data_path} 中没有 split='{split}' 的样本")
    return subset.reset_index(drop=True)


def build_pipeline(max_features=10000, ngram_range=(1, 2), C=1.0, random_state=42):
    """TF-IDF(1-2gram) + LogisticRegression 的推理流水线。"""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=ngram_range),
            ),
            ("clf", LogisticRegression(C=C, random_state=random_state)),
        ]
    )


def _prepare_text(series):
    return series.fillna("").astype(str).str.slice(0, MAX_TEXT_LENGTH)


def train_model(df, max_features=10000, ngram_range=(1, 2), C=1.0, random_state=42):
    """在 train split 上训练、test split 上评估，返回 (pipeline, f1)。"""
    if df is None or df.empty:
        raise ValueError("训练数据为空")
    for col in (TEXT_COLUMN, LABEL_COLUMN, SPLIT_COLUMN):
        if col not in df.columns:
            raise ValueError(f"训练数据缺少 '{col}' 列")

    train_df = df[df[SPLIT_COLUMN] == "train"]
    test_df = df[df[SPLIT_COLUMN] == "test"]
    if train_df.empty or test_df.empty:
        raise ValueError("train / test split 不能为空")

    pipeline = build_pipeline(
        max_features=max_features,
        ngram_range=ngram_range,
        C=C,
        random_state=random_state,
    )
    pipeline.fit(_prepare_text(train_df[TEXT_COLUMN]), train_df[LABEL_COLUMN])

    preds = pipeline.predict(_prepare_text(test_df[TEXT_COLUMN]))
    return pipeline, float(f1_score(test_df[LABEL_COLUMN], preds))


def write_registry(model_path, datapath_label, commit, dvc_hash, f1):
    """把「当前生产模型」写进 ml/registry/current_model.md。"""
    content = f"""# Current Production Model

- 模型文件：{model_path}
- 训练数据：{datapath_label}
- 训练脚本 commit：{commit[:8]} (短哈希)
- 数据 DVC Hash：{dvc_hash}
- 主要指标：test_f1 = {f1:.4f}
- 更新日期：{datetime.now().strftime('%Y-%m-%d')}
"""
    os.makedirs(os.path.dirname(DEFAULT_REGISTRY_PATH), exist_ok=True)
    with open(DEFAULT_REGISTRY_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] 已更新 {DEFAULT_REGISTRY_PATH}")


def main():
    parser = argparse.ArgumentParser(description="训练 IMDb 情感分类模型")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="训练数据 CSV")
    parser.add_argument(
        "--dvc-file", default=DEFAULT_DVC_FILE, help="数据对应的 .dvc 指针"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="模型输出路径")
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT, help="MLflow 实验名"
    )
    parser.add_argument("--run-name", default=None, help="MLflow run 名称")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--C", type=float, default=1.0, help="逻辑回归正则强度倒数")
    args = parser.parse_args()

    # 延迟导入：只是为了跑单元测试时不必安装/启动 mlflow
    import mlflow
    import mlflow.sklearn

    run_name = args.run_name or f"tfidf-lr-{datetime.now():%Y%m%d-%H%M%S}"
    commit = get_git_commit()
    dvc_hash = get_dvc_hash(args.dvc_file)

    print(f"1. 读取数据 {args.data} ...")
    df = pd.read_csv(args.data)

    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=run_name):
        print("2. 训练 TF-IDF + LogisticRegression ...")
        pipeline, f1 = train_model(df, max_features=args.max_features, C=args.C)

        mlflow.log_param("git_commit_sha", commit)
        mlflow.log_param("dvc_dataset_hash", dvc_hash)
        mlflow.log_param("data_path", args.data)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_text_length", MAX_TEXT_LENGTH)
        mlflow.log_metric("test_f1", f1)
        print(f"[OK] 训练完成，Test F1-score: {f1:.4f}")

        print("3. 保存模型 ...")
        os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
        joblib.dump(pipeline, args.model)
        mlflow.sklearn.log_model(pipeline, name="model")
        mlflow.log_artifact(args.model)
        print(f"[OK] 模型已保存至: {args.model}")

        write_registry(args.model, args.data, commit, dvc_hash, f1)

    return f1


if __name__ == "__main__":
    main()
