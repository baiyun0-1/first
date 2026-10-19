"""IMDb 数据构建流程。

纯函数（sample_balanced / split_dataset / build_dataset）不依赖 Hugging Face，
可以直接用合成数据做单元测试；只有 main() 才真正下载数据集。
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
SPLIT_COLUMN = "split"

DEFAULT_PER_CLASS = 1000
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15


def sample_balanced(df, per_class=DEFAULT_PER_CLASS, seed=DEFAULT_SEED):
    """按标签分层抽样，每个类别取 per_class 条，返回打乱后的 DataFrame。"""
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"数据集缺少 '{LABEL_COLUMN}' 列")
    if per_class <= 0:
        raise ValueError("per_class 必须是正整数")

    labels = sorted(df[LABEL_COLUMN].dropna().unique())
    if len(labels) != 2:
        raise ValueError(f"期望二分类标签，实际为 {labels}")

    parts = []
    for label in labels:
        subset = df[df[LABEL_COLUMN] == label]
        if len(subset) < per_class:
            raise ValueError(
                f"标签 {label} 只有 {len(subset)} 条，不足 per_class={per_class} 条"
            )
        parts.append(subset.sample(n=per_class, random_state=seed))

    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def split_dataset(
    df,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    seed=DEFAULT_SEED,
):
    """按 70/15/15 分层切分，并把切分结果写入 split 列。"""
    if df is None or df.empty:
        raise ValueError("数据集为空，无法切分")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"数据集缺少 '{LABEL_COLUMN}' 列")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio 和 val_ratio 必须在 (0, 1) 之间")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1（否则没有测试集）")

    # 用整数样本数而不是浮点比例，避免 1 - 0.7 - 0.15 的浮点误差导致切分偏一条
    n_rows = len(df)
    n_train = round(n_rows * train_ratio)
    n_val = round(n_rows * val_ratio)
    if min(n_train, n_val) < 1 or n_rows - n_train - n_val < 1:
        raise ValueError("数据量太小，无法切出 train / val / test 三个非空集合")

    train_df, holdout_df = train_test_split(
        df,
        train_size=n_train,
        stratify=df[LABEL_COLUMN],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        holdout_df,
        train_size=n_val,
        stratify=holdout_df[LABEL_COLUMN],
        random_state=seed,
    )

    train_df = train_df.assign(**{SPLIT_COLUMN: "train"})
    val_df = val_df.assign(**{SPLIT_COLUMN: "val"})
    test_df = test_df.assign(**{SPLIT_COLUMN: "test"})

    return (
        pd.concat([train_df, val_df, test_df])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def build_dataset(
    df,
    per_class=DEFAULT_PER_CLASS,
    seed=DEFAULT_SEED,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
):
    """抽样 + 切分，返回可以直接落盘的 DataFrame。"""
    sampled = sample_balanced(df, per_class=per_class, seed=seed)
    return split_dataset(
        sampled, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )


def main():
    # 延迟导入：跑单元测试时不需要安装 datasets / 访问 Hugging Face
    from datasets import load_dataset

    print("正在从 Hugging Face 下载 IMDb 数据集...")
    ds = load_dataset("stanfordnlp/imdb", split="train")
    df = ds.to_pandas()

    print(
        f"正在分层抽样 {DEFAULT_PER_CLASS * 2} 条数据（正负各 {DEFAULT_PER_CLASS}）..."
    )
    final_df = build_dataset(df, per_class=DEFAULT_PER_CLASS, seed=DEFAULT_SEED)

    os.makedirs("data", exist_ok=True)
    output_path = "data/imdb_v2.csv"
    final_df.to_csv(output_path, index=False)

    counts = final_df[SPLIT_COLUMN].value_counts().to_dict()
    print(f"[OK] v2 干净数据已保存至: {output_path}")
    print(
        f"总行数: {len(final_df)} (train: {counts.get('train')}, "
        f"val: {counts.get('val')}, test: {counts.get('test')})"
    )


if __name__ == "__main__":
    main()
