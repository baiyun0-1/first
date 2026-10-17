import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def main():
    print("正在从 Hugging Face 下载 IMDb 数据集...")
    ds = load_dataset("stanfordnlp/imdb", split="train")
    df = ds.to_pandas()

    print("正在分层抽样 1000 条数据（正负各 500）...")
    df_pos = df[df["label"] == 1].sample(n=500, random_state=42)
    df_neg = df[df["label"] == 0].sample(n=500, random_state=42)

    df_sample = (
        pd.concat([df_pos, df_neg])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    print("正在按 70/15/15 切分数据...")
    train_df, temp_df = train_test_split(
        df_sample, test_size=0.3, stratify=df_sample["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    output_path = "data/imdb_v2.csv"
    final_df.to_csv(output_path, index=False)

    print(f"✅ v2 干净数据已保存至: {output_path}")
    print(
        f"总行数: {len(final_df)} (train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)})"
    )


if __name__ == "__main__":
    main()
