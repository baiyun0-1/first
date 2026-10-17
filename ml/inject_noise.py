import os

import numpy as np
import pandas as pd


def main():
    print("正在加载 v2 干净数据...")
    df = pd.read_csv("data/imdb_v2.csv")

    print("开始注入重度异常（生成 v3 脏数据）...")
    np.random.seed(42)

    # 1. 注入 200 条纯乱码（无用信息）
    junk_rows = pd.DataFrame(
        {
            "text": ["asdfghjkl;qwertyuiop"] * 200,
            "label": [1] * 200,
            "split": ["train"] * 200,
        }
    )

    # 2. 注入 100 条空文本
    empty_rows = pd.DataFrame(
        {"text": [""] * 100, "label": [1] * 100, "split": ["train"] * 100}
    )

    # 3. 注入 200 条重复的“水军”评论
    spam_rows = pd.DataFrame(
        {
            "text": ["This is the best movie ever!"] * 200,
            "label": [1] * 200,
            "split": ["train"] * 200,
        }
    )

    df = pd.concat([df, junk_rows, empty_rows, spam_rows], ignore_index=True)

    # 4. 稍微翻转 5% 的标签（保留一点真实世界的无奈，这一般洗不掉）
    flip_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[flip_indices, "label"] = 1 - df.loc[flip_indices, "label"]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/imdb_v3.csv", index=False)
    print(f"[OK] v3 脏数据已生成: data/imdb_v3.csv (共 {len(df)} 行，包含大量垃圾数据)")


if __name__ == "__main__":
    main()
