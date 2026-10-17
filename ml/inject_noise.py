import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("data/imdb_v2.csv")
    np.random.seed(42)
    flip_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[flip_indices, "label"] = 1 - df.loc[flip_indices, "label"]
    df.loc[flip_indices, "text"] = df.loc[flip_indices, "text"].astype(str) + " [NOISE]"

    empty_rows = pd.DataFrame(
        {"text": [""] * 5, "label": [1] * 5, "split": ["train"] * 5}
    )
    df = pd.concat([df, empty_rows], ignore_index=True)

    long_rows = pd.DataFrame(
        {"text": ["A" * 6000] * 5, "label": [0] * 5, "split": ["train"] * 5}
    )
    df = pd.concat([df, long_rows], ignore_index=True)

    dup_rows = df.iloc[:5].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data/imdb_v3.csv", index=False)
    print(f"[OK] v3 dirty data generated: {len(df)} rows")


if __name__ == "__main__":
    main()
