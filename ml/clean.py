import pandas as pd


def main():
    df = pd.read_csv("data/imdb_v3.csv")
    initial_len = len(df)

    df = df[~df["text"].astype(str).str.contains(r"\[NOISE\]")]
    df = df[df["text"].astype(str).str.len() >= 5]
    df["text"] = df["text"].astype(str).str.slice(0, 1000)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df.to_csv("data/imdb_v4.csv", index=False)
    print(f"[OK] v4 cleaned data generated: {initial_len} -> {len(df)} rows")


if __name__ == "__main__":
    main()
