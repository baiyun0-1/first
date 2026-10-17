import pandas as pd

def main():
    df = pd.read_csv("data/imdb_v3.csv")
    initial_len = len(df)
    print("开始清洗（生成 v4 清洗数据）...")
    
    # 1. 剔除空文本
    df = df[df["text"].astype(str).str.len() >= 5]
    
    # 2. 截断超长文本
    df["text"] = df["text"].astype(str).str.slice(0, 1000)
    
    # 3. 去重（干掉那 200 条重复的水军评论）
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    # 4. 干掉纯乱码（包含 asdf 的）
    df = df[~df["text"].str.contains("asdfgh")]
    
    df.to_csv("data/imdb_v4.csv", index=False)
    print(f"[OK] v4 清洗后数据已生成: data/imdb_v4.csv (共 {len(df)} 行)")
    print(f"清洗前: {initial_len} 行 -> 清洗后: {len(df)} 行")

if __name__ == "__main__":
    main()