"""训练脚本测试：版本信息提取、流水线构建、训练/评估、异常输入、配置文件。"""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from ml.train import (
    LABEL_COLUMN,
    SPLIT_COLUMN,
    TEXT_COLUMN,
    build_pipeline,
    get_dvc_hash,
    get_git_commit,
    load_config,
    load_split,
    train_model,
    write_registry,
)


def toy_dataset(n=60):
    """构造正负样本线性可分的玩具数据集（前 70% 为 train）。"""
    rows = []
    for i in range(n):
        positive = i % 2 == 0
        rows.append(
            {
                TEXT_COLUMN: (
                    f"wonderful brilliant movie {i}"
                    if positive
                    else f"terrible awful movie {i}"
                ),
                LABEL_COLUMN: 1 if positive else 0,
                SPLIT_COLUMN: "train" if i < n * 0.7 else "test",
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 版本信息
def test_get_dvc_hash_reads_md5(tmp_path):
    dvc_file = tmp_path / "data.dvc"
    dvc_file.write_text("outs:\n- md5: abc123def\n  size: 1\n", encoding="utf-8")

    assert get_dvc_hash(str(dvc_file)) == "abc123def"


@pytest.mark.parametrize(
    "content",
    ["outs: []\n", "", "outs:\n- size: 1\n"],
    ids=["empty-outs", "empty-file", "no-md5"],
)
def test_get_dvc_hash_returns_unknown_for_unusable_file(tmp_path, content):
    dvc_file = tmp_path / "data.dvc"
    dvc_file.write_text(content, encoding="utf-8")

    assert get_dvc_hash(str(dvc_file)) == "unknown"


def test_get_dvc_hash_returns_unknown_for_missing_file(tmp_path):
    assert get_dvc_hash(str(tmp_path / "missing.dvc")) == "unknown"


def test_get_git_commit_returns_sha():
    sha = get_git_commit()
    if sha == "unknown":
        pytest.skip("不在 git 仓库中运行")

    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


# ------------------------------------------------------------------ 读取数据
def test_load_split_returns_requested_split(tmp_path):
    data_file = tmp_path / "data.csv"
    toy_dataset().to_csv(data_file, index=False)

    df = load_split(str(data_file), split="train")

    assert not df.empty
    assert set(df[SPLIT_COLUMN]) == {"train"}


def test_load_split_rejects_missing_columns(tmp_path):
    data_file = tmp_path / "data.csv"
    pd.DataFrame({TEXT_COLUMN: ["a"], LABEL_COLUMN: [1]}).to_csv(data_file, index=False)

    with pytest.raises(ValueError, match=SPLIT_COLUMN):
        load_split(str(data_file), split="train")


def test_load_split_rejects_unknown_split(tmp_path):
    data_file = tmp_path / "data.csv"
    toy_dataset().to_csv(data_file, index=False)

    with pytest.raises(ValueError, match="no-such-split"):
        load_split(str(data_file), split="no-such-split")


# ------------------------------------------------------------------ 流水线
def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline(max_features=100, ngram_range=(1, 2))

    assert [name for name, _ in pipeline.steps] == ["tfidf", "clf"]
    assert pipeline.named_steps["tfidf"].max_features == 100


# ------------------------------------------------------------------ 训练
def test_train_model_returns_usable_f1_and_predicts():
    pipeline, f1 = train_model(toy_dataset(), max_features=200)

    assert 0.0 <= f1 <= 1.0
    assert f1 > 0.9
    assert len(pipeline.predict(["wonderful brilliant movie"])) == 1


def test_train_model_pipeline_is_picklable(tmp_path):
    pipeline, _ = train_model(toy_dataset(), max_features=200)

    model_path = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)

    assert loaded.predict(["wonderful brilliant movie"])[0] == 1


@pytest.mark.parametrize(
    "bad_df",
    [
        pd.DataFrame(),
        pd.DataFrame({TEXT_COLUMN: ["x"], LABEL_COLUMN: [1]}),
    ],
    ids=["empty", "missing-split"],
)
def test_train_model_rejects_invalid_input(bad_df):
    with pytest.raises(ValueError):
        train_model(bad_df)


def test_train_model_rejects_missing_test_split():
    df = toy_dataset().assign(**{SPLIT_COLUMN: "train"})

    with pytest.raises(ValueError):
        train_model(df)


# ------------------------------------------------------------------ 配置文件
def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("run_name: demo\nC: 2.0\n", encoding="utf-8")

    cfg = load_config(str(config_file))

    assert cfg["run_name"] == "demo"
    assert cfg["C"] == 2.0


def test_load_config_rejects_non_mapping(tmp_path):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(str(config_file))


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))


@pytest.mark.parametrize("name", ["baseline.yaml", "improved.yaml", "cleaned-v2.yaml"])
def test_project_configs_are_usable(name):
    config_path = Path(__file__).resolve().parents[1] / "ml" / "configs" / name

    cfg = load_config(str(config_path))

    assert cfg["data"].startswith("data/")
    assert cfg["dvc_file"].endswith(".dvc")
    assert cfg["C"] > 0


# ------------------------------------------------------------------ registry
def test_write_registry_skips_non_production_model(tmp_path):
    """调参/对比实验的产物不能覆盖生产 registry。"""
    written = write_registry(
        str(tmp_path / "model_tuned.pkl"),
        "data/imdb_v2.csv",
        "abcdef12",
        "deadbeef",
        0.8,
    )

    assert written is False
