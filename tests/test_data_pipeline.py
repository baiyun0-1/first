"""数据构建流程测试：分层抽样、切分比例、异常输入。"""

import pandas as pd
import pytest

from ml.data_pipeline import (
    LABEL_COLUMN,
    SPLIT_COLUMN,
    TEXT_COLUMN,
    build_dataset,
    sample_balanced,
    split_dataset,
)


def toy_frame(n_pos=50, n_neg=50):
    return pd.DataFrame(
        {
            TEXT_COLUMN: [f"pos {i}" for i in range(n_pos)]
            + [f"neg {i}" for i in range(n_neg)],
            LABEL_COLUMN: [1] * n_pos + [0] * n_neg,
        }
    )


# ------------------------------------------------------------ sample_balanced
def test_sample_balanced_returns_equal_classes():
    out = sample_balanced(toy_frame(30, 30), per_class=10, seed=7)

    assert len(out) == 20
    assert out[LABEL_COLUMN].value_counts().to_dict() == {1: 10, 0: 10}


@pytest.mark.parametrize("per_class", [0, -1], ids=["zero", "negative"])
def test_sample_balanced_rejects_bad_per_class(per_class):
    with pytest.raises(ValueError):
        sample_balanced(toy_frame(), per_class=per_class)


def test_sample_balanced_rejects_missing_label_column():
    with pytest.raises(ValueError, match=LABEL_COLUMN):
        sample_balanced(pd.DataFrame({TEXT_COLUMN: ["a", "b"]}))


def test_sample_balanced_rejects_single_class():
    df = pd.DataFrame({TEXT_COLUMN: ["a", "b"], LABEL_COLUMN: [1, 1]})

    with pytest.raises(ValueError):
        sample_balanced(df, per_class=1)


def test_sample_balanced_rejects_too_few_rows():
    with pytest.raises(ValueError, match="不足"):
        sample_balanced(toy_frame(3, 3), per_class=10)


# --------------------------------------------------------------- split_dataset
@pytest.mark.parametrize(
    "train_ratio,val_ratio",
    [
        (0.70, 0.30),  # 加起来等于 1，没有测试集
        (0.90, 0.15),  # 加起来超过 1
        (0.0, 0.15),  # 训练集比例非法
        (0.70, 0.0),  # 验证集比例非法
    ],
    ids=["no-test-left", "sum-over-1", "zero-train", "zero-val"],
)
def test_split_dataset_rejects_bad_ratios(train_ratio, val_ratio):
    with pytest.raises(ValueError):
        split_dataset(toy_frame(), train_ratio=train_ratio, val_ratio=val_ratio)


def test_split_dataset_rejects_empty_frame():
    with pytest.raises(ValueError):
        split_dataset(pd.DataFrame({TEXT_COLUMN: [], LABEL_COLUMN: []}))


def test_split_dataset_keeps_all_rows_and_expected_ratio():
    df = toy_frame(50, 50)
    out = split_dataset(df, seed=1)
    counts = out[SPLIT_COLUMN].value_counts()

    assert set(out[SPLIT_COLUMN]) == {"train", "val", "test"}
    assert len(out) == len(df)
    assert counts["train"] == 70
    assert counts["val"] == 15
    assert counts["test"] == 15


def test_split_dataset_is_stratified():
    out = split_dataset(toy_frame(50, 50), seed=1)

    for _, part in out.groupby(SPLIT_COLUMN):
        counts = part[LABEL_COLUMN].value_counts()
        assert abs(counts[1] - counts[0]) <= 1


# --------------------------------------------------------------- build_dataset
def test_build_dataset_end_to_end():
    out = build_dataset(toy_frame(60, 60), per_class=20, seed=3)

    assert len(out) == 40
    assert out[SPLIT_COLUMN].value_counts().sum() == 40
    assert set(out[SPLIT_COLUMN]) <= {"train", "val", "test"}
