"""API 层测试：健康检查、入参校验、推理路径、模型缺失与模型加载。"""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import app.main as app_main
from app.main import app

client = TestClient(app)


class FakeModel:
    """可控的假模型，用于验证 API 的映射/截断逻辑。"""

    def __init__(self, label=1, proba=(0.2, 0.8)):
        self.label = label
        self.proba = list(proba)
        self.seen_lengths = []

    def predict(self, texts):
        self.seen_lengths.extend(len(t) for t in texts)
        return [self.label]

    def predict_proba(self, texts):
        return [self.proba]


# ---------------------------------------------------------------- /health
@pytest.mark.parametrize("expected_key", ["status", "model_loaded"])
def test_health_contract(expected_key, monkeypatch):
    monkeypatch.setattr(app_main, "model", None)
    response = client.get("/health")

    assert response.status_code == 200
    assert expected_key in response.json()


def test_health_reports_missing_model(monkeypatch):
    monkeypatch.setattr(app_main, "model", None)
    assert client.get("/health").json()["model_loaded"] is False


# ------------------------------------------------------- 入参边界（422）
@pytest.mark.parametrize(
    "payload",
    [
        {"wrong_field": "hello"},  # 字段名写错
        {},  # 空对象
        {"text": 123},  # 类型错误
        {"text": None},  # null
        {"text": ""},  # 空字符串（min_length=1 拦截）
    ],
    ids=["wrong-field", "empty-object", "wrong-type", "null-text", "empty-string"],
)
def test_predict_rejects_invalid_payload(payload):
    assert client.post("/predict", json=payload).status_code == 422


# ------------------------------------------------- 模型缺失时应 503 而非 200
def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(app_main, "model", None)
    response = client.post("/predict", json={"text": "a good movie"})

    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


# -------------------------------------------------------- 正常推理路径
@pytest.mark.parametrize(
    "label,proba,expected",
    [
        (1, (0.2, 0.8), "positive"),
        (0, (0.9, 0.1), "negative"),
    ],
    ids=["positive", "negative"],
)
def test_predict_maps_label_to_sentiment(monkeypatch, label, proba, expected):
    monkeypatch.setattr(app_main, "model", FakeModel(label, proba))
    body = client.post("/predict", json={"text": "some review"}).json()

    assert body["sentiment"] == expected
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_confidence_is_rounded(monkeypatch):
    monkeypatch.setattr(app_main, "model", FakeModel(1, (0.1234567, 0.8765433)))
    body = client.post("/predict", json={"text": "some review"}).json()

    assert body["confidence"] == 0.8765


def test_predict_truncates_long_text(monkeypatch):
    fake = FakeModel(1, (0.1, 0.9))
    monkeypatch.setattr(app_main, "model", fake)

    client.post("/predict", json={"text": "a" * 5000})

    assert fake.seen_lengths == [app_main.MAX_TEXT_LENGTH]


# --------------------------------------------------- lifespan 真正加载模型
def test_lifespan_loads_model_from_disk(tmp_path, monkeypatch):
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(["good", "great", "bad", "awful"], [1, 1, 0, 0])

    model_file = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_file)

    monkeypatch.setattr(app_main, "MODEL_PATH", str(model_file))
    monkeypatch.setattr(app_main, "model", None)

    with TestClient(app) as scoped_client:
        assert scoped_client.get("/health").json()["model_loaded"] is True
        response = scoped_client.post("/predict", json={"text": "good"})
        assert response.status_code == 200
        assert response.json()["sentiment"] in {"positive", "negative"}
