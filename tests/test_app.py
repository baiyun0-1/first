from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """测试正常路径：健康检查返回 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_missing_field():
    """测试边界情况：缺失 text 字段应返回 422"""
    response = client.post("/predict", json={"wrong_field": "hello"})
    assert response.status_code == 422


@patch("app.main.model")
def test_predict_mock_model(mock_model):
    """测试推理路径：使用 Mock 模型返回结果"""
    # 模拟模型的 predict 和 predict_proba 返回值
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]  # 80% 置信度

    response = client.post("/predict", json={"text": "Good movie"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert response.json()["confidence"] == 0.8
