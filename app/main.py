
import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 默认路径，但允许通过环境变量 MODEL_PATH 覆盖
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_v1.pkl")
model = None


# 使用最新的 lifespan 机制加载模型，替代被弃用的 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[OK] 成功加载模型: {MODEL_PATH}")
    else:
        print(f"[WARN] 找不到模型: {MODEL_PATH}，请先运行训练脚本。")
    yield


app = FastAPI(title="IMDb Sentiment API", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}

    # 预测
    text = request.text[:1000]  # 截断保护
    pred = model.predict([text])[0]

    # 修复点：使用内置的 max() 函数，兼容 list 和 numpy array
    probs = model.predict_proba([text])[0]
    prob = max(probs)

    return {
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": round(float(prob), 4),
    }
