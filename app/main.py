import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 默认路径，可通过环境变量覆盖（Dockerfile 中设置为 /app/models/model_v1.pkl）
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_v1.pkl")
# 与训练脚本保持一致的截断长度
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "1000"))

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型。

    模型缺失时不让进程崩溃：容器仍能通过 /health 探活，
    /predict 返回 503，便于在编排系统里看出是「模型没挂上」而不是「服务挂了」。
    """
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[OK] 成功加载模型: {MODEL_PATH}")
    else:
        print(f"[WARN] 找不到模型: {MODEL_PATH}，/predict 将返回 503")
    yield


app = FastAPI(title="IMDb Sentiment API", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.text[:MAX_TEXT_LENGTH]
    pred = int(model.predict([text])[0])
    probs = model.predict_proba([text])[0]
    confidence = float(max(probs))

    return {
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": round(confidence, 4),
    }
