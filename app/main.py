import sys
# 强制标准输出使用 utf-8，防止 Windows 终端中文乱码
sys.stdout.reconfigure(encoding='utf-8')

import os
import joblib
# ... 下面是原来的代码 ...
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="IMDb Sentiment API")

# 默认路径，但允许通过环境变量 MODEL_PATH 覆盖
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_v1.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[OK] 成功加载模型: {MODEL_PATH}")
    else:
        print(f"[WARN] 找不到模型: {MODEL_PATH}，请先运行训练脚本。")

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
    text = request.text[:1000] # 截断保护
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0].max()
    
    return {
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": round(float(prob), 4)
    }