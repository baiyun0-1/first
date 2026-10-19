# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# 关闭 .pyc 写入、日志实时输出；模型路径显式指定，避免依赖启动目录
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model_v1.pkl

# 先只拷贝依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 只拷贝推理真正需要的东西：
# app/ 是服务代码，models/ 是训练产物（model_v1.pkl 已入库，clone 即可构建）
COPY app/ ./app/
COPY models/ ./models/

# 以非 root 用户运行
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
