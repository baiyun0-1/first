# 使用轻量级的基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 防止 Python 写入 .pyc 文件，并确保日志实时输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 先只拷贝依赖文件，利用 Docker 缓存层（只要依赖没变，就不重新安装）
COPY requirements.txt .

# 安装依赖，--no-cache-dir 可以大幅减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝应用代码（严格按照大纲：不 COPY . .）
COPY app/ ./app/
COPY ml/ ./ml/

# 暴露端口
EXPOSE 8000

# 启动 FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]