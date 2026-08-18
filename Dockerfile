# ============================================================
# 年报智能问答 RAG 系统 —— 容器镜像
#
# 构建：docker build -t rag-workspace .
# 运行：
#   docker run -p 8000:8000 \
#     -e qwen_api_key=你的密钥 \
#     -v rag_data:/workspace/rag_system/data \
#     -v rag_chroma:/workspace/rag_system/chroma_db \
#     rag-workspace
# ============================================================
FROM python:3.10-slim

WORKDIR /workspace

# 1. 先复制依赖清单并安装（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 复制源码
COPY app/ app/
COPY rag_system/ rag_system/
COPY scripts/ scripts/
COPY pyproject.toml .

# 3. 端口与启动命令
EXPOSE 8000
# 注意：API 密钥通过运行时环境变量注入（qwen_api_key），不要打进镜像
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
