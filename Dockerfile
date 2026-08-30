# ============================================================
# 年报智能问答 RAG 系统 —— 容器镜像
#
# 构建：docker build -t rag-workspace .
# 运行（单容器，等价于 docker compose 中 rag-api 服务）：
#   docker run -p 8000:8000 \
#     -e deepseek_api_key=你的密钥 \
#     -e HF_ENDPOINT=https://hf-mirror.com \
#     -v rag_data:/workspace/rag_system/data \
#     -v rag_chroma:/workspace/rag_system/chroma_db \
#     -v rag_db:/workspace/data \
#     -v hf_cache:/root/.cache/huggingface \
#     rag-workspace
#
# 多服务编排（FastAPI + Streamlit）：见 docker-compose.yml
# ============================================================
FROM python:3.10-slim

WORKDIR /workspace

# 基础环境：无缓冲输出、时区、UTF-8
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# curl 用于 /health 健康检查；tzdata 用于时区
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# HuggingFace 国内镜像端点：阿里云 ECS 无法直连 huggingface.co，
# 首次启动会自动下载 BGE 嵌入/重排模型，默认走 hf-mirror 镜像站。
# 可通过 --build-arg HF_ENDPOINT=... 覆盖。
ARG HF_ENDPOINT=https://hf-mirror.com
ENV HF_ENDPOINT=${HF_ENDPOINT}

# 1. 先复制依赖清单并安装（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com

# 2. 复制源码
COPY app/ app/
COPY rag_system/ rag_system/
COPY scripts/ scripts/
COPY pyproject.toml .

# 3. 端口与启动命令
EXPOSE 8000 8501

# 健康检查：探测 FastAPI /health。
# start-period 需覆盖首次启动（下载模型 + 重建 Chroma 索引），故设为 300s。
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 注意：API 密钥通过运行时环境变量注入（deepseek_api_key），不要打进镜像。
# 说明：默认单 worker —— 本应用每个 worker 都会独立初始化 RAG（加载 BGE 模型、
#       重建 Chroma 索引），多 worker 会带来内存翻倍并引发 ChromaDB/SQLite
#       并发写锁冲突；如需横向扩展，建议先把 Chroma 迁移为 Server 模式、
#       SQLite 迁移为 PostgreSQL（RDS）。
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
