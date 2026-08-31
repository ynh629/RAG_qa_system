# config.py
"""统一配置中心。

所有路径、模型名、API 参数、服务端口等集中在此管理，
可通过仓库根目录 .env 中的同名环境变量覆盖默认值。

用法：from app.config import settings
"""
import os

from dotenv import load_dotenv

# 仓库根目录（app/ 的上一级）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 显式加载仓库根目录 .env，摆脱对当前工作目录的依赖
load_dotenv(os.path.join(ROOT_DIR, ".env"))


class Settings:
    # ===================== 路径 =====================
    ROOT_DIR: str = ROOT_DIR
    # RAG 引擎数据目录（JSON、PDF、评估数据集等）
    DATA_DIR: str = os.path.join(ROOT_DIR, "rag_system", "data")
    # Chroma 向量库持久化目录
    CHROMA_DIR: str = os.path.join(ROOT_DIR, "rag_system", "chroma_db")
    # 日志目录
    LOG_DIR: str = os.path.join(ROOT_DIR, "rag_system", "logs")
    # 示例年报 PDF
    PDF_PATH: str = os.path.join(DATA_DIR, "年报.pdf")
    # 结构化片段 / 高级切分结果
    STRUCTURED_JSON: str = os.path.join(DATA_DIR, "structured_segments.json")
    # 别名：检索/问答模块使用的默认数据文件
    DATA_JSON: str = STRUCTURED_JSON
    RECURSIVE_JSON: str = os.path.join(DATA_DIR, "chunks_recursive.json")
    SEMANTIC_JSON: str = os.path.join(DATA_DIR, "chunks_semantic.json")
    # 评估数据集
    EVAL_DATASET_PATH: str = os.path.join(DATA_DIR, "eval_dataset.json")
    # SQLite 数据库（对话历史 / 反馈）
    DB_PATH: str = os.path.join(ROOT_DIR, "data", "chat_app.db")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{DB_PATH.replace(os.sep, '/')}",
    )

    # ===================== LLM =====================
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
    LLM_BASE_URL: str = os.getenv(
        "LLM_BASE_URL", "https://api.deepseek.com/v1"
    )
    LLM_API_KEY_ENV: str = os.getenv("LLM_API_KEY_ENV", "deepseek_api_key")
    # 上下文预算：给 system prompt + 问题 + 答案预留的 token 数
    RESERVED_TOKENS: int = int(os.getenv("RESERVED_TOKENS", "2000"))
    # 模型上下文窗口（deepseek-chat 为 64K）
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "64000"))

    # ===================== 检索 / 重排序 =====================
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    RERANK_BACKEND: str = os.getenv("RERANK_BACKEND", "bge")

    # ===================== 多轮对话记忆 =====================
    # 历史轮数上限（1 轮 = 一问一答），超出只保留最近的
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "5"))
    # 历史注入 LLM 的 token 预算，超出时从最旧开始丢弃
    HISTORY_TOKEN_BUDGET: int = int(os.getenv("HISTORY_TOKEN_BUDGET", "1500"))

    # ===================== API 服务 =====================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
