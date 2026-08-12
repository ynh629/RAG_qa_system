# ragas_config.py
"""
RAGAS 评估配置模块。
封装 RAGAS 所需的 LLM 和 Embedding 对象，复用现有系统的通义千问和 BGE 模型。
"""
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 确保可以导入上级目录的公共模块
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from 系统日志.config import get_logger
from 异常处理.exceptions import ConfigError

# .env 文件在上级 python/ 目录
_env_path = os.path.join(os.path.dirname(BASE_DIR), ".env")
load_dotenv(_env_path)

logger = get_logger(__name__)

# 复用 qa_system.py 中的默认配置
DEFAULT_LLM_MODEL = "qwen3.7-plus"
DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_API_KEY_ENV = "qwen_api_key"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def get_eval_llm(
    api_key: str = None,
    base_url: str = None,
    model: str = DEFAULT_LLM_MODEL,
    api_key_env: str = DEFAULT_LLM_API_KEY_ENV,
) -> LangchainLLMWrapper:
    """
    创建 RAGAS 评估用的 LLM 封装。
    复用通义千问 DashScope OpenAI 兼容端点，temperature=0 保证评估稳定性。

    返回：
        LangchainLLMWrapper 实例
    """
    resolved_key = api_key or os.getenv(api_key_env)
    if not resolved_key:
        raise ConfigError(
            f"未设置 API 密钥（请传入 api_key 或设置环境变量 {api_key_env}）",
            code="MISSING_API_KEY"
        )
    resolved_url = base_url or DEFAULT_LLM_BASE_URL

    logger.info("创建 RAGAS 评估 LLM，model=%s, base_url=%s", model, resolved_url)
    chat_model = ChatOpenAI(
        model=model,
        api_key=resolved_key,
        base_url=resolved_url,
        temperature=0,
        timeout=60,
    )
    return LangchainLLMWrapper(chat_model)


def get_eval_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> LangchainEmbeddingsWrapper:
    """
    创建 RAGAS 评估用的 Embedding 封装。
    复用系统中的 BGE 中文嵌入模型。

    返回：
        LangchainEmbeddingsWrapper 实例
    """
    logger.info("创建 RAGAS 评估 Embedding，model=%s", model_name)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return LangchainEmbeddingsWrapper(embeddings)
