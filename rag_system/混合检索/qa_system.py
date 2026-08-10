# qa_system.py
import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

# 确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 确保可以导入上级目录的公共模块（日志、异常）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_retriever import HybridRetriever
from rerank import Reranker
from 系统日志.config import get_logger
from 异常处理.exceptions import LLMError, ConfigError

load_dotenv()

# 当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据文件路径（位于 ../data/structured_segments.json）
DATA_JSON = os.path.join(BASE_DIR, "..", "data", "structured_segments.json")

logger = get_logger(__name__)

# LLM 上下文窗口预留：给 system prompt + 用户问题 + 答案预留的 token 数
RESERVED_TOKENS = 2000
# 默认模型上下文窗口（qwen-plus 为 32K）
DEFAULT_MAX_CONTEXT_TOKENS = 32000

# 默认大模型配置（可被 create_llm_client 覆盖，便于切换不同大模型）
DEFAULT_LLM_MODEL = "qwen-plus"
DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_API_KEY_ENV = "qwen_api_key"


def create_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: str = DEFAULT_LLM_API_KEY_ENV,
) -> OpenAI:
    """
    创建大模型客户端（模块级函数，便于使用不同的大模型）。

    参数：
        api_key: API 密钥；若为 None，则从环境变量 api_key_env 读取
        base_url: API 端点；若为 None，使用默认的 DashScope 兼容端点
        api_key_env: 读取 API 密钥的环境变量名（默认 qwen_api_key）

    返回：
        配置好的 OpenAI 客户端实例

    示例：
        # 使用默认通义千问
        client = create_llm_client()

        # 使用其他大模型（如 DeepSeek、OpenAI 等）
        client = create_llm_client(
            api_key="sk-xxx",
            base_url="https://api.deepseek.com/v1",
        )
    """
    # 解析 API 密钥：优先使用传入的，否则从环境变量读取
    resolved_key = api_key or os.getenv(api_key_env)
    if not resolved_key:
        raise ConfigError(
            f"未设置 API 密钥（请传入 api_key 或设置环境变量 {api_key_env}）",
            code="MISSING_API_KEY"
        )

    resolved_url = base_url or DEFAULT_LLM_BASE_URL
    logger.info("创建 LLM 客户端，base_url=%s", resolved_url)
    return OpenAI(
        api_key=resolved_key,
        base_url=resolved_url,
    )


def _estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数。
    优先使用 tiktoken（若已安装），否则退化为字符数估算（中文约 1 字符 ≈ 0.6 token）。
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # 中文场景：约 1.5 字符 ≈ 1 token，保守估算
        return int(len(text) / 1.5)


class QASystem:
    """
    完整的检索增强问答系统：
    召回 → 重排序 → 拼接上下文 → LLM 回答
    """
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        llm_model: str = DEFAULT_LLM_MODEL,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        llm_client: Optional[OpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = DEFAULT_LLM_API_KEY_ENV,
    ):
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.llm_model = llm_model
        self.max_context_tokens = max_context_tokens

        # 创建 LLM 客户端：
        # 1. 若显式传入 llm_client，则直接使用（最灵活，可传入任意大模型客户端）
        # 2. 否则通过模块级 create_llm_client 创建（支持不同 api_key / base_url / 环境变量）
        if llm_client is not None:
            self.llm_client = llm_client
            logger.info("使用外部传入的 LLM 客户端，模型=%s", llm_model)
        else:
            self.llm_client = create_llm_client(
                api_key=api_key,
                base_url=base_url,
                api_key_env=api_key_env,
            )


    def answer(
        self,
        query: str,  # 用户问题
        max_candidates: int = 20,
        top_k: int = 5,  # 重排序后保留的文档数
        include_sources: bool = True,  # 是否在答案中包含引用来源
        min_rerank_score: Optional[float] = None,  # 质量阈值，低于此分数的文档不参与拼接
    ) -> Dict:
        # ---- 输入校验 ----
        if not query or not query.strip():
            logger.warning("answer 收到空查询")
            return {"answer": "请输入有效的问题。", "sources": []}

        # ---- 步骤1：召回 ----
        try:
            candidates = self.hybrid.search(query, top_k=max_candidates, method="rrf")
        except Exception as e:
            logger.error("混合检索失败: %s", e, exc_info=True)
            return {"answer": "检索过程中出现错误，请稍后重试。", "sources": []}

        if not candidates:
            logger.info("未找到相关信息，query=%r", query[:50])
            return {"answer": "抱歉，未找到相关信息。", "sources": []}

        # ---- 步骤2：重排序 ----
        try:
            reranked_docs = self.reranker.rerank(
                query, candidates, top_k=top_k, min_score=min_rerank_score
            )
        except Exception as e:
            logger.error("重排序失败: %s", e, exc_info=True)
            return {"answer": "重排序过程中出现错误，请稍后重试。", "sources": []}

        if not reranked_docs:
            logger.info("重排序后无有效文档（可能全部低于质量阈值）")
            return {"answer": "抱歉，未找到足够相关的信息。", "sources": []}

        # ---- 步骤3：拼接上下文（带 token 上限控制）----
        context_parts = []
        sources = []
        total_tokens = 0
        # 预留 token：system prompt + 用户问题 + 答案
        budget = self.max_context_tokens - RESERVED_TOKENS - _estimate_tokens(query)

        for i, doc in enumerate(reranked_docs):
            title_path = doc["metadata"].get("title_path", "无标题")
            part = f"【文档{i+1} 来源：{title_path}】\n{doc['text']}"
            part_tokens = _estimate_tokens(part)

            # 若加入该文档会超出预算，则停止拼接（文档已按 rerank 分数降序，越靠后越不重要）
            if total_tokens + part_tokens > budget:
                logger.warning(
                    "上下文 token 预算不足，已拼接 %d 篇文档（预算 %d tokens）",
                    len(context_parts), budget
                )
                break

            context_parts.append(part)
            total_tokens += part_tokens

            if include_sources:
                sources.append({
                    "title_path": title_path,
                    "text_snippet": doc["text"][:200] + "...",
                    "rerank_score": doc.get("rerank_score", 0.0)
                })

        if not context_parts:
            logger.warning("上下文为空（预算过小或文档过长）")
            return {"answer": "抱歉，检索到的内容过长，无法生成回答。", "sources": []}

        context = "\n\n".join(context_parts)
        logger.info("上下文拼接完成，共 %d 篇文档，约 %d tokens", len(context_parts), total_tokens)

        # ---- 步骤4：LLM 生成 ----
        system_prompt = (
            "你是一个专业的年报分析助手。请根据提供的上下文信息回答用户的问题。\n"
            "要求：\n"
            "1. 回答准确、简洁，基于给出的上下文。\n"
            "2. 如果上下文不足以回答问题，请明确说明。\n"
            "3. 如果使用了上下文中的具体数据，可以注明来源。"
        )
        user_prompt = f"上下文信息：\n{context}\n\n用户问题：{query}\n请回答："

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 低温度保证回答稳定性
                timeout=60,       # 设置超时，避免无限等待
            )
            answer_text = response.choices[0].message.content
        except Exception as e:
            logger.error("LLM 调用失败: %s", e, exc_info=True)
            return {
                "answer": "抱歉，大模型生成回答时出现错误，请稍后重试。",
                "sources": sources,
                "error": str(e),
            }

        # 构建返回结果
        result = {"answer": answer_text}
        if include_sources:
            result["sources"] = sources
        return result


def run_qa_pipeline(
    data_json: str = DATA_JSON,
    interactive: bool = False,
    test_questions: Optional[List[str]] = None,
) -> QASystem:
    """
    完整问答管线：加载数据 → 构建索引 → 创建问答系统 → 回答问题。

    参数：
        data_json: 结构化片段 JSON 路径（默认为 structured_segments.json，
                   也可传入高级切分后的 chunks_recursive.json / chunks_semantic.json）
        interactive: 是否进入交互式问答循环（输入 exit/quit 退出）
        test_questions: interactive=False 时使用的问题列表（默认使用内置测试问题）

    返回：
        QASystem 实例，便于外部继续调用 answer()
    """
    # 导入数据加载和索引构建函数
    from chroma import load_segments, build_chroma_index
    from bm25 import BM25Retriever

    # 1. 加载文档片段
    segments = load_segments(data_json)
    print(f"\n已加载 {len(segments)} 个文本片段（来源: {data_json}）")

    # 2. 构建 Chroma 向量索引
    print("正在构建 Chroma 索引...")
    chroma_coll = build_chroma_index(segments)

    # 3. 构建 BM25 索引
    print("正在构建 BM25 索引...")
    documents = [seg["content"] for seg in segments]
    bm25_retriever = BM25Retriever(documents)

    # 4. 创建混合检索器
    hybrid = HybridRetriever(chroma_coll, bm25_retriever)

    # 5. 初始化重排序器
    print("正在加载重排序模型...")
    reranker = Reranker(backend="bge")  # 可换成 "cohere"

    # 6. 创建完整问答系统
    qa = QASystem(hybrid, reranker)

    # 7. 问答环节
    if interactive:
        _run_interactive_qa(qa)
    else:
        questions = test_questions or [
            "公司2025年的净利润是多少？",
            "有哪些股东持股比例超过5%？",
            "公司面临的主要风险是什么？"
        ]
        for question in questions:
            print(f"\n{'='*60}")
            print(f"问题：{question}")
            result = qa.answer(question)
            print(f"\n答案：\n{result['answer']}")
            if result.get("sources"):
                print("\n参考来源：")
                for i, src in enumerate(result["sources"]):
                    print(f"  {i+1}. {src['title_path']} (重排序分数：{src['rerank_score']:.4f})")
                    print(f"     内容片段：{src['text_snippet']}...")

    return qa


def _run_interactive_qa(qa: QASystem) -> None:
    """交互式问答循环，输入 exit / quit / 退出 结束。"""
    print("\n" + "=" * 60)
    print("问答系统已就绪，输入问题开始提问（输入 exit / quit / 退出 结束）")
    print("=" * 60)
    while True:
        try:
            query = input("\n问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出问答。")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q", "退出"):
            print("退出问答。")
            break
        result = qa.answer(query)
        print(f"\n答案：\n{result['answer']}")
        if result.get("sources"):
            print("\n参考来源：")
            for i, src in enumerate(result["sources"]):
                print(f"  {i+1}. {src['title_path']} (重排序分数：{src['rerank_score']:.4f})")
                print(f"     内容片段：{src['text_snippet']}...")


# --------------------------- 主程序 ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="年报智能问答系统")
    parser.add_argument(
        "--data-json",
        default=DATA_JSON,
        help=f"结构化片段 JSON 路径（默认: {DATA_JSON}，可传 chunks_recursive.json 等）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入交互式问答循环（默认运行内置测试问题）",
    )
    args = parser.parse_args()

    run_qa_pipeline(data_json=args.data_json, interactive=args.interactive)
