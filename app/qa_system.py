# qa_system.py
import os
from typing import List, Dict, Optional, AsyncGenerator
from openai import OpenAI

from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.rerank import Reranker
from rag_system.retrieval.bm25 import BM25Retriever
from rag_system.common.logging_config import get_logger
from rag_system.common.exceptions import LLMError, ConfigError

from app.config import settings
from app.conversation import (
    estimate_tokens as _estimate_tokens,
    history_tokens,
    rewrite_query,
    trim_history,
)

logger = get_logger(__name__)

# 数据文件路径（统一由配置中心管理）
DATA_JSON = settings.DATA_JSON

# LLM 上下文窗口预留：给 system prompt + 用户问题 + 答案预留的 token 数
RESERVED_TOKENS = settings.RESERVED_TOKENS
# 默认模型上下文窗口（deepseek-chat 为 64K）
DEFAULT_MAX_CONTEXT_TOKENS = settings.MAX_CONTEXT_TOKENS

# 默认大模型配置（可被 create_llm_client 覆盖，便于切换不同大模型）
DEFAULT_LLM_MODEL = settings.LLM_MODEL
DEFAULT_LLM_BASE_URL = settings.LLM_BASE_URL
DEFAULT_LLM_API_KEY_ENV = settings.LLM_API_KEY_ENV


def create_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: str = DEFAULT_LLM_API_KEY_ENV,
) -> OpenAI:
    """
    创建大模型客户端（模块级函数，便于使用不同的大模型）。

    参数：
        api_key: API 密钥；若为 None，则从环境变量 api_key_env 读取
        base_url: API 端点；若为 None，使用默认的 DeepSeek 端点
        api_key_env: 读取 API 密钥的环境变量名（默认 deepseek_api_key）

    返回：
        配置好的 OpenAI 客户端实例

    示例：
        # 使用默认 DeepSeek
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


# 支持的检索模式
VALID_MODES = ("full", "vector_only", "bm25_only", "hybrid_no_rerank", "vector_rerank")
# 需要重排序的模式
RERANK_MODES = ("full", "vector_rerank")


class QASystem:
    """
    完整的检索增强问答系统：
    召回 → 重排序（可选）→ 拼接上下文 → LLM 回答

    支持多种检索模式（retrieval_mode）：
        - "full"：混合检索（BM25+向量+RRF融合）+ BGE 重排序（完整管线）
        - "vector_only"：仅向量检索，无重排序（基础 RAG）
        - "bm25_only"：仅 BM25 关键词检索，无重排序
        - "hybrid_no_rerank"：混合检索，无重排序
        - "vector_rerank"：仅向量检索 + BGE 重排序
    """
    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        llm_client: Optional[OpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = DEFAULT_LLM_API_KEY_ENV,
        retrieval_mode: str = "full",    #选择模式
        chroma_collection=None,
        bm25_retriever: Optional[BM25Retriever] = None,
        segments: Optional[List[Dict]] = None,
    ):
        if retrieval_mode not in VALID_MODES:
            raise ConfigError(
                f"不支持的检索模式: {retrieval_mode}（可选: {', '.join(VALID_MODES)}）",
                code="INVALID_MODE"
            )

        self.retrieval_mode = retrieval_mode
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.chroma_collection = chroma_collection
        self.bm25_retriever = bm25_retriever
        self.segments = segments
        self.llm_model = llm_model
        self.max_context_tokens = max_context_tokens
        # 最近一次流式问答的引用来源、上下文与查询改写结果（供前端展示/调试）
        self.last_sources = []
        self.last_retrieved_contexts = []
        self.last_rewritten_query = ""


        # 校验模式与组件的匹配关系
        needs_hybrid = retrieval_mode in ("full", "hybrid_no_rerank")
        needs_rerank = retrieval_mode in RERANK_MODES
        needs_chroma = retrieval_mode in ("vector_only", "vector_rerank")
        needs_bm25 = retrieval_mode == "bm25_only"

        if needs_hybrid and hybrid_retriever is None:
            raise ConfigError(f"模式 {retrieval_mode} 需要 hybrid_retriever", code="MISSING_COMPONENT")
        if needs_rerank and reranker is None:
            raise ConfigError(f"模式 {retrieval_mode} 需要 reranker", code="MISSING_COMPONENT")
        if needs_chroma and chroma_collection is None:
            raise ConfigError(f"模式 {retrieval_mode} 需要 chroma_collection", code="MISSING_COMPONENT")
        if needs_bm25 and bm25_retriever is None:
            raise ConfigError(f"模式 {retrieval_mode} 需要 bm25_retriever", code="MISSING_COMPONENT")
        # bm25_only 模式需要 segments 来获取 metadata
        if needs_bm25 and segments is None:
            raise ConfigError(f"模式 {retrieval_mode} 需要 segments（用于获取标题路径）", code="MISSING_COMPONENT")

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


    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """
        仅使用向量检索（Chroma），返回标准化结果列表。
        用于 vector_only 和 vector_rerank 模式。
        """
        try:
            raw = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_k
            )
        except Exception as e:
            logger.error("Chroma 查询失败: %s", e, exc_info=True)
            return []

        results = []
        try:
            ids = raw.get("ids", [[]])[0]
            distances = raw.get("distances", [[]])[0]
            documents = raw.get("documents", [[]])[0]
            metadatas = raw.get("metadatas", [None])[0]
        except (IndexError, KeyError, TypeError) as e:
            logger.error("Chroma 返回结果格式异常: %s", e, exc_info=True)
            return []

        if not ids:
            return []

        for i in range(len(ids)):
            dist = distances[i] if i < len(distances) else 1.0
            sim = 1.0 - dist / 2.0  # 余弦距离转相似度
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            results.append({
                "id": ids[i],
                "score": sim,
                "text": documents[i] if i < len(documents) else "",
                "metadata": meta if isinstance(meta, dict) else {}
            })

        logger.info("向量检索 query=%r 返回 %d 条结果", query[:50], len(results))
        return results

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """
        仅使用 BM25 检索，返回标准化结果列表（含 metadata）。
        用于 bm25_only 模式。
        """
        raw_results = self.bm25_retriever.search(query, top_k=top_k)

        results = []
        for res in raw_results:
            idx = res.get("doc_index", 0)
            # 从 segments 中获取标题路径
            if self.segments and idx < len(self.segments):
                title_path = self.segments[idx].get("title_path", [])
                metadata = {
                    "title_path": " > ".join(title_path) if title_path else "无标题"
                }
            else:
                metadata = {"title_path": "无标题"}

            results.append({
                "id": f"bm25_{idx}",
                "score": res["score"],
                "text": res["text"],
                "metadata": metadata
            })

        logger.info("BM25 检索 query=%r 返回 %d 条结果", query[:50], len(results))
        return results


    def answer(
        self,
        query: str,  # 用户问题
        max_candidates: int = 20,
        top_k: int = 5,  # 重排序后保留的文档数
        include_sources: bool = True,  # 是否在答案中包含引用来源
        min_rerank_score: Optional[float] = 0.5,  # 质量阈值，低于此分数的文档不参与拼接
        history: Optional[List[Dict]] = None,  # 多轮对话历史（OpenAI messages 格式）
    ) -> Dict:
        # ---- 输入校验 ----
        if not query or not query.strip():
            logger.warning("answer 收到空查询")
            return {"answer": "请输入有效的问题。", "sources": []}

        # ---- 步骤0：多轮记忆（裁剪历史 + 查询改写，改写结果仅用于检索）----
        history_msgs = trim_history(history) if history else []
        search_query = rewrite_query(self.llm_client, self.llm_model, query, history_msgs)

        # ---- 步骤1：召回（按检索模式分支，使用改写后的查询）----
        try:
            if self.retrieval_mode in ("full", "hybrid_no_rerank"):
                candidates = self.hybrid.search(search_query, top_k=max_candidates, method="rrf")
            elif self.retrieval_mode in ("vector_only", "vector_rerank"):
                candidates = self._vector_search(search_query, top_k=max_candidates)
            elif self.retrieval_mode == "bm25_only":
                candidates = self._bm25_search(search_query, top_k=max_candidates)
        except Exception as e:
            logger.error("检索失败 [%s]: %s", self.retrieval_mode, e, exc_info=True)
            return {"answer": "检索过程中出现错误，请稍后重试。", "sources": []}

        if not candidates:
            logger.info("未找到相关信息，query=%r", query[:50])
            return {"answer": "抱歉，未找到相关信息。", "sources": []}

        # ---- 步骤2：重排序（仅 full 和 vector_rerank 模式执行）----
        if self.retrieval_mode in RERANK_MODES:
            try:
                reranked_docs = self.reranker.rerank(
                    search_query, candidates, top_k=top_k, min_score=min_rerank_score
                )
            except Exception as e:
                logger.error("重排序失败: %s", e, exc_info=True)
                return {"answer": "重排序过程中出现错误，请稍后重试。", "sources": []}
        else:
            # 无需重排序，直接取 top_k 候选
            reranked_docs = candidates[:top_k]
            logger.info("跳过重排序 [%s]，直接取前 %d 条候选", self.retrieval_mode, len(reranked_docs))

        if not reranked_docs:
            logger.info("重排序后无有效文档（可能全部低于质量阈值）")
            return {"answer": "抱歉，未找到足够相关的信息。", "sources": []}

        # ---- 步骤3：拼接上下文（带 token 上限控制）----
        context_parts = []
        sources = []
        retrieved_contexts = []  # 完整文本列表，供 RAGAS 评估使用
        total_tokens = 0
        # 预留 token：system prompt + 用户问题 + 答案 + 注入的对话历史
        budget = self.max_context_tokens - RESERVED_TOKENS - _estimate_tokens(query) - history_tokens(history_msgs)

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
            retrieved_contexts.append(doc["text"])
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

        # ---- 步骤4：LLM 生成（注入对话历史）----
        system_prompt = (
            "你是一个专业的年报分析助手。请根据提供的上下文信息回答用户的问题。\n"
            "要求：\n"
            "1. 回答准确、简洁，基于给出的上下文。\n"
            "2. 如果上下文不足以回答问题，请明确说明。\n"
            "3. 如果使用了上下文中的具体数据，可以注明来源。\n"
            "4. 这是多轮对话：若问题中出现指代或省略（如「该公司」「上面提到的」），"
            "请结合历史对话理解。"
        )
        # 生成仍使用原始问题：历史消息已注入，模型可自行消解指代
        user_prompt = f"上下文信息：\n{context}\n\n用户问题：{query}\n请回答："

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history_msgs,
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
                "retrieved_contexts": retrieved_contexts,
                "error": str(e),
            }

        # 构建返回结果
        result = {
            "answer": answer_text,
            "retrieved_contexts": retrieved_contexts,
            "total_tokens": total_tokens,
            "rewritten_query": search_query,
        }
        if include_sources:
            result["sources"] = sources
        return result
    async def stream_answer(
        self,
        query: str,
        top_k: int = 5,
        include_sources: bool = True,
        min_rerank_score: Optional[float] = 0.5,  # 质量阈值，与 answer() 保持一致
        history: Optional[List[Dict]] = None,  # 多轮对话历史（OpenAI messages 格式）
    ) -> AsyncGenerator[str, None]:
        """
        流式问答：多轮记忆（裁剪 + 查询改写）+ 检索 + 重排序 + 拼接上下文 + 流式 LLM 生成。
        每次 yield 一段增量文本（通常是 token 或小片段）。

        history 传入最近几轮 user/assistant 消息：检索用改写后的完整问题，
        生成时历史作为 messages 注入 LLM。

        流式结束后，本次检索的引用来源与上下文保存在
        self.last_sources / self.last_retrieved_contexts（供前端展示引用）。
        """
        # ---- 输入校验 ----
        if not query or not query.strip():
            yield "请输入有效的问题。"
            return

        # ---- 步骤0：多轮记忆（裁剪历史 + 查询改写，改写结果仅用于检索）----
        history_msgs = trim_history(history) if history else []
        search_query = rewrite_query(self.llm_client, self.llm_model, query, history_msgs)
        self.last_rewritten_query = search_query

        # ---- 步骤1：召回（使用改写后的查询）----
        try:
            if self.retrieval_mode in ("full", "hybrid_no_rerank"):
                candidates = self.hybrid.search(search_query, top_k=20, method="rrf")
            elif self.retrieval_mode in ("vector_only", "vector_rerank"):
                candidates = self._vector_search(search_query, top_k=20)
            elif self.retrieval_mode == "bm25_only":
                candidates = self._bm25_search(search_query, top_k=20)
        except Exception as e:
            logger.error("检索失败: %s", e, exc_info=True)
            yield "检索过程中出现错误，请稍后重试。"
            return

        if not candidates:
            yield "抱歉，未找到相关信息。"
            return

        # ---- 步骤2：重排序 ----
        if self.retrieval_mode in RERANK_MODES:
            try:
                reranked_docs = self.reranker.rerank(
                    search_query, candidates, top_k=top_k, min_score=min_rerank_score
                )
            except Exception as e:
                logger.error("重排序失败: %s", e, exc_info=True)
                yield "重排序过程中出现错误，请稍后重试。"
                return
        else:
            reranked_docs = candidates[:top_k]

        if not reranked_docs:
            yield "抱歉，未找到足够相关的信息。"
            return

        # ---- 步骤3：拼接上下文（带 token 上限控制）----
        context_parts = []
        sources = []
        retrieved_contexts = []
        total_tokens = 0
        # 预留 token：system prompt + 用户问题 + 答案 + 注入的对话历史
        budget = self.max_context_tokens - RESERVED_TOKENS - _estimate_tokens(query) - history_tokens(history_msgs)

        for i, doc in enumerate(reranked_docs):
            title_path = doc["metadata"].get("title_path", "无标题")
            part = f"【文档{i+1} 来源：{title_path}】\n{doc['text']}"
            part_tokens = _estimate_tokens(part)

            # 如果加入当前文档会超出预算，则停止拼接
            if total_tokens + part_tokens > budget:
                logger.warning(
                    "流式上下文 token 预算不足，已拼接 %d 篇文档（预算 %d tokens）",
                    len(context_parts), budget
                )
                break

            context_parts.append(part)
            retrieved_contexts.append(doc["text"])
            total_tokens += part_tokens

            if include_sources:
                sources.append({
                    "title_path": title_path,
                    "text_snippet": doc["text"][:200] + "...",
                    "rerank_score": doc.get("rerank_score", 0.0),
                })

        if not context_parts:
            logger.warning("流式上下文为空（预算过小或文档过长）")
            yield "抱歉，检索到的内容过长，无法生成回答。"
            return

        # 保存本次检索的引用来源与上下文，供前端流式结束后展示
        self.last_sources = sources
        self.last_retrieved_contexts = retrieved_contexts

        context = "\n\n".join(context_parts)
        logger.info("流式上下文拼接完成，共 %d 篇文档，约 %d tokens", len(context_parts), total_tokens)

        # ---- 步骤4：流式调用 LLM（注入对话历史）----
        system_prompt = (
            "你是一个专业的年报分析助手。请根据提供的上下文信息回答用户的问题。\n"
            "要求：\n"
            "1. 回答准确、简洁，基于给出的上下文。\n"
            "2. 如果上下文不足以回答问题，请明确说明。\n"
            "3. 如果使用了上下文中的具体数据，可以注明来源。\n"
            "4. 这是多轮对话：若问题中出现指代或省略（如「该公司」「上面提到的」），"
            "请结合历史对话理解。"
        )
        # 生成仍使用原始问题：历史消息已注入，模型可自行消解指代
        user_prompt = f"上下文信息：\n{context}\n\n用户问题：{query}\n请回答："

        try:
            stream = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history_msgs,
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                stream=True,
                timeout=60,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("LLM 流式调用失败: %s", e, exc_info=True)
            yield "抱歉，大模型生成回答时出现错误，请稍后重试。"

def build_qa_system_from_segments(
    segments: List[Dict],
    llm_model: Optional[str] = None,
) -> QASystem:
    """
    从文档片段列表直接构建 QA 系统（无需先写 JSON 文件）。
    供文件上传等场景复用：解析后的片段可直接入库。

    参数：
        segments: 结构化片段列表，每个含 content / title_path 等字段
        llm_model: 覆盖默认大模型名（可选）

    返回：
        构建好的 QASystem 实例（full 模式：混合检索 + 重排序）
    """
    from rag_system.retrieval.chroma import build_chroma_index

    if not segments:
        raise ConfigError("文档片段为空，无法构建问答系统", code="EMPTY_SEGMENTS")

    # 1. 构建 Chroma 向量索引
    logger.info("正在构建 Chroma 索引（%d 个片段）...", len(segments))
    chroma_coll = build_chroma_index(segments)

    # 2. 构建 BM25 索引
    documents = [seg["content"] for seg in segments]
    bm25_retriever = BM25Retriever(documents)

    # 3. 创建混合检索器
    hybrid = HybridRetriever(chroma_coll, bm25_retriever)

    # 4. 初始化重排序器
    logger.info("正在加载重排序模型...")
    reranker = Reranker(backend=settings.RERANK_BACKEND)

    # 5. 创建完整问答系统（full 模式）
    qa = QASystem(hybrid, reranker, llm_model=llm_model or DEFAULT_LLM_MODEL)
    return qa


def initialize_qa_system(data_json: str = DATA_JSON) -> QASystem:
    """
    从结构化片段 JSON 文件构建并初始化 QA 系统，不执行任何问答。
    供 API 服务或外部程序调用。
    """
    from rag_system.retrieval.chroma import load_segments

    segments = load_segments(data_json)
    logger.info("已加载 %d 个文本片段（来源: %s）", len(segments), data_json)
    return build_qa_system_from_segments(segments)

def run_qa_pipeline(
    data_json: str = DATA_JSON,
    interactive: bool = False,
    test_questions: Optional[List[str]] = None,
) -> QASystem:
    """
    完整问答管线：构建系统 → 回答问题（测试或交互）。
    供开发/命令行使用，不建议在 API 服务中直接调用。
    """
    qa = initialize_qa_system(data_json)

    # 问答环节
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
