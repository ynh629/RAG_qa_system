# qa_system.py
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from hybrid_retriever import HybridRetriever
from rerank import Reranker
load_dotenv()
llm_client = OpenAI(
    api_key=os.getenv("qwen_api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
class QASystem:
    """
    完整的检索增强问答系统：
    召回 → 重排序 → 拼接上下文 → LLM 回答
    """
    def __init__(self, hybrid_retriever: HybridRetriever, reranker: Reranker, llm_model: str = "qwen-plus"):
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.llm_model = llm_model

    def answer(
        self,
        query: str,  #用户问题
        max_candidates: int = 20,
        top_k: int = 5,  #重排序后保留的文档数
        include_sources: bool = True  #是否在答案中包含引用来源
    ) -> Dict:
        # ---- 步骤1：召回 ----
        # 变量：混合检索返回的候选文档列表
        candidates = self.hybrid.search(query, top_k=max_candidates, method="rrf")
        if not candidates:
            return {"answer": "抱歉，未找到相关信息。", "sources": []}

        # ---- 步骤2：重排序 ----
        # 变量：重排序后的文档列表（默认会附加 rerank_score 和 original_score）
        reranked_docs = self.reranker.rerank(query, candidates, top_k=top_k)

        # ---- 步骤3：拼接上下文 ----
        context_parts = []
        # 变量：来源信息列表（用于返回给用户）
        sources = []
        for i, doc in enumerate(reranked_docs):
            # 获取标题路径，若无则用“无标题”
            title_path = doc["metadata"].get("title_path", "无标题")
            # 格式化：编号 + 标题路径 + 内容
            context_parts.append(f"【文档{i+1} 来源：{title_path}】\n{doc['text']}")
            if include_sources:
                sources.append({
                    "title_path": title_path,
                    "text_snippet": doc["text"][:200] + "...",
                    "rerank_score": doc.get("rerank_score", 0.0)
                })
        # 变量：用双换行拼接所有上下文片段
        context = "\n\n".join(context_parts)

        # ---- 步骤4：LLM 生成 ----
        # 变量：system prompt，要求模型基于上下文回答
        system_prompt = (
            "你是一个专业的年报分析助手。请根据提供的上下文信息回答用户的问题。\n"
            "要求：\n"
            "1. 回答准确、简洁，基于给出的上下文。\n"
            "2. 如果上下文不足以回答问题，请明确说明。\n"
            "3. 如果使用了上下文中的具体数据，可以注明来源。"
        )
        # 变量：user prompt，包含上下文和用户问题
        user_prompt = f"上下文信息：\n{context}\n\n用户问题：{query}\n请回答："

        # 调用大模型
        response = llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # 低温度保证回答稳定性
        )
        # 变量：模型生成的最终答案
        answer_text = response.choices[0].message.content

        # 构建返回结果
        result = {"answer": answer_text}
        if include_sources:
            result["sources"] = sources
        return result

# --------------------------- 主程序 ---------------------------
if __name__ == "__main__":
    # 导入数据加载和索引构建函数（可在之前模块中找到）
    from chroma import load_segments, build_chroma_index
    from bm25 import BM25Retriever

    # 1. 加载文档片段
    segments = load_segments("structured_segments.json")

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

    # 7. 测试提问
    test_questions = [
        "公司2025年的净利润是多少？",
        "有哪些股东持股比例超过5%？",
        "公司面临的主要风险是什么？"
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        result = qa.answer(question)
        print(f"\n答案：\n{result['answer']}")
        if result.get("sources"):
            print("\n参考来源：")
            for i, src in enumerate(result["sources"]):
                print(f"  {i+1}. {src['title_path']} (重排序分数：{src['rerank_score']:.4f})")
                print(f"     内容片段：{src['text_snippet']}...")