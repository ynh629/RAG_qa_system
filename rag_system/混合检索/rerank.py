# rerank_pipeline.py
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import os

# 当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据文件路径（位于 ../data/structured_segments.json）
DATA_JSON = os.path.join(BASE_DIR, "..", "data", "structured_segments.json")


class Reranker:
    """
        - 'bge': 使用 BAAI/bge-reranker-base 本地模型
        - 'cohere': 使用 Cohere Rerank API（需设置环境变量 COHERE_API_KEY）
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", backend: str = "bge"):
        """
        初始化重排序器。
        参数：
            model_name: 模型名称或 API 端点
            backend: 'bge' 或 'cohere'
        """
        self.backend = backend
        if backend == "bge":
            # 加载 CrossEncoder 模型（专门用于文本对评分）
            # 变量：本地 BGE Reranker 模型
            self.model = CrossEncoder(model_name)
        elif backend == "cohere":
            # 导入 Cohere 客户端
            import cohere
            # 从环境变量读取 API 密钥
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("请设置环境变量 COHERE_API_KEY")
            # 变量：Cohere 客户端实例
            self.co = cohere.Client(api_key)
            self.model_name = model_name  # Cohere 的模型名，如 "rerank-english-v3.0"
        else:
            raise ValueError(f"不支持的 backend: {backend}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        if not candidates:
            return []
        documents = [c["text"] for c in candidates]
        if self.backend == "bge":
            # 构造 (query, doc) 对，CrossEncoder 输入格式
            # 变量：输入对列表
            pairs = [[query, doc] for doc in documents]
            # 预测相关性分数（logits），可直接视为相似度
            # 变量：分数数组，shape (len(documents),)
            scores = self.model.predict(pairs)
            # 转为 Python float 列表
            scores = [float(s) for s in scores]
        elif self.backend == "cohere":
            # 调用 Cohere Rerank API
            # 变量：API 响应对象
            response = self.co.rerank(
                query=query,
                documents=documents,
                model=self.model_name,
                top_n=top_k if top_k else len(documents)
            )
            # Cohere 返回的是按分数排序的列表，但我们需要保持与本地模型一致的格式
            # 先构建 id->score 映射
            # 变量：文档索引与重排序分数的映射
            rerank_map = {r.index: r.relevance_score for r in response.results}
            # 按原始 candidates 顺序赋分，无分的置为 0
            scores = [rerank_map.get(i, 0.0) for i in range(len(documents))]
        else:
            raise ValueError(f"不支持的 backend: {self.backend}")

        # 将分数附加到候选文档中，并记入原始分数和 rerank 分数
        # 变量：增强后的文档列表
        enhanced_candidates = []
        for i, candidate in enumerate(candidates):
            new_candidate = candidate.copy()
            new_candidate["rerank_score"] = scores[i]
            # 保留原始混合分数以便对比
            new_candidate["original_score"] = candidate.get("score", 0.0)
            enhanced_candidates.append(new_candidate)

        # 按 rerank_score 降序排序
        enhanced_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        # 截取 top_k（如果指定）
        if top_k:
            enhanced_candidates = enhanced_candidates[:top_k]

        return enhanced_candidates

# --------------------------- 主程序（测试重排序管线）---------------------------
if __name__ == "__main__":
    # 导入已有模块（假设已存在 hybrid_search.py）
    import sys
    sys.path.append('.')
    try:
        from hybrid_retriever import HybridRetriever, load_segments, build_chroma_index
        from bm25 import BM25Retriever
        from chroma import load_segments as chroma_load_segments
    except ImportError as e:
        print(f"导入错误，请确保所有模块在相同目录下: {e}")
        sys.exit(1)

    # 1. 加载数据
    segments = chroma_load_segments(DATA_JSON)


    # 2. 构建 Chroma 索引（若已有 chroma_db 可跳过）
    chroma_coll = build_chroma_index(segments)

    # 3. 构建 BM25 索引
    documents = [seg["content"] for seg in segments]
    bm25_retriever = BM25Retriever(documents)

    # 4. 创建混合检索器
    hybrid = HybridRetriever(chroma_coll, bm25_retriever)

    # 5. 初始化 Reranker（这里用本地 BGE Reranker）
    print("正在加载 BGE Reranker 模型...")
    reranker = Reranker(backend="bge")

    # 6. 测试查询
    test_queries = [
        "公司2025年的净利润是多少？",
        "有哪些股东持有5%以上的股份？",
        "公司面临的主要风险有哪些？"
    ]

    for query in test_queries:
        print(f"\n查询：{query}")
        # 6.1 混合检索获取候选集（取 Top10 作为候选）
        candidates = hybrid.search(query, top_k=6, method="rrf")
        print(f"  混合检索返回 {len(candidates)} 个候选")

        # 6.2 重排序
        reranked = reranker.rerank(query, candidates, top_k=3)

        print("  重排序后的 Top3 结果：")
        for i, doc in enumerate(reranked):
            print(f"    {i+1}. 原始混合分数={doc['original_score']:.4f}, Rerank分数={doc['rerank_score']:.4f}")
            print(f"       标题路径：{doc['metadata'].get('title_path', '无')}")
            print(f"       内容预览：{doc['text'][:100]}...")
        print("-" * 60)