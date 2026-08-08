# hybrid_retriever.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from chroma import load_segments, build_chroma_index, search_similar 
from bm25 import BM25Retriever  # 你之前写的类

class HybridRetriever:
    def __init__(self, chroma_collection, bm25_retriever: BM25Retriever):
        self.chroma_collection = chroma_collection
        self.bm25 = bm25_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "weighted",
        weights: Tuple[float, float] = (0.6, 0.4),
        candidate_k: int = 20  #每种检索器初筛的候选数量
    ) -> List[Dict]:
        # 1. 向量检索，获取候选集
        vec_results = self._retrieve_from_chroma(query, candidate_k)
        # 2. BM25 检索，获取候选集
        bm25_results = self.bm25.search(query, top_k=candidate_k)  # 返回包含 doc_index, score, text 
        # 4. 根据method融合
        if method == "weighted":
            merged = self._weighted_fusion(vec_results, bm25_results, weights)
        elif method == "rrf":
            merged = self._rrf_fusion(vec_results, bm25_results, k=60)
        else:
            raise ValueError(f"不支持的融合方法: {method}")

        # 5. 按融合分数降序排序，截取 top_k
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def _retrieve_from_chroma(self, query: str, n_results: int) -> List[Dict]:
        """调用 Chroma 检索，返回标准化结果列表。"""
        raw = self.chroma_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        results = []
        if raw["ids"][0]:
            for i in range(len(raw["ids"][0])):
                # Chroma 返回的距离是余弦距离 (0~2)，越小越相似
                dist = raw["distances"][0][i]
                # 转换为相似度（1 - 距离/2）或 1/(1+dist)，这里用 1 - dist/2 映射到 [0,1]
                sim = 1.0 - dist / 2.0
                results.append({
                    "id": raw["ids"][0][i],
                    "score": sim,          # 相似度，越大越好
                    "text": raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {}
                })
        return results

    def _weighted_fusion(
        self,
        vec_results: List[Dict],
        bm25_results: List[Dict],
        weights: Tuple[float, float]
    ) -> List[Dict]:
        """加权分数融合。"""
        # 归一化向量相似度到 [0,1]（可能已经近似了，但确保）
        vec_scores = np.array([r["score"] for r in vec_results])
        bm25_scores = np.array([r["score"] for r in bm25_results])

        # Min-Max 归一化，如果只有一个值或全相同，避免除以0
        def safe_minmax(arr):
            if len(arr) == 0:
                return arr
            min_v, max_v = arr.min(), arr.max()
            if max_v == min_v:
                return np.ones_like(arr) * 0.5
            return (arr - min_v) / (max_v - min_v)

        vec_norm = safe_minmax(vec_scores)
        bm25_norm = safe_minmax(bm25_scores)

        # 构建以文档标识为键的分数映射（使用文本哈希或 id）
        # 为方便，我们使用文本内容的前100字符或实际存储的 id。
        # 这里用文本内容的前100字符作为近似键（实际应用中最好用统一 ID）
        merged_dict = {}

        def add_to_dict(results, norm_scores, weight, source):
            for res, norm_score in zip(results, norm_scores):
                key = res["text"][:200]  # 截取前200字符作为键（谨慎，可能碰撞）
                if key not in merged_dict:
                    merged_dict[key] = {
                        "text": res["text"],
                        "metadata": res.get("metadata", {}),
                        "score": 0.0,
                        "source_scores": {}
                    }
                merged_dict[key]["score"] += weight * norm_score
                merged_dict[key]["source_scores"][source] = res["score"]

        add_to_dict(vec_results, vec_norm, weights[0], "vector")
        add_to_dict(bm25_results, bm25_norm, weights[1], "bm25")

        # 转换为列表
        merged = [
            {
                "id": key,
                "score": val["score"],
                "text": val["text"],
                "metadata": val["metadata"]
            }
            for key, val in merged_dict.items()
        ]
        return merged

    def _rrf_fusion(
        self,
        vec_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """倒数排名融合。"""
        # 按原始分数排序（降序）以获得排名
        vec_sorted = sorted(vec_results, key=lambda x: x["score"], reverse=True)
        bm25_sorted = sorted(bm25_results, key=lambda x: x["score"], reverse=True)

        rrf_scores = {}
        # 处理向量结果
        for rank, res in enumerate(vec_sorted, start=1):
            key = res["text"][:200]  # 使用文本前200字符作为键
            rrf_scores.setdefault(key, {"text": res["text"], "metadata": res.get("metadata", {}), "score": 0.0})
            rrf_scores[key]["score"] += 1.0 / (k + rank)
        # 处理 BM25 结果
        for rank, res in enumerate(bm25_sorted, start=1):
            key = res["text"][:200]
            rrf_scores.setdefault(key, {"text": res["text"], "metadata": res.get("metadata", {}), "score": 0.0})
            rrf_scores[key]["score"] += 1.0 / (k + rank)

        merged = [
            {"id": key, "score": val["score"], "text": val["text"], "metadata": val["metadata"]}
            for key, val in rrf_scores.items()
        ]
        return merged
# --------------------------- 主程序 ---------------------------
if __name__ == "__main__":
    # 1. 加载数据（复用 chroma.py 里的 load_segments）
    segments = load_segments("structured_segments.json")

    # 2. 构建 Chroma 向量索引
    print("正在构建向量索引...")
    chroma_coll = build_chroma_index(segments)

    # 3. 构建 BM25 索引
    print("正在构建 BM25 索引...")
    # 提取文档内容列表（与向量库中的顺序一致）
    documents = [seg["content"] for seg in segments]
    bm25_retriever = BM25Retriever(documents)

    # 4. 创建混合检索器
    hybrid = HybridRetriever(chroma_coll, bm25_retriever)

    # 5. 测试查询
    test_queries = [
        "公司2025年的净利润是多少？",
        "有哪些股东持有5%以上的股份？",
        "公司面临的主要风险有哪些？"
    ]

    for method in ["weighted", "rrf"]:
        print(f"\n{'='*60}")
        print(f"融合策略：{method.upper()}")
        for query in test_queries:
            print(f"\n查询：{query}")
            results = hybrid.search(query, top_k=3, method=method)
            if not results:
                print("  无匹配结果")
            for i, res in enumerate(results):
                print(f"  Top {i+1} (融合分数={res['score']:.4f})")
                print(f"    标题路径：{res['metadata'].get('title_path', '无')}")
                print(f"    内容预览：{res['text'][:120]}...")
        print("-" * 60)