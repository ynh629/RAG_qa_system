import json
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

class BM25Retriever:
    """
    基于 BM25 算法的关键词检索器。
    """
    def __init__(self, documents: List[str]):
        """
        初始化并构建 BM25 索引。
        """
        # 变量：存储原始文档，便于检索后返回文本
        self.documents = documents
        # 变量：分词后的文档列表（每个文档是一个词语列表）
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        # 变量：构建 BM25 模型（内部计算词频、文档频率等参数）
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        # 使用 jieba 分词，默认精确模式
        # 变量：jieba 分词结果，去掉空格
        tokens = jieba.lcut(text)
        # 过滤掉空白词
        return [t.strip() for t in tokens if t.strip()]
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        根据查询词检索最相关的 top_k 个文档。
        返回：
            结果列表，每个元素是字典，包含：
                - doc_index: 原始文档列表中的索引
                - score: BM25 相关性分数（越高越相关）
                - text: 文档全文
        """
        # 变量：将查询文本分词
        query_tokens = self._tokenize(query)
        # 变量：计算所有文档的 BM25 分数（numpy 数组）
        scores = self.bm25.get_scores(query_tokens)
        # 变量：按照分数降序排列，取 top_k 个索引
        # 使用 numpy 的 argsort 返回从小到大排序的索引，取最后 k 个并反转
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            # 只返回分数大于0的结果（至少有一个关键词匹配）
            if scores[idx] > 0:
                results.append({
                    "doc_index": int(idx),
                    "score": float(scores[idx]),
                    "text": self.documents[idx]
                })
        return results

# --------------------------- 工具函数 ---------------------------
def load_documents_from_segments(json_path: str) -> List[str]:
    """
    从结构化片段 JSON 中提取所有文本块内容。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    # 变量：过滤出 content 不为空的片段
    documents = [seg["content"] for seg in segments if seg.get("content") and seg["content"].strip()]
    print(f"从 {json_path} 加载了 {len(documents)} 个文本块")
    return documents

# --------------------------- 主程序 ---------------------------
if __name__ == "__main__":
    # 1. 加载文档（使用之前 parse_pdf.py 生成的 structured_segments.json）
    docs = load_documents_from_segments("structured_segments.json") 
    # 2. 构建 BM25 索引
    bm25_retriever = BM25Retriever(docs)
    print("BM25 索引构建完成\n")
    
    # 3. 测试检索
    test_queries = [
        "净利润",
        "股东持股比例",
        "风险分析"
    ]
    for query in test_queries:
        print(f"查询：{query}")
        results = bm25_retriever.search(query, top_k=2)
        if not results:
            print("  无匹配结果")
        for i, res in enumerate(results):
            print(f"  Top {i+1} (BM25 分数={res['score']:.2f}):")
            # 截断文本前100字显示
            print(f"    内容：{res['text'][:100]}...")
        print("-" * 50)