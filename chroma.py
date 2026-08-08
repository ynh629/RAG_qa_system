import json
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict

# --------------------------- 1、引入数据 ---------------------------
def load_segments(json_path: str) -> List[Dict]:
    """从 JSON 加载结构化片段，返回片段列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    # 变量：筛选出 content 非空的片段
    valid_segments = [seg for seg in segments if seg.get("content") and seg["content"].strip()]
    print(f"成功加载 {len(valid_segments)} 个有效片段")
    return valid_segments

# --------------------------- 2. 构建向量索引 ---------------------------
def build_chroma_index(segments: List[Dict], collection_name: str = "annual_report"):
    """
    使用 Chroma 构建向量索引。
    返回：collection 对象，可用于检索。
    """
    # 创建 Chroma 客户端（数据会持久化到 ./chroma_db 目录）
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 定义 embedding 函数（使用 BGE 中文小模型，首次会自动下载）
    # 变量：embedding 函数，负责将文本转为向量
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    
    # 如果 collection 已存在，先删除（可选，这里为了重建）
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    # 创建 collection（类似关系数据库的表）
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "年报摘要向量库"}
    )
    
    # 准备批量数据
    # 变量：文档 id 列表，每个片段自动生成一个唯一 ID
    ids = []
    # 变量：要向量化的文本内容列表
    documents = []
    # 变量：元数据列表（每个片段附带的信息，如标题路径）
    metadatas = []
    
    for i, seg in enumerate(segments):
        ids.append(f"chunk_{i}")
        documents.append(seg["content"])
        # 存储标题路径为元数据，方便检索后查看来源
        title_path = seg.get("title_path", [])
        metadatas.append({
            "title_path": " > ".join(title_path) if title_path else "无标题",
            "level": seg.get("level", 0)
        })
    
    # 批量添加文档
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"索引构建完成，共 {len(documents)} 个向量")
    return collection

# --------------------------- 3. 检索函数 ---------------------------
def search_similar(query: str, collection, top_k: int = 3):
    """
    在 collection 中搜索与 query 最相似的 top_k 个文本块。
    返回：查询结果列表，包含 id, content, metadata, distance
    """
    # 变量：查询结果，包含 distances, metadatas, documents
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results

# --------------------------- 4. 主程序 ---------------------------
if __name__ == "__main__":
    # 步骤1：加载片段
    segments = load_segments("structured_segments.json")
    
    # 步骤2：构建索引
    collection = build_chroma_index(segments)
    
    # 步骤3：测试检索
    test_queries = [
        "公司2025年的净利润是多少？",
        "有哪些股东持有5%以上的股份？",
        "公司面临的主要风险有哪些？"
    ]
    
    print("\n" + "="*60)
    for query in test_queries:
        print(f"查询问题：{query}")
        results = search_similar(query, collection, top_k=2)
        for i, (doc_id, distance, doc_text, metadata) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0]
        )):
            print(f"  第{i+1}个结果（距离：{distance:.3f}）:")
            print(f"    标题路径：{metadata.get('title_path', '无')}")
            print(f"    内容预览：{doc_text[:100]}...")
        print("-"*60)
