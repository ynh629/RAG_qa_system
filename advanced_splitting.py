#切片
import json
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
def load_text_from_segments(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
        # 变量：将所有片段的 content 用双换行连接，保留段落边界
    full_text = "\n\n".join(seg["content"] for seg in segments if seg["content"].strip())
    return full_text
#---------------------RecursiveCharacterTextSplitter递归按分隔符切分------------------
def run_recursive_splitter(text,chunk_size=1024,chunk_overlap=200):  #递归字符拆分器
    separators = ["\n\n", "\n", "。", ".", "；",]  #优先分隔列表
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,  #块大小
        chunk_overlap=chunk_overlap,  #块重叠
        length_function=len
    )
    docs = splitter.create_documents([text])
    chunks=[doc.page_content for doc in docs]
    print(f"RecursiveSplitter (size={chunk_size}, overlap={chunk_overlap})")
    print(f"  块数量: {len(chunks)}")
    print(f"  平均长度: {np.mean([len(c) for c in chunks]):.0f} 字符")
    print(f"  示例块前100字: {chunks[0][:100]}...")
    return chunks
#--------------------------语义切片，根据向量embedding差异---------------------
def semantic_chunking(text,model_name="BAAI/bge-small-zh-v1.5",threshold=0.5):
    model=SentenceTransformer(model_name)
    sentences = text.replace("?", "。").replace("!", "。").split("。")
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <=1:
        return [text]
    embeddings=model.encode(sentences)
    similarities=[]
    for i in range(len(embeddings)-1):
        sim=cosine_similarity([embeddings[i]],[embeddings[i+1]])[0][0]
        similarities.append(sim)
    breakpoints=[0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i+1)
    breakpoints.append(len(sentences))
    chunks=[]
    for start,end in zip(breakpoints[:-1], breakpoints[1:]):
        chunk_text="。".join(sentences[start:end])
        chunks.append(chunk_text)
    print(f"SemanticChunker (threshold={threshold})")
    print(f"  块数量: {len(chunks)}")
    print(f"  平均长度: {np.mean([len(c) for c in chunks]):.0f} 字符")
    print(f"  示例块前100字: {chunks[0][:100]}...")
    return chunks
# -----------------------对比实验主程序--------------------------
if __name__ == "__main__":
    full_text = load_text_from_segments("structured_segments.json")
    print(f"原始文本总长度: {len(full_text)} 字符\n")
    # 测试不同参数的 RecursiveCharacterTextSplitter
    print("=" * 50)
    chunks_r_500 = run_recursive_splitter(full_text, chunk_size=500, chunk_overlap=50)
    print()
    chunks_r_1000 = run_recursive_splitter(full_text, chunk_size=1000, chunk_overlap=100)
    print()
    chunks_r_1500 = run_recursive_splitter(full_text, chunk_size=1500, chunk_overlap=150)
    print()
    # 测试语义切片
    print("=" * 50)
    chunks_sem = semantic_chunking(full_text, threshold=0.5)
    print()
    # 可尝试不同阈值，如 0.3、0.7，chunks_sem_high = semantic_chunking(full_text, threshold=0.7)

    # 简单对比
    print("=" * 50)
    print("对比摘要：")
    print(f"  Recursive(500,50)  块数: {len(chunks_r_500)}  平均长度: {np.mean([len(c) for c in chunks_r_500]):.0f}")
    print(f"  Recursive(1000,100) 块数: {len(chunks_r_1000)} 平均长度: {np.mean([len(c) for c in chunks_r_1000]):.0f}")
    print(f"  Recursive(1500,150) 块数: {len(chunks_r_1500)} 平均长度: {np.mean([len(c) for c in chunks_r_1500]):.0f}")
    print(f"  Semantic(0.5)      块数: {len(chunks_sem)}     平均长度: {np.mean([len(c) for c in chunks_sem]):.0f}")
