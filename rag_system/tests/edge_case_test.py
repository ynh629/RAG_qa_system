# edge_case_test.py
"""
RAG 系统边界情况测试脚本。
覆盖以下场景：
1. 空查询 / 无意义查询
2. 超长查询
3. 查询不存在的概念
4. 空文档构建索引
5. 数据文件不存在
6. 融合空结果
7. 上下文 token 预算控制
8. 重排序质量过滤
"""
import os
import sys
import json
import tempfile

# 确保控制台输出支持 UTF-8（避免 Windows GBK 编码问题）
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 确保仓库根目录在 sys.path，支持直接运行（python rag_system/tests/edge_case_test.py）与任意工作目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag_system.common.logging_config import get_logger
from rag_system.common.exceptions import DocumentError, IndexError_

logger = get_logger("edge_case_test")

# 导入被测模块
from rag_system.retrieval.chroma import load_segments, build_chroma_index
from rag_system.retrieval.bm25 import BM25Retriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.parsing.markdown_splitter import MarkdownHeadingSplitter

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    """输出测试结果。"""
    global PASS, FAIL
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {status} {name} {detail}")


def test_empty_query():
    """测试空查询。"""
    print("\n[测试1] 空查询")
    # 构造最小可用的检索器
    docs = ["湖北美尔雅股份有限公司2025年年度报告摘要"]
    bm25 = BM25Retriever(docs)
    # 用 mock 的 chroma collection
    class MockCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
    hybrid = HybridRetriever(MockCollection(), bm25)
    result = hybrid.search("", top_k=3)
    report("空字符串查询返回空列表", result == [])
    result2 = hybrid.search("   ", top_k=3)
    report("纯空白查询返回空列表", result2 == [])


def test_meaningless_query():
    """测试无意义查询（不存在的概念）。"""
    print("\n[测试2] 无意义查询")
    docs = ["公司2025年净利润为1.2亿元", "股东持股比例超过5%的有3家"]
    bm25 = BM25Retriever(docs)
    class MockCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
    hybrid = HybridRetriever(MockCollection(), bm25)
    # 查询完全不存在的概念
    result = hybrid.search("量子计算机的研发进展", top_k=3)
    # 不应崩溃，返回列表（可能为空或兜底结果）
    report("无意义查询不崩溃", isinstance(result, list))


def test_long_query():
    """测试超长查询。"""
    print("\n[测试3] 超长查询")
    docs = ["公司2025年净利润为1.2亿元"]
    bm25 = BM25Retriever(docs)
    class MockCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
    hybrid = HybridRetriever(MockCollection(), bm25)
    long_query = "净利润" * 1000  # 2000 字符
    result = hybrid.search(long_query, top_k=3)
    report("超长查询不崩溃", isinstance(result, list))


def test_empty_documents_bm25():
    """测试空文档构建 BM25 索引。"""
    print("\n[测试4] 空文档构建 BM25 索引")
    try:
        BM25Retriever([])
        report("空文档应抛出异常", False)
    except IndexError_:
        report("空文档抛出 IndexError_", True)
    except Exception as e:
        report(f"空文档抛出其他异常: {type(e).__name__}", False)


def test_missing_data_file():
    """测试数据文件不存在。"""
    print("\n[测试5] 数据文件不存在")
    try:
        load_segments("/nonexistent/path/segments.json")
        report("文件不存在应抛出异常", False)
    except DocumentError:
        report("文件不存在抛出 DocumentError", True)
    except Exception as e:
        report(f"文件不存在抛出其他异常: {type(e).__name__}", False)


def test_invalid_json():
    """测试 JSON 格式错误。"""
    print("\n[测试6] JSON 格式错误")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write("这不是合法的JSON{{{")
        tmp_path = f.name
    try:
        load_segments(tmp_path)
        report("非法 JSON 应抛出异常", False)
    except DocumentError:
        report("非法 JSON 抛出 DocumentError", True)
    except Exception as e:
        report(f"非法 JSON 抛出其他异常: {type(e).__name__}", False)
    finally:
        os.unlink(tmp_path)


def test_fusion_empty_results():
    """测试融合时空结果（weighted 方法不应崩溃）。"""
    print("\n[测试7] 融合空结果")
    docs = ["测试文档内容"]
    bm25 = BM25Retriever(docs)
    class MockCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
    hybrid = HybridRetriever(MockCollection(), bm25)
    # 向量为空，BM25 有结果
    result = hybrid.search("测试", top_k=3, method="weighted")
    report("weighted 融合空向量结果不崩溃", isinstance(result, list))
    # rrf 方法
    result2 = hybrid.search("测试", top_k=3, method="rrf")
    report("rrf 融合空向量结果不崩溃", isinstance(result2, list))


def test_markdown_splitter_empty():
    """测试空 Markdown 文本切分。"""
    print("\n[测试8] 空 Markdown 文本切分")
    splitter = MarkdownHeadingSplitter("")
    result = splitter.split_by_headings(mode="leaf")
    report("空文本返回空列表", result == [])
    splitter2 = MarkdownHeadingSplitter("   \n  ")
    result2 = splitter2.split_by_headings(mode="leaf")
    report("纯空白文本返回空列表", result2 == [])


def test_markdown_splitter_no_heading():
    """测试无标题 Markdown 文本切分。"""
    print("\n[测试9] 无标题 Markdown 文本切分")
    splitter = MarkdownHeadingSplitter("这是一段没有标题的普通文本内容。")
    result = splitter.split_by_headings(mode="leaf")
    report("无标题文本返回单个片段", len(result) == 1 and result[0]["content"] != "")


def test_markdown_splitter_empty_content_filter():
    """测试空内容片段过滤。"""
    print("\n[测试10] 空内容片段过滤")
    md = "# 标题一\n\n# 标题二\n\n# 标题三\n"
    splitter = MarkdownHeadingSplitter(md)
    result = splitter.split_by_headings(mode="leaf")
    # 所有片段内容都应为空，应被过滤
    report("空内容片段被过滤", all(seg["content"] for seg in result))


def test_token_budget():
    """测试上下文 token 预算控制（qa_system 的 _estimate_tokens）。"""
    print("\n[测试11] token 估算函数")
    from app.qa_system import _estimate_tokens
    tokens = _estimate_tokens("公司2025年净利润是多少？")
    report("token 估算返回正数", tokens > 0)
    tokens_empty = _estimate_tokens("")
    report("空文本 token 估算为 0", tokens_empty == 0)


def test_rerank_empty_candidates():
    """测试重排序空候选。"""
    print("\n[测试12] 重排序空候选")
    # 不加载真实模型，直接测试空候选的短路逻辑
    # 通过 monkeypatch 避免加载模型
    import rag_system.retrieval.rerank as rerank
    original_init = rerank.Reranker.__init__
    def fake_init(self, model_name="x", backend="bge"):
        self.backend = backend
        self.model = None
    rerank.Reranker.__init__ = fake_init
    try:
        r = rerank.Reranker()
        result = r.rerank("query", [])
        report("空候选返回空列表", result == [])
    finally:
        rerank.Reranker.__init__ = original_init


def run_all():
    """运行所有测试。"""
    print("=" * 60)
    print("RAG 系统边界情况测试")
    print("=" * 60)

    test_empty_query()
    test_meaningless_query()
    test_long_query()
    test_empty_documents_bm25()
    test_missing_data_file()
    test_invalid_json()
    test_fusion_empty_results()
    test_markdown_splitter_empty()
    test_markdown_splitter_no_heading()
    test_markdown_splitter_empty_content_filter()
    test_token_budget()
    test_rerank_empty_candidates()

    print("\n" + "=" * 60)
    print(f"测试完成：{PASS} 通过，{FAIL} 失败")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
