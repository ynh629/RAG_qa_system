"""快速冒烟测试：验证应用层与引擎层的包导入是否正常。

用法：python scripts/smoke_test.py
"""
import os
import sys
import traceback

# 确保仓库根目录在 sys.path 中（支持 python scripts/smoke_test.py 直接运行；
# 正式部署建议 pip install -e . 让 app/rag_system 全局可导入）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


def main() -> int:
    results = []

    # ---------- 必需模块（必须导入成功） ----------
    required = [
        ("app.config", "from app.config import settings"),
        ("app.qa_system", "import app.qa_system"),
        ("app.api", "import app.api"),
        ("app.cli", "import app.cli"),
        ("app.database", "import app.database"),
        ("app.models", "import app.models"),
        ("app.schemas", "import app.schemas"),
        ("rag_system.retrieval.bm25", "import rag_system.retrieval.bm25"),
        ("rag_system.retrieval.chroma", "import rag_system.retrieval.chroma"),
        ("rag_system.retrieval.hybrid_retriever", "import rag_system.retrieval.hybrid_retriever"),
        ("rag_system.retrieval.rerank", "import rag_system.retrieval.rerank"),
        ("rag_system.parsing.parse_pdf", "import rag_system.parsing.parse_pdf"),
        ("rag_system.parsing.markdown_splitter", "import rag_system.parsing.markdown_splitter"),
        ("rag_system.splitting.advanced_splitting", "import rag_system.splitting.advanced_splitting"),
        ("rag_system.common.logging_config", "import rag_system.common.logging_config"),
        ("rag_system.common.exceptions", "import rag_system.common.exceptions"),
    ]
    for name, stmt in required:
        try:
            exec(stmt)
            results.append((name, OK))
        except Exception as e:
            results.append((name, FAIL, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    # ---------- 可选模块（依赖 ragas 等，未安装则告警） ----------
    optional = [
        ("rag_system.evaluation.ragas_config", "import rag_system.evaluation.ragas_config"),
        ("rag_system.evaluation.generate_qa_pairs", "import rag_system.evaluation.generate_qa_pairs"),
        ("rag_system.evaluation.review_qa_pairs", "import rag_system.evaluation.review_qa_pairs"),
        ("rag_system.evaluation.run_eval", "import rag_system.evaluation.run_eval"),
    ]
    for name, stmt in optional:
        try:
            exec(stmt)
            results.append((name, OK))
        except ImportError as e:
            results.append((name, WARN, f"可选依赖未安装: {e.name}"))
        except Exception as e:
            results.append((name, WARN, f"{type(e).__name__}: {e}"))

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    for row in results:
        name, status = row[0], row[1]
        detail = f" -> {row[2]}" if len(row) > 2 else ""
        print(f"[{status}] {name}{detail}")
    print("=" * 60)

    n_fail = sum(1 for r in results if r[1] == FAIL)
    n_warn = sum(1 for r in results if r[1] == WARN)
    summary = f"结果：FAIL={n_fail}  WARN={n_warn}  总模块={len(results)}"
    print(summary)

    # 同时写入 UTF-8 日志文件，便于跨终端查看
    lines = [f"[{r[1]}] {r[0]}" + (f" -> {r[2]}" if len(r) > 2 else "") for r in results]
    lines.append("=" * 60)
    lines.append(summary)
    out_path = os.path.join(ROOT_DIR, "scripts", "smoke_test_out.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
