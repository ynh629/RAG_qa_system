"""轻量运行时验证：加载真实数据 + 构建 BM25 + 初始化数据库（不加载重型模型）。

用法：python scripts/verify_runtime.py
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

out_lines = []


def log(msg: str):
    print(msg)
    out_lines.append(msg)


def main() -> int:
    ok = True
    try:
        from app.config import settings

        log(f"[1] 配置中心: model={settings.LLM_MODEL}")
        log(f"    DATA_DIR={settings.DATA_DIR}")
        log(f"    PDF_PATH 存在={os.path.exists(settings.PDF_PATH)}")
        log(f"    DATA_JSON 存在={os.path.exists(settings.DATA_JSON)}")

        # 2. 加载真实片段数据
        from rag_system.retrieval.chroma import load_segments

        segments = load_segments(settings.DATA_JSON)
        log(f"[2] 加载结构化片段: {len(segments)} 条")
        assert segments, "无有效片段"

        # 3. 构建 BM25 并检索
        from rag_system.retrieval.bm25 import BM25Retriever

        bm25 = BM25Retriever([s["content"] for s in segments])
        res = bm25.search("净利润", top_k=3)
        log(f"[3] BM25 检索 '净利润' -> {len(res)} 条结果")

        # 4. 初始化数据库
        from app.database import init_db

        init_db()
        log(f"[4] 数据库初始化完成: {settings.DB_PATH}")
        log(f"    数据库文件存在={os.path.exists(settings.DB_PATH)}")
    except Exception as e:
        import traceback

        log("=" * 60)
        log("运行时验证失败：")
        log(traceback.format_exc())
        ok = False

    # 写 UTF-8 日志
    out_path = os.path.join(ROOT_DIR, "scripts", "verify_runtime_out.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    log(f"\n>>> 运行时验证：{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
