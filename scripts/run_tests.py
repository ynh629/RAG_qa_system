"""运行 RAG 引擎边界测试，并将结果保存为 UTF-8 日志文件。

用法：python scripts/run_tests.py
"""
import contextlib
import io
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

out_path = os.path.join(ROOT_DIR, "scripts", "edge_case_test_out.txt")
buf = io.StringIO()

try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        from rag_system.tests.edge_case_test import run_all

        ok = run_all()
    buf.write(f"\n>>> 边界测试退出码：{'PASS' if ok else 'FAIL'}\n")
except Exception:
    import traceback

    buf.write("\n" + "=" * 60 + "\n")
    buf.write("测试运行失败：\n")
    buf.write(traceback.format_exc())
    ok = False

with open(out_path, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())

print(f"测试结果已写入: {out_path}")
print("=" * 60)
print(buf.getvalue())
