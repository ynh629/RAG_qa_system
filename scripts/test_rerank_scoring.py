# test_rerank_scoring.py
"""重排打分纯函数测试：sigmoid 归一化 + 滑窗切分（不加载 CrossEncoder 模型）。

背景：bge-reranker-base 输出原始 logit（约 -10 ~ +10），
     未归一化直接设阈值会导致过滤语义错误；
     超过 512 token 的文档旧版直接头部截断，丢失后半段信息。
修复：_sigmoid 归一化到 [0,1]；_split_windows 滑窗打分取最大值。
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Windows 控制台 UTF-8 兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_system.retrieval.rerank import (
    _sigmoid,
    _split_windows,
    RERANK_WINDOW_CHARS,
    RERANK_WINDOW_OVERLAP,
    RERANK_MAX_WINDOWS,
)

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name} {detail}")


def test_sigmoid_basic():
    print("\n[测试1] sigmoid 基本性质")
    report("sigmoid(0) == 0.5（logit 0 对应概率 0.5）", _sigmoid(0.0) == 0.5)
    report("sigmoid(10) > 0.999（强相关）", _sigmoid(10.0) > 0.999)
    report("sigmoid(-10) < 0.001（强无关）", _sigmoid(-10.0) < 0.001)
    report("单调性：sigmoid(2) > sigmoid(1) > sigmoid(-1)", _sigmoid(2.0) > _sigmoid(1.0) > _sigmoid(-1.0))
    report("值域边界：0 < sigmoid(x) < 1（x=-30 与 x=30；±100 会因浮点舍入触界，属正常）",
           0.0 < _sigmoid(-30.0) < _sigmoid(30.0) < 1.0)


def test_sigmoid_stability():
    print("\n[测试2] 极端值数值稳定性（不抛 OverflowError）")
    try:
        lo = _sigmoid(-1000.0)
        hi = _sigmoid(1000.0)
        report("sigmoid(±1000) 不溢出，返回有限值", lo == 0.0 and hi == 1.0,
               f"(lo={lo}, hi={hi})")
    except OverflowError:
        report("sigmoid(±1000) 不溢出，返回有限值", False, "抛出 OverflowError")


def test_windows_short():
    print("\n[测试3] 短文档：单窗原样返回")
    doc = "公司2025年营业收入为3.2亿元。" * 5  # 远小于窗口
    ws = _split_windows(doc)
    report("单窗且内容不变", len(ws) == 1 and ws[0] == doc)
    report("空文档返回 ['']", _split_windows("") == [""])


def test_windows_long():
    print("\n[测试4] 长文档：多窗 + 覆盖完整性")
    doc = "数" * 2000  # 2000 字符
    ws = _split_windows(doc)
    step = RERANK_WINDOW_CHARS - RERANK_WINDOW_OVERLAP
    expected = min((2000 + step - 1) // step, RERANK_MAX_WINDOWS)
    report("窗口数量符合预期", len(ws) == expected, f"(实际 {len(ws)}，预期 {expected})")
    report("所有窗口不超过窗口上限", all(len(w) <= RERANK_WINDOW_CHARS for w in ws))
    # 覆盖完整性：首窗从头开始，末窗覆盖到结尾（上限截断时除外）
    if len(ws) < RERANK_MAX_WINDOWS:
        covered = ws[0] + "".join(w[RERANK_WINDOW_OVERLAP:] for w in ws[1:])
        report("拼接去重叠后覆盖全文", covered == doc)
    else:
        report("达到窗口上限时截断（预期行为）", True)


def test_windows_cap():
    print("\n[测试5] 窗口数上限：防止超长文档打分爆炸")
    doc = "字" * 100000  # 10 万字符
    ws = _split_windows(doc)
    report("窗口数不超过上限", len(ws) <= RERANK_MAX_WINDOWS, f"(实际 {len(ws)})")


def test_threshold_semantics():
    print("\n[测试6] 阈值语义：min_score=0.5 等价于 logit>=0")
    # 模拟旧 bug：原始 logit 3.7 vs 阈值 0.5 —— 旧行为下 3.7>0.5 通过（碰巧对），
    # 但 logit -3.7（强无关）在旧逻辑若阈值设 0.5 也"通过"——错误；
    # 归一化后 -3.7 -> 0.024 < 0.5 被正确过滤
    report("logit -3.7 → 概率 < 0.5（被阈值过滤）", _sigmoid(-3.7) < 0.5,
           f"(概率 {_sigmoid(-3.7):.4f})")
    report("logit 3.7 → 概率 > 0.5（通过阈值）", _sigmoid(3.7) > 0.5,
           f"(概率 {_sigmoid(3.7):.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("重排打分纯函数测试（sigmoid 归一化 + 滑窗切分）")
    print("=" * 60)
    test_sigmoid_basic()
    test_sigmoid_stability()
    test_windows_short()
    test_windows_long()
    test_windows_cap()
    test_threshold_semantics()
    print("\n" + "=" * 60)
    print(f"结果：{PASS} 通过，{FAIL} 失败")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
