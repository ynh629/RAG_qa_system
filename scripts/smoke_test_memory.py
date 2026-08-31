# 多轮记忆模块冒烟测试（不调用真实 LLM）
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.conversation import estimate_tokens, history_tokens, trim_history, rewrite_query


def test_trim_turns():
    msgs = []
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"消息{i}"})
    trimmed = trim_history(msgs, max_turns=3, token_budget=100000)
    assert len(trimmed) == 6, f"应保留 3 轮=6 条，实际 {len(trimmed)}"
    assert trimmed[0]["content"] == "消息2", "应从最旧截断"
    print("PASS: 轮数裁剪（8 条 → 6 条，从最旧截断）")


def test_trim_token_budget():
    msgs = [
        {"role": "user", "content": "问题" * 500},
        {"role": "assistant", "content": "回答" * 500},
        {"role": "user", "content": "新问题"},
    ]
    trimmed = trim_history(msgs, max_turns=5, token_budget=200)
    total = history_tokens(trimmed)
    assert total <= 200, f"token 预算超限: {total}"
    assert trimmed[-1]["content"] == "新问题", "最新消息必须保留"
    print(f"PASS: token 预算裁剪（{total} tokens <= 200，保留最新）")


def test_trim_strips_extra_fields():
    msgs = [
        {"role": "assistant", "content": "答案", "sources": ["x"], "chat_id": 42},
    ]
    trimmed = trim_history(msgs, max_turns=5, token_budget=100000)
    assert set(trimmed[0].keys()) == {"role", "content"}, f"多余字段未剔除: {trimmed[0].keys()}"
    print("PASS: 剔除 sources/chat_id 等额外字段（可直接发给 LLM）")


def test_rewrite_no_history():
    assert rewrite_query(None, "m", "净利润是多少", None) == "净利润是多少"
    assert rewrite_query(None, "m", "净利润是多少", []) == "净利润是多少"
    print("PASS: 无历史时不改写、不调用 LLM")


def test_rewrite_llm_failure_fallback():
    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("模拟 LLM 故障")

    q = "那净利润呢"
    out = rewrite_query(Boom(), "deepseek-chat", q, [{"role": "user", "content": "贵州茅台2023年营收"}])
    assert out == q, f"LLM 故障应回退原始问题，实际 {out}"
    print("PASS: LLM 调用失败时回退原始问题（可用性优先）")


def test_rewrite_success():
    class Fake:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class Msg:
                        content = "“贵州茅台2023年净利润是多少”"

                    class Choice:
                        message = Msg()

                    class Resp:
                        choices = [Choice()]

                    return Resp()

    out = rewrite_query(Fake(), "deepseek-chat", "那净利润呢", [
        {"role": "user", "content": "贵州茅台2023年营业收入是多少"}
    ])
    assert out == "贵州茅台2023年净利润是多少", f"应去除引号，实际 {out!r}"
    print(f"PASS: 正常改写并去除引号 → {out}")


def test_estimate_tokens():
    t = estimate_tokens("公司2025年净利润是多少？")
    assert t > 0
    assert estimate_tokens("") == 0
    print(f"PASS: token 估算（'公司2025年净利润是多少？' ≈ {t} tokens）")


if __name__ == "__main__":
    test_trim_turns()
    test_trim_token_budget()
    test_trim_strips_extra_fields()
    test_rewrite_no_history()
    test_rewrite_llm_failure_fallback()
    test_rewrite_success()
    test_estimate_tokens()
    print("\n=== ALL MEMORY TESTS PASSED ===")
