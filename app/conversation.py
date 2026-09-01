# conversation.py
"""多轮对话记忆：历史裁剪与检索查询改写。

历史消息统一使用 OpenAI 消息格式：[{"role": "user" | "assistant", "content": str}]
- trim_history：把任意来源（前端 session / 数据库）的历史裁剪到轮数与 token 预算内
- rewrite_query：结合历史把省略、指代型问题改写为完整独立的检索问题
"""
from typing import Dict, List, Optional

from rag_system.common.logging_config import get_logger

from app.config import settings

logger = get_logger(__name__)

# 记忆容量：保留最近 N 轮（1 轮 = 一问一答）
MAX_HISTORY_TURNS = settings.MAX_HISTORY_TURNS
# 记忆 token 预算：历史总 token 超过此值时从最旧开始丢弃
HISTORY_TOKEN_BUDGET = settings.HISTORY_TOKEN_BUDGET


# tiktoken 编码器懒加载单例（首次调用初始化，失败则永久退化为字符估算）
_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    global _ENCODER, _ENCODER_TRIED
    if not _ENCODER_TRIED:
        _ENCODER_TRIED = True
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数。
    优先使用 tiktoken（若已安装），否则退化为字符数估算（中文约 1.5 字符 ≈ 1 token）。
    """
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    # 中文场景：约 1.5 字符 ≈ 1 token，保守估算
    return int(len(text) / 1.5)


def history_tokens(messages: List[Dict]) -> int:
    """计算一组历史消息的总 token 数（只统计 content）。"""
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def trim_history(
    messages: List[Dict],
    max_turns: int = MAX_HISTORY_TURNS,
    token_budget: int = HISTORY_TOKEN_BUDGET,
) -> List[Dict]:
    """
    裁剪历史消息：只保留最近 max_turns 轮，且总 token 不超预算（超出时丢最旧的）。

    返回新列表，不修改传入数据；输出只保留 role / content 两个字段
    （前端消息可能带 sources、chat_id 等额外字段，直接发给 LLM 会被拒绝）。
    """
    if not messages:
        return []
    trimmed = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in messages[-max_turns * 2:]
    ]
    total = history_tokens(trimmed)
    while trimmed and total > token_budget:
        dropped = trimmed.pop(0)
        total -= estimate_tokens(dropped.get("content", ""))
        logger.info("历史超出 token 预算，丢弃最旧消息（%d 字符）", len(dropped.get("content", "")))
    return trimmed


QUERY_REWRITE_SYSTEM_PROMPT = (
    "你是检索查询改写器。请根据对话历史，把用户的最新问题改写成一个独立、完整、"
    "适合知识库检索的问题：\n"
    "1. 补全省略的主语和宾语（例如「那净利润呢」应改写为「贵州茅台2023年净利润是多少」）。\n"
    "2. 保留原问题的所有限定条件（年份、指标、对象等）。\n"
    "3. 只输出改写后的问题本身，不要解释、不要引号。\n"
    "4. 如果最新问题本身已经完整独立，原样输出它。"
)


def rewrite_query(
    llm_client,
    llm_model: str,
    query: str,
    history: Optional[List[Dict]] = None,
) -> str:
    """
    结合对话历史改写用户问题，消除指代与省略，用于提升检索召回质量。

    可用性优先：无历史、改写失败或结果异常时，原样返回 query。
    """
    if not history:
        return query

    # 改写只需少量历史：裁到 3 轮 / 1000 token 以内，控制成本与延迟
    recent = trim_history(history, max_turns=3, token_budget=1000)
    conversation = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
        for m in recent
    )
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"对话历史：\n{conversation}\n\n最新问题：{query}\n\n改写后的问题："},
            ],
            temperature=0.0,
            max_tokens=200,
            timeout=15,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        # 去掉模型可能自带的引号
        rewritten = rewritten.strip('"“”\'').strip()
        if rewritten and len(rewritten) <= 200:
            if rewritten != query:
                logger.info("查询改写：%r → %r", query, rewritten)
            return rewritten
        logger.warning("查询改写结果为空或过长，使用原始问题")
        return query
    except Exception as e:
        logger.warning("查询改写失败，使用原始问题：%s", e)
        return query
