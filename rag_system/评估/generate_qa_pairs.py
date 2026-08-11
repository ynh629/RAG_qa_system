# generate_qa_pairs.py
"""
Phase 1：使用大模型自动生成 QA 对（评估数据集候选）。

流程：
    读取文本切分结果（chunks_recursive.json 或 structured_segments.json），
    逐块调用 qwen-plus（instructor 结构化输出），每块生成 N 条问答对，
    结果保存为 data/qa_pairs_raw.json 供人工审查（Phase 2）。

用法：
    # 全量生成（每块 4 条，目标约 64 条候选）
    python 评估/generate_qa_pairs.py

    # 自定义输入/输出/每块条数
    python 评估/generate_qa_pairs.py \
        --input data/structured_segments.json \
        --output data/qa_pairs_raw.json \
        --per-chunk 5

    # 只处理部分块（断点续跑 / 分批生成）
    python 评估/generate_qa_pairs.py --start 0 --end 4
"""
import argparse
import json
import os
import sys
import time
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 确保可以导入上级目录的公共模块（日志、异常）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from 系统日志.config import get_logger

logger = get_logger(__name__)

# 数据文件路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "chunks_recursive.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "qa_pairs_raw.json")

load_dotenv()

# 问题类型与难度（用于约束和校验模型输出）
QUESTION_TYPES = ["数值提取", "事实检索", "归纳概括", "风险分析", "对比分析"]
DIFFICULTIES = ["easy", "medium", "hard"]

# 默认大模型配置
DEFAULT_MODEL = "qwen-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "qwen_api_key"


class QAPair(BaseModel):
    """单条问答对（instructor 结构化输出）。"""

    question: str = Field(..., description="基于该文本块内容可以回答的问题")
    ground_truth: str = Field(..., description="标准答案，必须直接从原文中提取，禁止编造")
    question_type: str = Field(
        ..., description="问题类型，取值: 数值提取/事实检索/归纳概括/风险分析/对比分析"
    )
    difficulty: str = Field(..., description="难度: easy/medium/hard")
    source_sentence: str = Field(..., description="支撑答案的原文句子（原样引用）")


class QAPairList(BaseModel):
    """一个文本块生成的问答对集合。"""

    pairs: List[QAPair]


# 系统提示词：约束大模型生成高质量、可落地校验的 QA 对
SYSTEM_PROMPT = """你是一位专业的年报分析助手，正在为问答系统构建评估数据集。
任务：阅读给定的年报文本块，从中生成指定数量的问答对（QA pair）。

生成规则：
1. 问题必须仅基于该文本块中的信息即可回答，不要提问文本块之外的内容。
2. 答案必须直接从原文中提取或严格依据原文概括，禁止编造、推测或引入外部知识。
3. 优先选择包含具体数据、事实陈述、定义、结论的句子作为提问点。
4. 问题类型要多样化，各类型示例：
   - 数值提取：问具体数字/比例/金额（如"2025年服装行业工业增加值同比下降多少？"）
   - 事实检索：问是什么/在哪里/谁/何时（如"公司主营什么业务？"）
   - 归纳概括：问包含哪些内容/由哪几部分构成
   - 风险分析：问风险/挑战/影响（如"公司面临哪些外部风险？"）
   - 对比分析：问对比/差异/变化趋势
5. 难度分级：
   - easy：答案可在单个句子中直接找到
   - medium：需要整合 2~3 个句子
   - hard：需要通读文本块后归纳总结
6. source_sentence 字段必须原样引用支撑答案的原文句子，用于人工校验答案是否 grounded。
7. 问题表述要自然，模拟真实用户提问，避免照抄原文句式。
8. 若文本块大部分是表格/图片占位符等无有效信息内容，可适当减少条数，但不要编造。"""


def create_instructor_client():
    """创建 instructor 客户端（结构化输出）。"""
    import instructor
    from openai import OpenAI

    api_key = os.getenv(DEFAULT_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"未设置环境变量 {DEFAULT_API_KEY_ENV}，请检查项目根目录 .env 文件"
        )
    base_client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
    return instructor.from_openai(base_client, mode=instructor.Mode.JSON)


def build_user_prompt(chunk_text: str, title_path: List[str], per_chunk: int) -> str:
    """构造单块的用户提示词。"""
    title = " > ".join(title_path) if title_path else "（无标题）"
    return f"""请阅读下面的年报文本块，生成 {per_chunk} 条问答对。

【所属章节】{title}
【文本块】
{chunk_text}
"""


def generate_for_chunk(
    client,
    chunk: dict,
    per_chunk: int,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> List[dict]:
    """
    为单个文本块生成 QA 对（带重试）。
    返回已附加上下文元数据的 QA 对列表（dict 形式）。
    """
    chunk_text = chunk.get("content", "")
    title_path = chunk.get("title_path", [])

    if not chunk_text or not chunk_text.strip():
        logger.warning("跳过空文本块: %s", title_path)
        return []

    result_pairs = []
    for attempt in range(1, max_retries + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                response_model=QAPairList,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(chunk_text, title_path, per_chunk)},
                ],
                temperature=0.7,
            )
            result_pairs = [p for p in result.pairs if p.question and p.ground_truth]
            break
        except Exception as e:
            logger.warning("第 %d/%d 次生成失败: %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(2)
    if not result_pairs:
        logger.error("块生成失败（重试 %d 次后仍无结果）: %s", max_retries, title_path)

    # 校验问题类型/难度字段，非法值降级处理
    valid_pairs = []
    for p in result_pairs:
        qtype = p.question_type if p.question_type in QUESTION_TYPES else "事实检索"
        diff = p.difficulty if p.difficulty in DIFFICULTIES else "medium"
        valid_pairs.append({
            "question": p.question.strip(),
            "ground_truth": p.ground_truth.strip(),
            "question_type": qtype,
            "difficulty": diff,
            "source_sentence": (p.source_sentence or "").strip(),
        })
    return valid_pairs


def main():
    parser = argparse.ArgumentParser(description="Phase 1：LLM 自动生成 QA 对（评估数据集候选）")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"输入文本块 JSON（默认: {DEFAULT_INPUT}）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"输出候选 QA 对 JSON（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--per-chunk", type=int, default=4, help="每个文本块生成条数（默认 4）")
    parser.add_argument("--start", type=int, default=0, help="从第几个块开始（默认 0）")
    parser.add_argument("--end", type=int, default=None, help="到第几个块结束（默认全部）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"大模型名称（默认 {DEFAULT_MODEL}）")
    args = parser.parse_args()

    # 1. 加载文本块
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    with open(args.input, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    valid_chunks = [c for c in chunks if c.get("content", "").strip()]
    print(f"加载文本块 {len(chunks)} 个（有效 {len(valid_chunks)} 个）")

    end = args.end if args.end is not None else len(valid_chunks)
    start = max(0, args.start)
    end = min(end, len(valid_chunks))
    if start >= end:
        print("错误: start >= end，请检查参数")
        sys.exit(1)
    print(f"本次处理块范围: [{start}, {end})，共 {end - start} 块")

    # 2. 创建客户端
    client = create_instructor_client()

    # 3. 逐块生成
    all_pairs = []
    for i in range(start, end):
        chunk = valid_chunks[i]
        title = " > ".join(chunk.get("title_path", [])) or "（无标题）"
        print(f"\n[{i - start + 1}/{end - start}] 生成中: {title[:50]}...")
        pairs = generate_for_chunk(client, chunk, args.per_chunk, model=args.model)
        for p in pairs:
            p["source_chunk_id"] = i
            p["source_title_path"] = chunk.get("title_path", [])
            p["source_page"] = chunk.get("page")
        all_pairs.extend(pairs)
        print(f"  本块生成 {len(pairs)} 条，累计 {len(all_pairs)} 条")
        if i < end - 1:
            time.sleep(1)  # 限速，避免触发 API 限流

    # 4. 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    # 5. 统计摘要
    if all_pairs:
        from collections import Counter
        type_counter = Counter(p["question_type"] for p in all_pairs)
        diff_counter = Counter(p["difficulty"] for p in all_pairs)
        print("\n" + "=" * 60)
        print(f"生成完成，共 {len(all_pairs)} 条 QA 对 → {args.output}")
        print(f"问题类型分布: {dict(type_counter)}")
        print(f"难度分布: {dict(diff_counter)}")
    else:
        print("\n警告: 未生成任何 QA 对，请检查 API 配置或输入数据")


if __name__ == "__main__":
    main()

