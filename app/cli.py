# cli.py
"""
统一管线入口：解析 PDF → 选择切分策略 → 构建索引 → 问答

用法：
    python -m app.cli

流程：
    1. parse_pdf：将 PDF 解析为结构化片段（structured_segments.json）
    2. 用户选择切分策略：
        [0] 保持原样   —— 使用 MarkdownHeadingSplitter 标题结构切分结果
        [1] 递归字符切分 —— RecursiveCharacterTextSplitter（可调 chunk_size / overlap）
        [2] 语义切分   —— 基于句子向量相似度（可调 threshold）
    3. 构建索引（Chroma + BM25）并进入交互式问答
"""
import os
import json

from rag_system.common.logging_config import get_logger
from rag_system.common.exceptions import DocumentError
from app.config import settings

logger = get_logger(__name__)

# 数据目录与文件路径（统一由配置中心管理）
DATA_DIR = settings.DATA_DIR
PDF_PATH = settings.PDF_PATH
STRUCTURED_JSON = settings.STRUCTURED_JSON
RECURSIVE_JSON = settings.RECURSIVE_JSON
SEMANTIC_JSON = settings.SEMANTIC_JSON


def step1_parse_pdf(pdf_path: str = PDF_PATH, force: bool = False) -> str:
    """
    第 1 步：解析 PDF。
    若 structured_segments.json 已存在且不强制重新解析，则直接复用。
    返回数据文件路径。
    """
    if os.path.exists(STRUCTURED_JSON) and not force:
        print(f"已存在 {STRUCTURED_JSON}，跳过 PDF 解析")
        return STRUCTURED_JSON

    if not os.path.exists(pdf_path):
        raise DocumentError(
            f"PDF 文件不存在: {pdf_path}",
            code="PDF_FILE_NOT_FOUND"
        )

    # 延迟导入：仅在需要解析 PDF 时加载 Marker 相关依赖
    from rag_system.parsing.parse_pdf import parse_pdf_to_segments

    print(f"正在解析 PDF: {pdf_path}")
    segments = parse_pdf_to_segments(pdf_path, mode="leaf")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STRUCTURED_JSON, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"PDF 解析完成，共 {len(segments)} 个片段 → {STRUCTURED_JSON}")
    return STRUCTURED_JSON


def _input_with_default(prompt: str, default: str) -> str:
    """读取用户输入，为空时返回默认值。"""
    try:
        value = input(f"{prompt} [默认: {default}]：").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value if value else default


def step2_choose_splitter() -> tuple:
    """
    第 2 步：用户交互选择切分策略。
    返回 (method, output_json, kwargs)：
        method: "keep" | "recursive" | "semantic"
        output_json: 切分结果输出路径（method="keep" 时返回原始 JSON）
        kwargs: 切分参数
    """
    print("\n" + "=" * 60)
    print("请选择文本切分策略（parse_pdf 之后的可选步骤）：")
    print("  [0] 保持原样 —— 使用 MarkdownHeadingSplitter 标题结构切分结果")
    print("  [1] 递归字符切分 —— RecursiveCharacterTextSplitter（按分隔符递归切分）")
    print("  [2] 语义切分 —— 基于句子向量相似度切分（首次运行需下载 BGE 模型）")
    choice = _input_with_default("请输入选项 [0/1/2]", "0").strip()

    if choice == "1":
        chunk_size = int(_input_with_default("请输入 chunk_size（块大小）", "1000"))
        chunk_overlap = int(_input_with_default("请输入 chunk_overlap（块重叠）", "100"))
        return "recursive", RECURSIVE_JSON, {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

    if choice == "2":
        threshold = float(_input_with_default("请输入相似度阈值 threshold（越小块越大）", "0.5"))
        return "semantic", SEMANTIC_JSON, {"threshold": threshold}

    return "keep", STRUCTURED_JSON, {}


def step3_apply_split(method: str, output_json: str, **kwargs) -> str:
    """
    第 3 步：应用高级切分。
    若对应输出文件已存在，询问是否重新生成。
    返回实际使用的数据文件路径。
    """
    if os.path.exists(output_json):
        answer = _input_with_default(f"已存在 {output_json}，是否重新切分？[y/N]", "N").lower()
        if answer not in ("y", "yes"):
            print(f"复用已有切分结果: {output_json}")
            return output_json

    # 延迟导入：避免提前加载 sentence_transformers 等重型依赖
    from rag_system.splitting.advanced_splitting import apply_advanced_splitting

    print(f"\n正在执行 {method} 切分...")
    apply_advanced_splitting(
        segments_json_path=STRUCTURED_JSON,
        output_json_path=output_json,
        method=method,
        **kwargs,
    )
    return output_json


def step4_run_qa(data_json: str) -> None:
    """第 4 步：加载数据、构建索引并进入交互式问答。"""
    from app.qa_system import run_qa_pipeline

    print("\n" + "=" * 60)
    print("进入问答阶段")
    print("=" * 60)
    run_qa_pipeline(data_json=data_json, interactive=True)


def main():
    print("=" * 60)
    print("年报智能问答系统 — 统一管线")
    print("=" * 60)

    # Step 1: 解析 PDF
    step1_parse_pdf()

    # Step 2: 选择切分策略
    method, output_json, kwargs = step2_choose_splitter()

    # Step 3: 应用高级切分（method="keep" 时跳过）
    if method != "keep":
        output_json = step3_apply_split(method, output_json, **kwargs)
    else:
        print(f"\n使用原始结构切分结果: {STRUCTURED_JSON}")

    # Step 4: 运行问答
    step4_run_qa(output_json)


if __name__ == "__main__":
    main()
