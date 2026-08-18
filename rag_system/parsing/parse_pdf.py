import os
import re
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from rag_system.parsing.markdown_splitter import MarkdownHeadingSplitter
from rag_system.common.logging_config import get_logger
from rag_system.common.exceptions import DocumentError

logger = get_logger(__name__)


def pdf_to_markdown_text(pdf_path):
    """使用 Marker 将 PDF 转换为 Markdown 文本。"""
    if not pdf_path or not os.path.exists(pdf_path):
        raise DocumentError(
            f"PDF 文件不存在: {pdf_path}",
            code="PDF_FILE_NOT_FOUND"
        )
    try:
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config={"disable_ocr": True},   # 电脑无显卡，暂时用不了 OCR
        )
        rendered = converter(pdf_path)
        return rendered
    except Exception as e:
        raise DocumentError(
            f"Marker 转换 PDF 失败: {pdf_path}",
            code="PDF_CONVERT_ERROR",
            detail=str(e)
        )


def add_page_numbers_to_segments(segments, rendered):
    """
    根据 Marker 的 metadata.table_of_contents 为每个片段标注页码。
    参数：
        segments: 切分好的片段列表（每个有 "title_path" 字段）
        rendered: Marker 返回的完整对象（含 metadata）
    """
    # 从 metadata 中提取目录（标题 -> 页码映射）
    metadata = getattr(rendered, "metadata", None) or {}
    toc = metadata.get("table_of_contents", []) if isinstance(metadata, dict) else []
    # 变量：标题 -> 页码（page_id 是 0 索引，转成 1 索引）
    title_page_map = {}
    for item in toc:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "").strip()
        page_id = item.get("page_id")
        if title and page_id is not None:
            title_page_map[title] = page_id + 1

    def _normalize(s):
        """去除多余空白，便于匹配。"""
        return " ".join(s.split())

    def find_page(title_path):
        """在 title_path 中从后往前查找匹配的标题页码。"""
        for title in reversed(title_path):
            norm_title = _normalize(title)
            # 精确匹配
            if norm_title in title_page_map:
                return title_page_map[norm_title]
            # 前缀匹配（标题可能被截断或归一化）
            for toc_title, page in title_page_map.items():
                norm_toc = _normalize(toc_title)
                if norm_title.startswith(norm_toc) or norm_toc.startswith(norm_title):
                    return page
        return None

    for seg in segments:
        seg["page"] = find_page(seg.get("title_path", []))

    return segments


def _is_numbered_heading(content):  # 归一化辅助函数
    """
    判断数字序号后的内容是否为"标题"而非正文。
    标题特征：标题部分（到第一个句号/逗号/冒号为止）较短，且不含财务数据、不含多个数字列举。
    正文特征：标题部分很长，或包含小数/货币单位/百分比等财务数据，或包含多个数字列举（如年份列表）。
    """
    # 取到第一个句子结束符（。！？）或冒号为止的"标题部分"。
    title_part = re.split(r'[。！？:：]|(?<!\d),(?!\d)', content)[0].strip()

    # 标题部分过长（超过45字）则视为正文
    if len(title_part) > 45:
        return False
    # 标题部分含财务数据（小数、货币单位、百分比）则视为正文
    if re.search(r'\d+\.\d+|[元万亿]|%', title_part):
        return False
    # 标题部分含多个数字列举（如"2024、2025"年份列表）则视为正文
    if re.search(r'\d+[、,，]\d+', title_part):
        return False
    return True


def normalize_heading_levels(md_text):  # 第一次没有归一化，导致二级标题识别不精确，因此导入归一化函数
    """
    将 Marker 输出的不统一标题级别，归一化为统一结构：
    - `第X节 ...`            -> `# `  （一级，章节）
    - `年度报告摘要` 文档标题  -> `# `  （一级）
    - `数字、 ...`（短标题）   -> `## `（二级）
    - `数字.数字 ...`（短标题）-> `### `（三级）
    - 其余被误标为标题的 `#`/`####` 行（句号结尾的正文、√适用等）-> 降为正文
    """
    if not md_text:
        logger.warning("normalize_heading_levels 收到空文本")
        return ""

    out = []
    for line in md_text.splitlines():
        stripped = line.strip()
        # 规则0：以数字序号开头的行（无#前缀），判断是否为标题
        m0 = re.match(r'^(\d+)\s*[、.．]\s*(.*)$', stripped)
        if m0 and not stripped.startswith('#'):
            content = m0.group(2)
            if _is_numbered_heading(content):
                out.append('## ' + stripped)  # 强制添加为二级标题
            else:
                out.append(stripped)  # 正文，原样保留
            continue
        m = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if not m:
            # 普通行原样保留
            out.append(line)
            continue
        content = m.group(2)
        # 规则1：第X节 -> 一级标题
        if re.match(r'^第[一二三四五六七八九十百]+节\s', content):
            out.append('# ' + content)
            continue
        # 规则2：年度报告摘要等文档大标题 -> 一级标题
        if re.search(r'年度报告摘要', content):
            out.append('# ' + content)
            continue
        # 规则3：数字编号标题（已有#前缀），数字、 -> 二级，数字.数字 -> 三级
        heading = None
        if re.match(r'^\d+\s*[、.．]', content):
            m_num = re.match(r'^(\d+)\s*[、.．]\s*(.*)$', content)
            num_content = m_num.group(2)
            # 已有#前缀的，Marker 已识别为标题，直接按层级处理
            if re.match(r'^\d+\.\d+', content):
                heading = '### '
            else:
                heading = '## '
        if heading:
            out.append(heading + content)
            continue
        # 规则4：其余被误标为标题的行，降为正文
        out.append(content)
    return '\n'.join(out)


def parse_pdf_to_segments(pdf_path, mode="leaf", target_level=None, rendered=None):
    """
    基于 Marker 的 PDF 解析，返回结构化片段。
    参数：
        pdf_path: PDF 文件路径
        mode: "leaf" 或 "level"
        target_level: mode="level" 时指定切分的标题级别（如2表示按##切）
        rendered: 可选，已转换好的 Marker 对象；若传入则跳过 Marker 转换
    """
    # 1. Marker 转换（仅在未传入 rendered 时执行，避免重复转换）
    if rendered is None:
        rendered = pdf_to_markdown_text(pdf_path)
    md_text = getattr(rendered, "markdown", None) or ""
    if not md_text.strip():
        raise DocumentError(
            "Marker 转换结果为空，PDF 可能无法解析",
            code="PDF_EMPTY_RESULT"
        )
    # 2. 标题归一化
    md_text = normalize_heading_levels(md_text)
    # 3. 使用 MarkdownHeadingSplitter 进行结构切片
    splitter = MarkdownHeadingSplitter(md_text)
    segments = splitter.split_by_headings(mode=mode, target_level=target_level)
    segments = add_page_numbers_to_segments(segments, rendered)
    logger.info("PDF 解析完成，共 %d 个片段", len(segments))
    return segments


if __name__ == "__main__":
    import json
    # rag_system 包根目录（parsing 的上一级）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    pdf_path = os.path.join(data_dir, "年报.pdf")
    output_md = os.path.join(data_dir, "output.md")
    output_json = os.path.join(data_dir, "structured_segments.json")
    # 1. 只转换一次 Marker，结果同时用于切分和导出
    rendered = pdf_to_markdown_text(pdf_path)
    # 2. 示例：使用 leaf 模式（最小粒度），也可以改成 mode="level", target_level=2 来按二级标题切
    segments = parse_pdf_to_segments(pdf_path, mode="leaf", rendered=rendered)
    # 3. 导出归一化后的 Markdown 文件（用于检查）
    md_normalized = normalize_heading_levels(rendered.markdown)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_normalized)
    # 4. 保存结构化片段
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"解析完成，共 {len(segments)} 个片段，结果保存在 {output_json}")
