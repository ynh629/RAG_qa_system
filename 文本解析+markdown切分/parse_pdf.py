import os
import re
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from markdown_splitter import MarkdownHeadingSplitter

def pdf_to_markdown_text(pdf_path):
    """使用 Marker 将 PDF 转换为 Markdown 文本。"""
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"disable_ocr": True},   # 电脑无显卡，暂时用不了 OCR
    )
    rendered = converter(pdf_path)
    return rendered.markdown   # 全量 Markdown 字符串

def _is_numbered_heading(content):#归一化辅助函数
    """
    判断数字序号后的内容是否为"标题"而非正文。
    标题特征：标题部分（到第一个句号/逗号/冒号为止）较短，且不含财务数据、不含多个数字列举。
    正文特征：标题部分很长，或包含小数/货币单位/百分比等财务数据，或包含多个数字列举（如年份列表）。
    """
    # 取到第一个句子结束符（。！？）或冒号为止的"标题部分"。
    # 注意：逗号若夹在数字之间（如 96,516,658.96 的千分位分隔符）不算句子分隔，
    # 因此用 (?<!\d),(?!\d) 只匹配"非数字间的逗号"。
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

def normalize_heading_levels(md_text):  #第一次没有归一化，导致二级标题识别不精确，因此导入归一化函数
    """
    将 Marker 输出的不统一标题级别，归一化为统一结构：
    - `第X节 ...`            -> `# `  （一级，章节）
    - `年度报告摘要` 文档标题  -> `# `  （一级）
    - `数字、 ...`（短标题）   -> `## `（二级）
    - `数字.数字 ...`（短标题）-> `### `（三级）
    - 其余被误标为标题的 `#`/`####` 行（句号结尾的正文、√适用等）-> 降为正文
    """
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


def parse_pdf_to_segments(pdf_path, mode="leaf", target_level=None, md_text=None):
    """
    基于 Marker 的 PDF 解析，返回结构化片段。
    参数：
        pdf_path: PDF 文件路径
        mode: "leaf" 或 "level"
        target_level: mode="level" 时指定切分的标题级别（如2表示按##切）
        md_text: 可选，已转换好的 Markdown 全文；若传入则跳过 Marker 转换
    """
    # 1. Marker 转换（仅在未传入 md_text 时执行，避免重复转换）
    if md_text is None:
        md_text = pdf_to_markdown_text(pdf_path)
    # 2. 标题归一化
    md_text = normalize_heading_levels(md_text)
    # 3. 使用 MarkdownHeadingSplitter 进行结构切片
    splitter = MarkdownHeadingSplitter(md_text)
    segments = splitter.split_by_headings(mode=mode, target_level=target_level)
    # 4. 页码占位保留（可从 Marker 的 page_blocks 补全）
    for seg in segments:
        seg["page"] = None
    return segments

if __name__ == "__main__":
    import json
    # 1. 只转换一次 Marker，结果同时用于切分和导出
    md_raw = pdf_to_markdown_text(r"c:\Users\Administrator\Desktop\python\文本解析+markdown切分\年报.pdf")
    # 2. 示例：使用 leaf 模式（最小粒度），也可以改成 mode="level", target_level=2 来按二级标题切
    segments = parse_pdf_to_segments(r"c:\Users\Administrator\Desktop\python\文本解析+markdown切分\年报.pdf", mode="leaf", md_text=md_raw)
    # 3. 导出归一化后的 Markdown 文件（用于检查）
    md_normalized = normalize_heading_levels(md_raw)
    with open(r"c:\Users\Administrator\Desktop\python\文本解析+markdown切分\output.md", "w", encoding="utf-8") as f:
        f.write(md_normalized)
    # 4. 保存结构化片段
    with open(r"c:\Users\Administrator\Desktop\python\文本解析+markdown切分\structured_segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"解析完成，共 {len(segments)} 个片段，结果保存在 structured_segments.json")



