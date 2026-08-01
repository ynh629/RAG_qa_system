# parse_pdf.py (Marker 升级版)
import os
import re
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

def pdf_to_markdown_text(pdf_path):
    """使用 Marker 将 PDF 转换为高质量 Markdown 文本。"""
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(pdf_path)
    return rendered.markdown   # 全量 Markdown 字符串

def split_markdown_by_headings(markdown_text, heading_level=2):
    """
    按指定级别的标题（如 ##）拆分 Markdown 文本为片段列表。
    每个片段保留标题和内容，同时保留页码占位（后续可从渲染结果精细提取）。
    """
    # 匹配二级标题开头的行
    pattern = rf'^(?={ "#" * heading_level }\s)'
    blocks = re.split(pattern, markdown_text, flags=re.MULTILINE)
    segments = []
    for block in blocks:
        if not block.strip():
            continue
        # 第一个 block 可能是文档开头无标题的部分，也保留
        segments.append({
            "type": "markdown",
            "content": block.strip()
        })
    return segments

def parse_pdf_to_segments(pdf_path, heading_level=2):
    """
    基于 Marker 的 PDF 解析，返回结构化片段。
    每个片段为一段以指定标题开头的 Markdown 文本。
    """
    # 1. 用 Marker 转换整个 PDF
    md_text = pdf_to_markdown_text(pdf_path)
    # 2. 按标题拆分
    segments = split_markdown_by_headings(md_text, heading_level)
    # 3. 可选：为每个片段添加页码（这里留空，后续可优化）
    for seg in segments:
        seg["page"] = None   # 占位
    return segments

if __name__ == "__main__":
    segments = parse_pdf_to_segments("年报.pdf", heading_level=2)
    # 保存为 JSON 便于查看
    import json
    with open("structured_segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"解析完成，共 {len(segments)} 个片段，结果保存在 structured_segments.json")