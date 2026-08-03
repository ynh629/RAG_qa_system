import os
import re
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

def pdf_to_markdown_text(pdf_path):
    """使用 Marker 将 PDF 转换为 Markdown 文本。"""
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"disable_ocr": True},   # 电脑无显卡，暂时用不了 OCR
    )
    rendered = converter(pdf_path)
    return rendered.markdown   # 全量 Markdown 字符串

def normalize_heading_levels(md_text):#第一次没有归一化，导致二级标题识别不精确，因此导入归一化函数
    """
    将 Marker 输出的不统一标题级别，归一化为统一结构：
    - `第X节 ...`            -> `# `  （一级，章节）
    - `年度报告摘要` 文档标题  -> `# `  （一级）
    - `数字、 ...`（非句号结尾） -> `## `（二级）
    - `数字.数字 ...`（非句号结尾）-> `### `（三级）
    - 其余被误标为标题的 `#`/`####` 行（句号结尾的正文、√适用等）-> 降为正文
    """
    out = []
    for line in md_text.splitlines():
        stripped = line.strip()
        # 规则0：检查所有以数字序号开头的行，如果本应作为标题，
        # 无论它前面有没有#，都给它加上## 
        if re.match(r'^\d+\s*、', stripped) and not stripped.rstrip().endswith(('。', '：')):
            out.append('## ' + stripped)  # 强制添加为二级标题
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
        # 规则3：数字编号标题（不以句号结尾），数字、 -> 二级，数字.数字 -> 三级
        heading = None
        if re.match(r'^\d+\s*、', content) and not content.rstrip().endswith(('。', '：')):
            heading = '## '
        elif re.match(r'^\d+\.\d+\s*', content) and not content.rstrip().endswith(('。', '：')):
            heading = '### '
        if heading:
            out.append(heading + content)
            continue
        # 规则4：其余被误标为标题的行，降为正文
        out.append(content)
    return '\n'.join(out)

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
    md_text = pdf_to_markdown_text(pdf_path)
    md_text = normalize_heading_levels(md_text)
    segments = split_markdown_by_headings(md_text, heading_level)
    # 3. 可选：为每个片段添加页码（这里留空，后续可优化）
    for seg in segments:
        seg["page"] = None   # 占位
    return segments

if __name__ == "__main__":
    segments = parse_pdf_to_segments("年报.pdf", heading_level=2)
    # 保存为 JSON 便于查看
    import json
    with open("output.md","w",encoding="utf-8") as f:
        f.write(normalize_heading_levels(pdf_to_markdown_text("年报.pdf")))
    with open("structured_segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"解析完成，共 {len(segments)} 个片段，结果保存在 structured_segments.json")