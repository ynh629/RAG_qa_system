# pdf_parser_pymupdf.py
"""PyMuPDF 文本与表格提取（备选方案，无 OCR）。"""
import json
import os

import fitz  # pymupdf
import pandas as pd


def parse_pdf_with_pymupdf(pdf_path, output_text=None, output_tables=None):
    """使用 PyMuPDF 提取 PDF 的文本与表格，默认输出到 rag_system/data/。"""
    # 默认输出目录：rag_system/data/
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    output_text = output_text or os.path.join(data_dir, "output.txt")
    output_tables = output_tables or os.path.join(data_dir, "output_tables.json")

    # 打开 PDF 文件
    doc = fitz.open(pdf_path)
    all_text = []
    all_tables = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # 提取文本
        text = page.get_text("text")
        all_text.append(f"--- 第{page_num+1}页 ---\n{text}")

        # 提取表格
        tables_on_page = page.find_tables()
        if tables_on_page:
            for table in tables_on_page:
                data = table.extract()
                df = pd.DataFrame(data[1:], columns=data[0])  # 将表格数据转换为 DataFrame
                all_tables.append({
                    "page": page_num + 1,
                    "table": df.to_dict(orient="records")  # 将 DataFrame 转换为字典列表
                })

    # 将提取的文本写入文件
    os.makedirs(os.path.dirname(os.path.abspath(output_text)), exist_ok=True)
    with open(output_text, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    # 将提取的表格写入 JSON 文件
    with open(output_tables, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, ensure_ascii=False, indent=2)
    print(f"文本已提取到 {output_text}")
    print(f"表格已提取到 {output_tables}")
    doc.close()


if __name__ == "__main__":
    _data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    parse_pdf_with_pymupdf(os.path.join(_data_dir, "年报.pdf"))