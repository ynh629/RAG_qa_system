import fitz   #pymupdf
import pandas as pd
import json
def parse_pdf_with_pymupdf(pdf_path,output_text="output.txt",output_tables="output_tables.json"):
    # 打开PDF文件
    pdf_path =  "年报.pdf"
    doc = fitz.open(pdf_path)
    all_text = []
    all_tables = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 提取文本
        text = page.get_text("text")
        all_text.append(f"--- 第{page_num+1}页 ---\n{text}")
        
        # 提取表格
        tables_on_page=page.find_tables()
        if tables_on_page:
            for table in tables_on_page:
                data=table.extract()
                df=pd.DataFrame(data[1:],columns=data[0])  # 将表格数据转换为DataFrame
                all_tables.append({
                    "page": page_num + 1,
                    "table": df.to_dict(orient="records")  # 将DataFrame转换为字典列表
                })
    
    # 将提取的文本写入文件
    with open(output_text, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    
    # 将提取的表格写入JSON文件
    with open(output_tables, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, ensure_ascii=False, indent=2)
    print(f"文本已提取到 {output_text}")
    print(f"表格已提取到 {output_tables}")
    doc.close()
if __name__ == "__main__":
    parse_pdf_with_pymupdf("年报.pdf")