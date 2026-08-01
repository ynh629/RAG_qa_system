from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

def convert_pdf_to_markdown(pdf_path, output_md="output.md"):
    pdf_path = "年报.pdf"
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"disable_ocr": True},
    )
    rendered = converter(pdf_path)
    markdown_text = rendered.markdown

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"Markdown 已保存至 {output_md}")

if __name__ == "__main__":
    convert_pdf_to_markdown("年报.pdf")