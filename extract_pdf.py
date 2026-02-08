import fitz # PyMuPDF
import sys

def extract_text(pdf_path, output_path):
    try:
        doc = fitz.open(pdf_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for page in doc:
                f.write(page.get_text())
        print(f"Text extracted to {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        extract_text(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python extract_pdf.py <input_pdf> <output_txt>")
