"""
loader.py — extracts text from PDF files, page by page.
"""

from pathlib import Path
import fitz  # PyMuPDF


def load_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from a single PDF, page by page.
    Returns a list of dicts: {"source": filename, "page": page_number, "text": page_text}
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:  # skip blank pages
            pages.append({
                "source": pdf_path.name,
                "page": page_num,
                "text": text
            })

    doc.close()
    return pages


def load_all_pdfs(folder_path: str = "data/raw_pdfs") -> list[dict]:
    """
    Load and extract text from every PDF in the given folder.
    Returns a combined list of page dicts from all PDFs.
    """
    folder = Path(folder_path)
    all_pages = []

    pdf_files = sorted(folder.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {folder_path}")

    for pdf_file in pdf_files:
        pages = load_pdf(pdf_file)
        print(f"  {pdf_file.name}: {len(pages)} page(s) with text")
        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":
    pages = load_all_pdfs()
    print(f"\nTotal pages loaded: {len(pages)}")
    if pages:
        print("\n--- Sample (first page) ---")
        print(f"Source: {pages[0]['source']}, Page: {pages[0]['page']}")
        print(pages[0]['text'][:300], "...")