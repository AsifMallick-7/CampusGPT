
import fitz


def extract_pdf_pages(pdf_bytes, filename):
    """Extract text page-by-page from an uploaded PDF."""
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "text": text,
                    "page": page_number,
                    "filename": filename
                })

    return pages


def chunk_pages(pages, chunk_size=900, overlap=150):
    """Split extracted page text into overlapping chunks."""
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                    "filename": page["filename"]
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks
