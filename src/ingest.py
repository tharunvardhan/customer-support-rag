import re
from pathlib import Path

from pypdf import PdfReader

from . import config

def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    text = re.sub(r"\s+"," ",text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size,len(text))
        if end <len(text):
            window = text[start:end]
            cut = max(window.rfind("."),window.rfind("? ",window.rfind("! ")))
            if cut > size * 0.5:
                end = start + cut + 1
            else:
                space = window.rfind(" ")
                if space >size * 0.5:
                    end = start + space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf(source, name=None):
    reader = PdfReader(source)
    if name is None:
        name = Path(str(source)).name
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for piece in chunk_text(page_text):
            chunks.append({"text": piece, "source": name, "page": page_number})
    return chunks

def load_folder(folder=config.DATA_DIR):
    all_chunks = []
    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        all_chunks.extend(load_pdf(pdf_path, pdf_path.name))
    return all_chunks
                      
