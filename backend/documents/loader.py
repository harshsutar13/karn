from pathlib import Path

from pypdf import PdfReader
from docx import Document


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def load_docx(file_path: str) -> str:
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return "\n\n".join(paragraphs)

def load_txt(file_path: str) -> str:
    """
    Read a text file, tolerant of encoding.

    Tries UTF-8, then UTF-8-BOM, then Windows-1252 (Notepad's "ANSI"),
    then latin-1 (which never fails). Handles files saved by Word/
    Notepad in mixed encodings.
    """
    raw = Path(file_path).read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # latin-1 cannot fail, so we never reach here.
    return raw.decode("latin-1", errors="replace")

def load_document(file_path: str) -> str:
    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    raise ValueError(
        f"Unsupported document type: {extension}"
    )